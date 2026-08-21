// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Core AI MPSGraph Samplers
//
// GPU-accelerated token sampling for Core AI's pipelined inference engine.
// These samplers use MPSGraph's runAsync with completion handlers for
// non-blocking execution, enabling true GPU pipelining.
//
// ## Design Decisions
//
// ### Protocol-Based Architecture
// Both argmax (greedy) and composite (probabilistic) samplers conform to the
// `MPSGraphSampler` protocol, enabling runtime selection based on temperature:
// - temperature == 0: Argmax sampler (deterministic, fastest)
// - temperature > 0:  Composite sampler (probabilistic with topK/topP/minP)
//
// The factory pattern (`MPSGraphSamplerFactory`) selects the appropriate sampler
// once at generation start, with the sampler cached for the entire generation.
//
// ### Fixed Vocab Size at Compile Time
// These samplers fix the vocab size at compile time. This enables better
// MPSGraph optimization and eliminates runtime shape inference.
//
// ### Temperature at Init (Immutable)
// Temperature is baked into the TopK sampler at initialization rather than
// per-call. This matches the caching pattern where the sampler is created
// once and reused. Changing temperature requires engine reset + new sampler.
//
// ### Slice Handling for Prefill
// The `encodeWithSlice` method handles multi-token prefill scenarios by
// extracting the last token's logits using a blit encoder before sampling.
// This is critical for efficient prefill where we only need to sample from
// the final position.

// MARK: - MPSGraph Sampler Protocol

/// Protocol for GPU-based token samplers using MPSGraph.
///
/// Both argmax (greedy) and TopK (probabilistic) samplers conform to this protocol,
/// enabling a single sampler to be selected at engine init time based on configuration.
protocol MPSGraphSampler: AnyObject, Sendable {
    /// The vocabulary size this sampler was compiled for
    var vocabSize: Int { get }

    /// MTLBuffer for constrained-generation bitmasks, lazily allocated.
    /// Returns nil if the sampler does not support bitmask application.
    var bitmaskBuffer: MTLBuffer? { get throws }

    /// Encode sampling for single-token decode.
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        applyBitmask: Bool,
        completion: @escaping (Int32, Error?) -> Void
    ) throws

    /// Encode sampling with slice support for prefill.
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    )
}

extension MPSGraphSampler {
    var bitmaskBuffer: MTLBuffer? { get throws { nil } }

    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) throws {
        try encode(
            to: queue, logitsBuffer: logitsBuffer, logitsOffset: logitsOffset,
            outputBuffer: outputBuffer, outputOffset: outputOffset,
            applyBitmask: false, completion: completion)
    }
}

// MARK: - Sampler Factory

/// Factory for creating the appropriate MPSGraph sampler based on configuration.
enum MPSGraphSamplerFactory {
    /// Create a sampler appropriate for the given sampling configuration.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - vocabSize: Vocabulary size
    ///   - config: Sampling configuration (temperature determines sampler type)
    /// - Returns: An MPSGraphSampler instance
    ///
    /// Selection logic:
    /// - temperature == 0: Returns argmax sampler (greedy, deterministic)
    /// - temperature > 0: Returns composite sampler (topK + topP + minP)
    static func makeSampler(
        device: MTLDevice,
        vocabSize: Int,
        config: SamplingConfiguration
    ) throws -> any MPSGraphSampler {
        if config.temperature == 0 {
            return try MPSGraphArgmaxSampler(device: device, vocabSize: vocabSize)
        }

        // Determine effective K for the topK operation:
        // - If topK is explicitly set, use it
        // - If only topP or minP is set, use a generous window (1000)
        // - Default (just temperature): use 40
        let effectiveK: Int
        if let k = config.topK {
            effectiveK = k
        } else if config.topP != nil || config.minP != nil {
            effectiveK = min(1000, vocabSize)
        } else {
            effectiveK = 40
        }

        return try MPSGraphCompositeSampler(
            device: device,
            vocabSize: vocabSize,
            k: effectiveK,
            temperature: Float(config.temperature),
            topP: config.topP.map { Float($0) } ?? 1.0,
            minP: config.minP.map { Float($0) } ?? 0.0,
            penaltyEnabled: config.needsRepetitionPenalty
        )
    }

    /// Legacy convenience for temperature-only creation (used by tests).
    static func makeSampler(
        device: MTLDevice,
        vocabSize: Int,
        temperature: Double
    ) throws -> any MPSGraphSampler {
        let config = SamplingConfiguration(temperature: temperature)
        return try makeSampler(device: device, vocabSize: vocabSize, config: config)
    }
}

// MARK: - Bitmask Expansion Helper

/// Builds the MPSGraph subgraph that expands a packed Int32 bitmask into a Float16 logits mask.
///
/// The bitmask format matches xgrammar: each Int32 word covers 32 token IDs,
/// bit `i%32` in word `i/32` = 1 means token `i` is allowed.
///
/// Returns the masked logits tensor `[1, vocabSize]` ready for topK or argmax.
private func buildBitmaskExpansionGraph(
    graph: MPSGraph,
    logits: MPSGraphTensor,
    bitmaskPlaceholder: MPSGraphTensor,
    vocabSize: Int,
    bitmaskSize: Int
) -> MPSGraphTensor {
    // Reshape bitmask for broadcast: [bitmaskSize] -> [bitmaskSize, 1]
    let bitmask2D = graph.reshape(bitmaskPlaceholder, shape: [bitmaskSize as NSNumber, 1], name: "bitmask_2d")

    // Bit position indices: [0, 1, 2, ..., 31] shaped [1, 32]
    var bitIndicesData = (0..<32).map { Int32($0) }
    let bitIndices = bitIndicesData.withUnsafeMutableBufferPointer { buf in
        graph.constant(
            Data(buffer: buf),
            shape: [1, 32],
            dataType: .int32
        )
    }

    // Bit masks: [1, 2, 4, ..., 2^31] = 1 << bitIndices
    let one = graph.constant(1, dataType: .int32)
    let bitMasks = graph.bitwiseLeftShift(one, bitIndices, name: "bit_masks")

    // AND: [bitmaskSize, 1] & [1, 32] -> [bitmaskSize, 32] (broadcasts)
    let andResult = graph.bitwiseAND(bitmask2D, bitMasks, name: "and_result")

    // Non-zero check -> bool mask
    let zero = graph.constant(0, dataType: .int32)
    let boolMask = graph.notEqual(andResult, zero, name: "bool_mask")

    // Flatten [bitmaskSize, 32] -> [bitmaskSize * 32], slice to [vocabSize]
    let flat = graph.reshape(boolMask, shape: [bitmaskSize * 32 as NSNumber], name: "flat_mask")
    let sliced = graph.sliceTensor(flat, dimension: 0, start: 0, length: vocabSize, name: "sliced_mask")

    // Reshape to [1, vocabSize] to match logits
    let predicate = graph.reshape(sliced, shape: [1, vocabSize as NSNumber], name: "predicate")

    // Apply mask: select(allowed → logits, blocked → -65504)
    let negInf = graph.constant(-65504.0, shape: [1, vocabSize as NSNumber], dataType: .float16)
    return graph.select(predicate: predicate, trueTensor: logits, falseTensor: negInf, name: "masked_logits")
}

// MARK: - MPSGraph Argmax Sampler

/// MPSGraph-based argmax sampler using Apple's optimized reductionArgMaximum.
///
/// This sampler builds an MPSGraph with argmax operation at init time and uses
/// `runAsync` with completion handlers for non-blocking sampling.
///
/// ## Usage with Core AI's ComputeStream
/// ```swift
/// computeStream.withMetal3Queue { queue in
///     mpsGraphSampler.encode(
///         to: queue,
///         logitsBuffer: logitsBuffer,
///         vocabSize: vocabSize,
///         queryLength: 1,
///         outputBuffer: tokenBuffer,
///         completion: { token in
///             continuation.yield(token)
///         }
///     )
/// }
/// ```
final class MPSGraphArgmaxSampler: @unchecked Sendable {
    private let device: MTLDevice
    private let mpsDevice: MPSGraphDevice
    private let graph: MPSGraph
    private let inputPlaceholder: MPSGraphTensor
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable

    /// The vocabulary size this sampler was compiled for
    let vocabSize: Int

    // Pre-allocated objects reused every step to avoid ~70µs of CPU object creation.
    // MPSGraphTensorData wraps MTLBuffer references — safe to reuse when buffers match.
    private var cachedInputData: MPSGraphTensorData?
    private var cachedOutputData: MPSGraphTensorData?
    private var cachedInputBuffer: MTLBuffer?
    private var cachedOutputBuffer: MTLBuffer?

    // Constrained sampling — compiled lazily on first applyBitmask: true call.
    private var constrainedExecutable: MPSGraphExecutable?
    private var constrainedBitmaskBuffer: MTLBuffer?
    private var constrainedBitmaskData: MPSGraphTensorData?

    /// Number of Int32 words in the bitmask.
    let bitmaskSize: Int

    /// Initialize the MPSGraph argmax sampler.
    /// - Parameters:
    ///   - device: Metal device
    ///   - vocabSize: Vocabulary size (fixed for compilation)
    init(device: MTLDevice, vocabSize: Int) throws {
        guard vocabSize > 0 else {
            throw MPSGraphSamplerError.graphCompilationFailed
        }
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.vocabSize = vocabSize
        self.bitmaskSize = (vocabSize + 31) / 32

        // Build the argmax graph
        let graph = MPSGraph()
        self.graph = graph

        // Input: logits for a single token position [1, vocabSize] as Float16
        let inputPlaceholder = graph.placeholder(
            shape: [1, vocabSize as NSNumber],
            dataType: .float16,
            name: "logits"
        )
        self.inputPlaceholder = inputPlaceholder

        // Argmax along axis 1 (vocab dimension) - returns Int64
        // No reshape needed! Just reduce along the vocab dimension.
        let argmaxInt64 = graph.reductionArgMaximum(
            with: inputPlaceholder,
            axis: 1,  // axis 1 = vocab dimension in [1, vocabSize]
            name: "argmax"
        )

        // Cast to Int32 for token ID
        let outputTensor = graph.cast(
            argmaxInt64,
            to: .int32,
            name: "token_id"
        )
        self.outputTensor = outputTensor

        // Compile to executable
        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            inputPlaceholder: MPSGraphShapedType(
                shape: [1, vocabSize as NSNumber],
                dataType: .float16
            )
        ]

        let targetTensors = [outputTensor]

        let compilationDescriptor = MPSGraphCompilationDescriptor()
        // Enable optimizations
        compilationDescriptor.optimizationLevel = .level0

        self.executable = graph.compile(
            with: mpsDevice,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil,
            compilationDescriptor: compilationDescriptor
        )
    }

    // MARK: - Constrained Sampling (Lazy)

    /// Returns the bitmask buffer, allocating it on first access.
    /// The caller writes the xgrammar bitmask into this buffer before calling
    /// `encode(..., applyBitmask: true)`.
    var bitmaskBuffer: MTLBuffer? {
        get throws {
            if let buf = constrainedBitmaskBuffer { return buf }
            let byteCount = max(bitmaskSize * MemoryLayout<Int32>.size, 64)
            guard
                let buf = device.makeBuffer(
                    length: byteCount, options: .storageModeShared
                )
            else {
                throw MPSGraphSamplerError.bufferAllocationFailed
            }
            constrainedBitmaskBuffer = buf
            constrainedBitmaskData = MPSGraphTensorData(
                buf, shape: [bitmaskSize as NSNumber], dataType: .int32)
            return buf
        }
    }

    /// Lazily compile the constrained graph and return (executable, bitmaskData).
    private func ensureConstrainedResources() throws -> (MPSGraphExecutable, MPSGraphTensorData) {
        if let exec = constrainedExecutable, let data = constrainedBitmaskData {
            return (exec, data)
        }

        let _ = try bitmaskBuffer

        let cGraph = MPSGraph()
        let cLogits = cGraph.placeholder(
            shape: [1, vocabSize as NSNumber], dataType: .float16, name: "logits")
        let cBitmask = cGraph.placeholder(
            shape: [bitmaskSize as NSNumber], dataType: .int32, name: "bitmask")

        let maskedLogits = buildBitmaskExpansionGraph(
            graph: cGraph, logits: cLogits, bitmaskPlaceholder: cBitmask,
            vocabSize: vocabSize, bitmaskSize: bitmaskSize)

        let cArgmax = cGraph.reductionArgMaximum(with: maskedLogits, axis: 1, name: "argmax")
        let cOutput = cGraph.cast(cArgmax, to: .int32, name: "token_id")

        let cFeeds: [MPSGraphTensor: MPSGraphShapedType] = [
            cLogits: MPSGraphShapedType(shape: [1, vocabSize as NSNumber], dataType: .float16),
            cBitmask: MPSGraphShapedType(shape: [bitmaskSize as NSNumber], dataType: .int32),
        ]
        let desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = .level0

        let exec = cGraph.compile(
            with: mpsDevice, feeds: cFeeds,
            targetTensors: [cOutput], targetOperations: nil,
            compilationDescriptor: desc)
        constrainedExecutable = exec
        return (exec, constrainedBitmaskData!)
    }

    /// Encode argmax sampling with optional bitmask constraint.
    ///
    /// When `applyBitmask` is true, the bitmask in `bitmaskBuffer` is applied to
    /// logits before argmax — blocked tokens get -65504 and will never be selected.
    /// The caller must fill `bitmaskBuffer` before calling this.
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        applyBitmask: Bool,
        completion: @escaping (Int32, Error?) -> Void
    ) throws {
        if applyBitmask {
            guard let (constrained, bitmaskTensorData) = try? ensureConstrainedResources() else {
                completion(0, MPSGraphSamplerError.bufferAllocationFailed)
                return
            }
            let inputData = MPSGraphTensorData(
                logitsBuffer, shape: [1, vocabSize as NSNumber], dataType: .float16)
            let outputData = MPSGraphTensorData(
                outputBuffer, shape: [1 as NSNumber], dataType: .int32)
            let execDesc = MPSGraphExecutableExecutionDescriptor()
            execDesc.completionHandler = { [outputBuffer, outputOffset] (_, error) in
                if let error = error {
                    completion(0, error)
                    return
                }
                let result = outputBuffer.contents()
                    .advanced(by: outputOffset)
                    .assumingMemoryBound(to: Int32.self).pointee
                completion(result, nil)
            }
            constrained.runAsync(
                with: queue, inputs: [inputData, bitmaskTensorData],
                results: [outputData], executionDescriptor: execDesc)
        } else {
            encode(
                to: queue, logitsBuffer: logitsBuffer, logitsOffset: logitsOffset,
                outputBuffer: outputBuffer, outputOffset: outputOffset,
                completion: completion)
        }
    }

    /// Encode argmax with slice support and optional bitmask.
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        applyBitmask: Bool,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        if queryLength == 1 {
            do {
                try encode(
                    to: queue, logitsBuffer: logitsBuffer, logitsOffset: 0,
                    outputBuffer: outputBuffer, outputOffset: outputOffset,
                    applyBitmask: applyBitmask, completion: completion)
            } catch {
                completion(0, error)
            }
            return
        }
        // For prefill with bitmask, slice last token then apply constrained path
        let logitsOffset = (queryLength - 1) * vocabSize * MemoryLayout<UInt16>.size
        let sliceSize = vocabSize * MemoryLayout<UInt16>.size
        guard let tempBuffer = device.makeBuffer(length: sliceSize, options: .storageModeShared),
            let blitCmdBuffer = queue.makeCommandBuffer(),
            let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder()
        else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitEncoder.copy(
            from: logitsBuffer, sourceOffset: logitsOffset,
            to: tempBuffer, destinationOffset: 0, size: sliceSize)
        blitEncoder.endEncoding()
        blitCmdBuffer.commit()

        do {
            try encode(
                to: queue, logitsBuffer: tempBuffer, logitsOffset: 0,
                outputBuffer: outputBuffer, outputOffset: outputOffset,
                applyBitmask: applyBitmask, completion: completion)
        } catch {
            completion(0, error)
        }
    }

    /// Encode argmax sampling.
    ///
    /// This method uses MPSGraph's runAsync with a completion handler,
    /// providing non-blocking execution similar to our custom Metal kernel approach.
    ///
    /// - Parameters:
    ///   - queue: The command queue (from Core AI's ComputeStream via withMetal3Queue)
    ///   - logitsBuffer: MTLBuffer containing Float16 logits [1, queryLen, vocabSize]
    ///   - logitsOffset: Byte offset to the target token's logits
    ///   - outputBuffer: MTLBuffer to write the Int32 result
    ///   - outputOffset: Byte offset for the output
    ///   - completion: Called with the sampled token when GPU completes
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        // Reuse MPSGraphTensorData if buffers haven't changed (avoids object creation overhead)
        let inputData: MPSGraphTensorData
        if logitsBuffer === cachedInputBuffer, let cached = cachedInputData {
            inputData = cached
        } else {
            inputData = MPSGraphTensorData(
                logitsBuffer,
                shape: [1, vocabSize as NSNumber],
                dataType: .float16
            )
            cachedInputData = inputData
            cachedInputBuffer = logitsBuffer
        }

        let outputData: MPSGraphTensorData
        if outputBuffer === cachedOutputBuffer, let cached = cachedOutputData {
            outputData = cached
        } else {
            outputData = MPSGraphTensorData(
                outputBuffer,
                shape: [1 as NSNumber],
                dataType: .int32
            )
            cachedOutputData = outputData
            cachedOutputBuffer = outputBuffer
        }

        // Reuse pre-allocated execution descriptor, update completion handler
        let execDescriptor = MPSGraphExecutableExecutionDescriptor()
        execDescriptor.completionHandler = { [outputBuffer, outputOffset] (resultsDictionary, error) in
            if error != nil {
                completion(0, error)
                return
            }

            // Read result from output buffer
            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result, nil)
        }

        executable.runAsync(
            with: queue,
            inputs: [inputData],
            results: [outputData],
            executionDescriptor: execDescriptor
        )
    }

    /// Encode argmax sampling with offset support.
    ///
    /// This version handles the logits offset by using a separate command buffer
    /// and copying the relevant slice to a temporary buffer if needed.
    ///
    /// - Parameters:
    ///   - queue: The command queue
    ///   - logitsBuffer: Full logits buffer [1, queryLen, vocabSize]
    ///   - queryLength: Number of tokens in the query
    ///   - outputBuffer: Where to write the result
    ///   - outputOffset: Byte offset in output buffer
    ///   - completion: Called with sampled token
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        // Calculate offset to last token's logits
        let logitsOffset = (queryLength - 1) * vocabSize * MemoryLayout<UInt16>.size

        // For single-token decode (queryLength = 1), offset is 0 and we can use direct binding
        if queryLength == 1 {
            encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: outputOffset,
                completion: completion
            )
            return
        }

        // For multi-token (prefill), we need to handle the offset
        // Pattern: Commit blit separately, then use runAsync for sampling
        // This avoids the issue where encode() to MPSCommandBuffer commits internally

        // Create a temporary buffer for the single token's logits
        let sliceSize = vocabSize * MemoryLayout<UInt16>.size
        guard let tempBuffer = device.makeBuffer(length: sliceSize, options: .storageModeShared) else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }

        // Step 1: Create and commit blit command buffer separately
        guard let blitCmdBuffer = queue.makeCommandBuffer() else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitCmdBuffer.label = "MPSGraph Argmax Blit"

        guard let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder() else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitEncoder.copy(
            from: logitsBuffer,
            sourceOffset: logitsOffset,
            to: tempBuffer,
            destinationOffset: 0,
            size: sliceSize
        )
        blitEncoder.endEncoding()
        blitCmdBuffer.commit()  // Commit blit immediately (GPU will order operations)

        // Step 2: Use runAsync for sampling (executes after blit due to GPU queue ordering)
        let inputData = MPSGraphTensorData(
            tempBuffer,
            shape: [1, vocabSize as NSNumber],
            dataType: .float16
        )

        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: [1 as NSNumber],
            dataType: .int32
        )

        let execDescriptor = MPSGraphExecutableExecutionDescriptor()
        execDescriptor.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if error != nil {
                completion(0, error)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result, nil)
        }

        // Run async - GPU naturally orders this after the blit due to queue ordering
        executable.runAsync(
            with: queue,
            inputs: [inputData],
            results: [outputData],
            executionDescriptor: execDescriptor
        )
    }
}

// Conformance to MPSGraphSampler protocol
extension MPSGraphArgmaxSampler: MPSGraphSampler {}

// MARK: - MPSGraph Top-K Sampler

/// MPSGraph-based composite sampler with temperature, TopK, TopP, and MinP.
///
/// This sampler uses Apple's optimized `topK` operation combined with softmax
/// for probabilistic token sampling. Supports:
/// - Temperature-controlled randomness
/// - Top-K filtering for quality/diversity tradeoff
/// - Top-P (nucleus) filtering for adaptive vocabulary
/// - Min-P filtering for relative probability thresholding
///
/// ## Sampling Algorithm
/// 1. Extract Top-K logits and indices from full vocab
/// 2. Apply temperature scaling: logits / temperature
/// 3. Apply softmax to get probabilities
/// 4. Apply MinP filter: keep probs >= minP × max_prob
/// 5. Apply TopP filter: keep probs where exclusive cumsum < topP
/// 6. Re-normalize masked probabilities
/// 7. Sample using multinomial (cumsum + random comparison)
final class MPSGraphCompositeSampler: @unchecked Sendable {
    private let device: MTLDevice
    private let mpsDevice: MPSGraphDevice
    private let graph: MPSGraph

    // Graph tensors
    private let logitsPlaceholder: MPSGraphTensor
    private let penaltyPlaceholder: MPSGraphTensor?
    private let temperaturePlaceholder: MPSGraphTensor
    private let randomPlaceholder: MPSGraphTensor
    private let topPPlaceholder: MPSGraphTensor
    private let minPPlaceholder: MPSGraphTensor
    private let outputTensor: MPSGraphTensor

    private let executable: MPSGraphExecutable

    /// The vocabulary size this sampler was compiled for
    let vocabSize: Int

    /// The K value (number of top tokens to consider)
    let k: Int

    /// The temperature this sampler was configured with
    let temperature: Float

    /// The topP value (1.0 = disabled)
    let topP: Float

    /// The minP value (0.0 = disabled)
    let minP: Float

    /// Whether repetition penalty is compiled into this sampler's graph
    let penaltyEnabled: Bool

    /// Pre-allocated buffer for random value
    private let randomBuffer: MTLBuffer

    /// Pre-allocated buffer for temperature
    private let temperatureBuffer: MTLBuffer

    /// Pre-allocated buffer for topP value
    private let topPBuffer: MTLBuffer

    /// Pre-allocated buffer for minP value
    private let minPBuffer: MTLBuffer

    // Pre-allocated objects reused every step to avoid CPU object creation overhead.
    private var cachedLogitsData: MPSGraphTensorData?
    private var cachedOutputData: MPSGraphTensorData?
    private var cachedLogitsBuffer: MTLBuffer?
    private var cachedOutputBuffer: MTLBuffer?
    private let temperatureData: MPSGraphTensorData
    private let randomData: MPSGraphTensorData
    private let topPData: MPSGraphTensorData
    private let minPData: MPSGraphTensorData

    // Constrained sampling — compiled lazily on first applyBitmask: true call.
    private var constrainedExecutable: MPSGraphExecutable?
    private var constrainedBitmaskBuffer: MTLBuffer?
    private var constrainedBitmaskData: MPSGraphTensorData?

    /// Number of Int32 words in the bitmask.
    let bitmaskSize: Int

    /// Testing only: Override random value for deterministic tests.
    var testingOnlyRandomOverride: Float?

    /// Initialize the MPSGraph composite sampler.
    /// - Parameters:
    ///   - device: Metal device
    ///   - vocabSize: Vocabulary size (fixed for compilation)
    ///   - k: Number of top tokens to consider
    ///   - temperature: Sampling temperature
    ///   - topP: Nucleus sampling threshold (1.0 = disabled)
    ///   - minP: Minimum probability threshold (0.0 = disabled)
    init(
        device: MTLDevice, vocabSize: Int, k: Int = 40, temperature: Float = 1.0, topP: Float = 1.0, minP: Float = 0.0,
        penaltyEnabled: Bool = false
    )
        throws
    {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.vocabSize = vocabSize
        self.k = k
        self.temperature = temperature
        self.topP = topP
        self.minP = minP
        self.penaltyEnabled = penaltyEnabled
        self.bitmaskSize = (vocabSize + 31) / 32

        // Pre-allocate buffers
        guard let randomBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared),
            let temperatureBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared),
            let topPBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared),
            let minPBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)
        else {
            throw MPSGraphSamplerError.bufferAllocationFailed
        }
        self.randomBuffer = randomBuffer
        self.temperatureBuffer = temperatureBuffer
        self.topPBuffer = topPBuffer
        self.minPBuffer = minPBuffer

        // Build the composite sampling graph
        let graph = MPSGraph()
        self.graph = graph

        // Input: logits for a single token position [1, vocabSize] as Float16
        let logitsPlaceholder = graph.placeholder(
            shape: [1, vocabSize as NSNumber],
            dataType: .float16,
            name: "logits"
        )
        self.logitsPlaceholder = logitsPlaceholder

        if penaltyEnabled {
            let pp = graph.placeholder(
                shape: [1, vocabSize as NSNumber],
                dataType: .float16,
                name: "penalty"
            )
            self.penaltyPlaceholder = pp
        } else {
            self.penaltyPlaceholder = nil
        }

        // Temperature scalar [1]
        let temperaturePlaceholder = graph.placeholder(
            shape: [1 as NSNumber],
            dataType: .float32,
            name: "temperature"
        )
        self.temperaturePlaceholder = temperaturePlaceholder

        // Random value for sampling [1]
        let randomPlaceholder = graph.placeholder(
            shape: [1 as NSNumber],
            dataType: .float32,
            name: "random"
        )
        self.randomPlaceholder = randomPlaceholder

        // TopP threshold [1]
        let topPPlaceholder = graph.placeholder(
            shape: [1 as NSNumber],
            dataType: .float32,
            name: "topP"
        )
        self.topPPlaceholder = topPPlaceholder

        // MinP threshold [1]
        let minPPlaceholder = graph.placeholder(
            shape: [1 as NSNumber],
            dataType: .float32,
            name: "minP"
        )
        self.minPPlaceholder = minPPlaceholder

        // Cast logits to Float32 for numerical stability
        let logitsFloat32 = graph.cast(logitsPlaceholder, to: .float32, name: "logits_f32")

        // Build sampling pipeline using composable stage helpers
        let penalizedLogits: MPSGraphTensor
        if penaltyEnabled {
            penalizedLogits = Self.applyPenaltyStage(
                graph: graph, logits: logitsFloat32, penaltyTensor: penaltyPlaceholder!, name: "penalty")
        } else {
            penalizedLogits = logitsFloat32
        }

        let (topKValues, topKIndices) = Self.topKStage(
            graph: graph, logits: penalizedLogits, k: k, name: "topk")

        let scaledValues = Self.temperatureStage(
            graph: graph, values: topKValues, temperature: temperaturePlaceholder, name: "temp")

        let probabilities = Self.softmaxStage(graph: graph, values: scaledValues, name: "sm")

        let minPMask = Self.minPStage(
            graph: graph, probs: probabilities, minP: minPPlaceholder, name: "minp")

        let topPMask = Self.topPStage(
            graph: graph, probs: probabilities, topP: topPPlaceholder, name: "topp")

        let normalizedProbs = Self.maskAndNormalizeStage(
            graph: graph, probs: probabilities, masks: [minPMask, topPMask], name: "norm")

        let selectedIdx = Self.multinomialStage(
            graph: graph, probs: normalizedProbs, random: randomPlaceholder, name: "sample")

        let outputTensor = Self.gatherTokenStage(
            graph: graph, topKIndices: topKIndices, selectedIdx: selectedIdx, k: k, name: "gather")
        self.outputTensor = outputTensor

        // Compile to executable
        var feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            logitsPlaceholder: MPSGraphShapedType(shape: [1, vocabSize as NSNumber], dataType: .float16),
            temperaturePlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            randomPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            topPPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            minPPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
        ]
        if let pp = penaltyPlaceholder {
            feeds[pp] = MPSGraphShapedType(shape: [1, vocabSize as NSNumber], dataType: .float16)
        }

        let compilationDescriptor = MPSGraphCompilationDescriptor()
        compilationDescriptor.optimizationLevel = .level0

        self.executable = graph.compile(
            with: mpsDevice,
            feeds: feeds,
            targetTensors: [outputTensor],
            targetOperations: nil,
            compilationDescriptor: compilationDescriptor
        )

        // Pre-allocate tensor data for buffers
        self.temperatureData = MPSGraphTensorData(
            temperatureBuffer,
            shape: [1 as NSNumber],
            dataType: .float32
        )
        self.randomData = MPSGraphTensorData(
            randomBuffer,
            shape: [1 as NSNumber],
            dataType: .float32
        )
        self.topPData = MPSGraphTensorData(
            topPBuffer,
            shape: [1 as NSNumber],
            dataType: .float32
        )
        self.minPData = MPSGraphTensorData(
            minPBuffer,
            shape: [1 as NSNumber],
            dataType: .float32
        )
    }

    // MARK: - Constrained Sampling (Lazy)

    /// Returns the bitmask buffer, allocating it on first access.
    var bitmaskBuffer: MTLBuffer? {
        get throws {
            if let buf = constrainedBitmaskBuffer { return buf }
            guard
                let buf = device.makeBuffer(
                    length: bitmaskSize * MemoryLayout<Int32>.size,
                    options: .storageModeShared
                )
            else {
                throw MPSGraphSamplerError.bufferAllocationFailed
            }
            constrainedBitmaskBuffer = buf
            constrainedBitmaskData = MPSGraphTensorData(
                buf, shape: [bitmaskSize as NSNumber], dataType: .int32)
            return buf
        }
    }

    /// Lazily compile the constrained composite graph.
    private func ensureConstrainedResources() throws -> (MPSGraphExecutable, MPSGraphTensorData) {
        if let exec = constrainedExecutable, let data = constrainedBitmaskData {
            return (exec, data)
        }

        let _ = try bitmaskBuffer

        let cGraph = MPSGraph()
        let cLogits = cGraph.placeholder(shape: [1, vocabSize as NSNumber], dataType: .float16, name: "logits")
        let cTemp = cGraph.placeholder(shape: [1 as NSNumber], dataType: .float32, name: "temperature")
        let cRandom = cGraph.placeholder(shape: [1 as NSNumber], dataType: .float32, name: "random")
        let cTopP = cGraph.placeholder(shape: [1 as NSNumber], dataType: .float32, name: "topP")
        let cMinP = cGraph.placeholder(shape: [1 as NSNumber], dataType: .float32, name: "minP")
        let cBitmask = cGraph.placeholder(shape: [bitmaskSize as NSNumber], dataType: .int32, name: "bitmask")

        let cMaskedLogits = buildBitmaskExpansionGraph(
            graph: cGraph, logits: cLogits, bitmaskPlaceholder: cBitmask,
            vocabSize: vocabSize, bitmaskSize: bitmaskSize)

        let cLogitsF32 = cGraph.cast(cMaskedLogits, to: .float32, name: "logits_f32")
        let cTopK = cGraph.topK(cLogitsF32, k: k, name: "topk")
        let cScaled = cGraph.division(cTopK[0], cTemp, name: "scaled")
        let cProbs = cGraph.softMax(with: cScaled, axis: 1, name: "probs")

        let cMaxProb = cGraph.sliceTensor(cProbs, dimension: 1, start: 0, length: 1, name: "max_prob")
        let cMinPThreshold = cGraph.multiplication(cMinP, cMaxProb, name: "minp_threshold")
        let cMinPMask = cGraph.greaterThanOrEqualTo(cProbs, cMinPThreshold, name: "minp_mask")

        let cExclCumsum = cGraph.cumulativeSum(cProbs, axis: 1, exclusive: true, reverse: false, name: "excl_cumsum")
        let cTopPMask = cGraph.lessThan(cExclCumsum, cTopP, name: "topp_mask")

        let cCombinedMask = cGraph.logicalAND(cMinPMask, cTopPMask, name: "combined_mask")
        let cMaskF = cGraph.cast(cCombinedMask, to: .float32, name: "mask_float")
        let cMaskedProbs = cGraph.multiplication(cProbs, cMaskF, name: "masked_probs")
        let cSumMasked = cGraph.reductionSum(with: cMaskedProbs, axis: 1, name: "sum_masked")
        let cEpsilon = cGraph.constant(1e-10, dataType: .float32)
        let cSafeDenom = cGraph.maximum(cSumMasked, cEpsilon, name: "safe_denom")
        let cNormProbs = cGraph.division(cMaskedProbs, cSafeDenom, name: "normalized_probs")

        let cCumsum = cGraph.cumulativeSum(cNormProbs, axis: 1, exclusive: false, reverse: false, name: "cumsum")
        let cSelMask = cGraph.greaterThanOrEqualTo(cCumsum, cRandom, name: "selection_mask")
        let cSelMaskF = cGraph.cast(cSelMask, to: .float32, name: "selection_mask_float")
        let cSelIdx = cGraph.reductionArgMaximum(with: cSelMaskF, axis: 1, name: "selected_idx")
        let cSelI32 = cGraph.cast(cSelIdx, to: .int32, name: "selected_idx_i32")
        let cIndFlat = cGraph.reshape(cTopK[1], shape: [k as NSNumber], name: "indices_flat")
        let cSelFlat = cGraph.reshape(cSelI32, shape: [1 as NSNumber], name: "selected_flat")
        let cOutput = cGraph.gatherAlongAxis(0, updates: cIndFlat, indices: cSelFlat, name: "token_id")

        let cFeeds: [MPSGraphTensor: MPSGraphShapedType] = [
            cLogits: MPSGraphShapedType(shape: [1, vocabSize as NSNumber], dataType: .float16),
            cTemp: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            cRandom: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            cTopP: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            cMinP: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            cBitmask: MPSGraphShapedType(shape: [bitmaskSize as NSNumber], dataType: .int32),
        ]
        let desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = .level0

        let exec = cGraph.compile(
            with: mpsDevice, feeds: cFeeds,
            targetTensors: [cOutput], targetOperations: nil,
            compilationDescriptor: desc)
        constrainedExecutable = exec
        return (exec, constrainedBitmaskData!)
    }

    /// Encode composite sampling with optional bitmask constraint.
    ///
    /// When `applyBitmask` is true, the bitmask in `bitmaskBuffer` masks logits
    /// before topK. The caller must fill `bitmaskBuffer` before calling this.
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        applyBitmask: Bool,
        completion: @escaping (Int32, Error?) -> Void
    ) throws {
        if applyBitmask {
            guard let (constrained, bitmaskTensorData) = try? ensureConstrainedResources() else {
                completion(0, MPSGraphSamplerError.bufferAllocationFailed)
                return
            }
            temperatureBuffer.contents().assumingMemoryBound(to: Float.self).pointee = max(temperature, 0.01)
            topPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = topP
            minPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = minP
            let randomValue = testingOnlyRandomOverride ?? Float.random(in: 0..<1)
            randomBuffer.contents().assumingMemoryBound(to: Float.self).pointee = randomValue

            let logitsData = MPSGraphTensorData(
                logitsBuffer, shape: [1, vocabSize as NSNumber], dataType: .float16)
            let outputData = MPSGraphTensorData(
                outputBuffer, shape: [1 as NSNumber], dataType: .int32)
            let execDesc = MPSGraphExecutableExecutionDescriptor()
            execDesc.completionHandler = { [outputBuffer, outputOffset] (_, error) in
                if let error = error {
                    completion(0, error)
                    return
                }
                let result = outputBuffer.contents()
                    .advanced(by: outputOffset)
                    .assumingMemoryBound(to: Int32.self).pointee
                completion(result, nil)
            }
            constrained.runAsync(
                with: queue,
                inputs: [logitsData, temperatureData, randomData, topPData, minPData, bitmaskTensorData],
                results: [outputData], executionDescriptor: execDesc)
        } else {
            encode(
                to: queue, logitsBuffer: logitsBuffer, logitsOffset: logitsOffset,
                outputBuffer: outputBuffer, outputOffset: outputOffset,
                completion: completion)
        }
    }

    /// Encode composite sampling with slice and optional bitmask.
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        applyBitmask: Bool,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        if queryLength == 1 {
            do {
                try encode(
                    to: queue, logitsBuffer: logitsBuffer, logitsOffset: 0,
                    outputBuffer: outputBuffer, outputOffset: outputOffset,
                    applyBitmask: applyBitmask, completion: completion)
            } catch {
                completion(0, error)
            }
            return
        }
        let logitsOffset = (queryLength - 1) * vocabSize * MemoryLayout<UInt16>.size
        let sliceSize = vocabSize * MemoryLayout<UInt16>.size
        guard let tempBuffer = device.makeBuffer(length: sliceSize, options: .storageModeShared),
            let blitCmdBuffer = queue.makeCommandBuffer(),
            let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder()
        else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitEncoder.copy(
            from: logitsBuffer, sourceOffset: logitsOffset,
            to: tempBuffer, destinationOffset: 0, size: sliceSize)
        blitEncoder.endEncoding()
        blitCmdBuffer.commit()

        do {
            try encode(
                to: queue, logitsBuffer: tempBuffer, logitsOffset: 0,
                outputBuffer: outputBuffer, outputOffset: outputOffset,
                applyBitmask: applyBitmask, completion: completion)
        } catch {
            completion(0, error)
        }
    }

    /// Encode sampling with repetition penalty buffer.
    /// The penalty buffer must be Float16[vocabSize] with 1.0 for unpenalized tokens.
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        penaltyBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        guard penaltyEnabled else {
            encode(
                to: queue, logitsBuffer: logitsBuffer, logitsOffset: logitsOffset,
                outputBuffer: outputBuffer, outputOffset: outputOffset, completion: completion)
            return
        }

        temperatureBuffer.contents().assumingMemoryBound(to: Float.self).pointee = max(temperature, 0.01)
        topPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = topP
        minPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = minP
        let randomValue = testingOnlyRandomOverride ?? Float.random(in: 0..<1)
        randomBuffer.contents().assumingMemoryBound(to: Float.self).pointee = randomValue

        let logitsData = MPSGraphTensorData(
            logitsBuffer, shape: [1, vocabSize as NSNumber], dataType: .float16)
        let penaltyData = MPSGraphTensorData(
            penaltyBuffer, shape: [1, vocabSize as NSNumber], dataType: .float16)
        let outputData = MPSGraphTensorData(
            outputBuffer, shape: [1 as NSNumber], dataType: .int32)

        let tensorDataMap: [MPSGraphTensor: MPSGraphTensorData] = [
            logitsPlaceholder: logitsData,
            penaltyPlaceholder!: penaltyData,
            temperaturePlaceholder: temperatureData,
            randomPlaceholder: randomData,
            topPPlaceholder: topPData,
            minPPlaceholder: minPData,
        ]
        let inputs = executable.feedTensors!.map { tensorDataMap[$0]! }

        let execDesc = MPSGraphExecutableExecutionDescriptor()
        execDesc.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if let error = error {
                completion(0, error)
                return
            }
            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self).pointee
            completion(result, nil)
        }
        executable.runAsync(
            with: queue,
            inputs: inputs,
            results: [outputData], executionDescriptor: execDesc)
    }

    /// Encode composite sampling asynchronously (protocol conformance).
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        // Write runtime values to buffers
        temperatureBuffer.contents().assumingMemoryBound(to: Float.self).pointee = max(temperature, 0.01)
        topPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = topP
        minPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = minP

        let randomValue = testingOnlyRandomOverride ?? Float.random(in: 0..<1)
        randomBuffer.contents().assumingMemoryBound(to: Float.self).pointee = randomValue

        // Reuse MPSGraphTensorData if buffers haven't changed
        let logitsData: MPSGraphTensorData
        if logitsBuffer === cachedLogitsBuffer, let cached = cachedLogitsData {
            logitsData = cached
        } else {
            logitsData = MPSGraphTensorData(
                logitsBuffer,
                shape: [1, vocabSize as NSNumber],
                dataType: .float16
            )
            cachedLogitsData = logitsData
            cachedLogitsBuffer = logitsBuffer
        }

        let outputData: MPSGraphTensorData
        if outputBuffer === cachedOutputBuffer, let cached = cachedOutputData {
            outputData = cached
        } else {
            outputData = MPSGraphTensorData(
                outputBuffer,
                shape: [1 as NSNumber],
                dataType: .int32
            )
            cachedOutputData = outputData
            cachedOutputBuffer = outputBuffer
        }

        let desc = MPSGraphExecutableExecutionDescriptor()
        desc.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if error != nil {
                completion(0, error)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result, nil)
        }

        executable.runAsync(
            with: queue,
            inputs: [logitsData, temperatureData, randomData, topPData, minPData],
            results: [outputData],
            executionDescriptor: desc
        )
    }

    /// Encode composite sampling with slice support for prefill scenarios.
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32, Error?) -> Void
    ) {
        if queryLength == 1 {
            encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: outputOffset,
                completion: completion
            )
            return
        }

        let logitsOffset = (queryLength - 1) * vocabSize * MemoryLayout<UInt16>.size
        let sliceSize = vocabSize * MemoryLayout<UInt16>.size

        guard let tempBuffer = device.makeBuffer(length: sliceSize, options: .storageModeShared) else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }

        guard let blitCmdBuffer = queue.makeCommandBuffer() else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitCmdBuffer.label = "MPSGraph Composite Blit"

        guard let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder() else {
            completion(0, MPSGraphSamplerError.bufferAllocationFailed)
            return
        }
        blitEncoder.copy(
            from: logitsBuffer,
            sourceOffset: logitsOffset,
            to: tempBuffer,
            destinationOffset: 0,
            size: sliceSize
        )
        blitEncoder.endEncoding()
        blitCmdBuffer.commit()

        // Write runtime values
        temperatureBuffer.contents().assumingMemoryBound(to: Float.self).pointee = max(self.temperature, 0.01)
        topPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = topP
        minPBuffer.contents().assumingMemoryBound(to: Float.self).pointee = minP
        let randomValue = testingOnlyRandomOverride ?? Float.random(in: 0..<1)
        randomBuffer.contents().assumingMemoryBound(to: Float.self).pointee = randomValue

        let logitsData = MPSGraphTensorData(tempBuffer, shape: [1, vocabSize as NSNumber], dataType: .float16)
        let outputData = MPSGraphTensorData(outputBuffer, shape: [1 as NSNumber], dataType: .int32)

        let prefillExecDescriptor = MPSGraphExecutableExecutionDescriptor()
        prefillExecDescriptor.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if error != nil {
                completion(0, error)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result, nil)
        }

        executable.runAsync(
            with: queue,
            inputs: [logitsData, temperatureData, randomData, topPData, minPData],
            results: [outputData],
            executionDescriptor: prefillExecDescriptor
        )
    }

    // MARK: - Graph Stage Helpers

    /// Apply repetition penalty: where(logits > 0, logits / penalty, logits * penalty)
    static func applyPenaltyStage(
        graph: MPSGraph, logits: MPSGraphTensor, penaltyTensor: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let penaltyF32 = graph.cast(penaltyTensor, to: .float32, name: "\(name)_f32")
        let zero = graph.constant(0.0, dataType: .float32)
        let positive = graph.greaterThan(logits, zero, name: "\(name)_pos")
        let divided = graph.division(logits, penaltyF32, name: "\(name)_div")
        let multiplied = graph.multiplication(logits, penaltyF32, name: "\(name)_mul")
        return graph.select(predicate: positive, trueTensor: divided, falseTensor: multiplied, name: name)
    }

    /// Extract top-K values and indices from logits.
    static func topKStage(
        graph: MPSGraph, logits: MPSGraphTensor, k: Int, name: String
    ) -> (values: MPSGraphTensor, indices: MPSGraphTensor) {
        let result = graph.topK(logits, k: k, name: name)
        return (result[0], result[1])
    }

    /// Scale values by temperature: values / temperature.
    static func temperatureStage(
        graph: MPSGraph, values: MPSGraphTensor, temperature: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        graph.division(values, temperature, name: name)
    }

    /// Softmax over the K dimension (axis 1).
    static func softmaxStage(graph: MPSGraph, values: MPSGraphTensor, name: String) -> MPSGraphTensor {
        graph.softMax(with: values, axis: 1, name: name)
    }

    /// MinP mask: probs >= minP * max_prob.
    static func minPStage(
        graph: MPSGraph, probs: MPSGraphTensor, minP: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let maxProb = graph.sliceTensor(probs, dimension: 1, start: 0, length: 1, name: "\(name)_max")
        let threshold = graph.multiplication(minP, maxProb, name: "\(name)_thr")
        return graph.greaterThanOrEqualTo(probs, threshold, name: "\(name)_mask")
    }

    /// TopP mask: exclusive_cumsum < topP.
    static func topPStage(
        graph: MPSGraph, probs: MPSGraphTensor, topP: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let cumsum = graph.cumulativeSum(probs, axis: 1, exclusive: true, reverse: false, name: "\(name)_cs")
        return graph.lessThan(cumsum, topP, name: "\(name)_mask")
    }

    /// Combine boolean masks, apply to probs, and re-normalize.
    static func maskAndNormalizeStage(
        graph: MPSGraph, probs: MPSGraphTensor, masks: [MPSGraphTensor], name: String
    ) -> MPSGraphTensor {
        var combined = masks[0]
        for i in 1..<masks.count {
            combined = graph.logicalAND(combined, masks[i], name: "\(name)_and\(i)")
        }
        let maskFloat = graph.cast(combined, to: .float32, name: "\(name)_mf")
        let masked = graph.multiplication(probs, maskFloat, name: "\(name)_masked")
        let sum = graph.reductionSum(with: masked, axis: 1, name: "\(name)_sum")
        let eps = graph.constant(1e-10, dataType: .float32)
        let safeDenom = graph.maximum(sum, eps, name: "\(name)_denom")
        return graph.division(masked, safeDenom, name: name)
    }

    /// Multinomial sampling: cumsum + random comparison → argmax of selection mask.
    static func multinomialStage(
        graph: MPSGraph, probs: MPSGraphTensor, random: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let cumsum = graph.cumulativeSum(probs, axis: 1, exclusive: false, reverse: false, name: "\(name)_cs")
        let mask = graph.greaterThanOrEqualTo(cumsum, random, name: "\(name)_sel")
        let maskFloat = graph.cast(mask, to: .float32, name: "\(name)_sf")
        return graph.reductionArgMaximum(with: maskFloat, axis: 1, name: name)
    }

    /// Gather the final token ID from topK indices using the selected position.
    static func gatherTokenStage(
        graph: MPSGraph, topKIndices: MPSGraphTensor, selectedIdx: MPSGraphTensor, k: Int, name: String
    ) -> MPSGraphTensor {
        let idxI32 = graph.cast(selectedIdx, to: .int32, name: "\(name)_i32")
        let flat = graph.reshape(topKIndices, shape: [k as NSNumber], name: "\(name)_flat")
        let idxFlat = graph.reshape(idxI32, shape: [1 as NSNumber], name: "\(name)_idx")
        return graph.gatherAlongAxis(0, updates: flat, indices: idxFlat, name: name)
    }
}

// Conformance to MPSGraphSampler protocol
extension MPSGraphCompositeSampler: MPSGraphSampler {}

// MARK: - Errors

enum MPSGraphSamplerError: Error {
    case bufferAllocationFailed
    case graphCompilationFailed
    case unsupportedDevice
}
