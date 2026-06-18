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

    /// Encode sampling for single-token decode.
    ///
    /// - Parameters:
    ///   - queue: The command queue
    ///   - logitsBuffer: MTLBuffer containing Float16 logits
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
        completion: @escaping (Int32) -> Void
    )

    /// Encode sampling with slice support for prefill.
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
        completion: @escaping (Int32) -> Void
    )
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
            minP: config.minP.map { Float($0) } ?? 0.0
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

    /// Initialize the MPSGraph argmax sampler.
    /// - Parameters:
    ///   - device: Metal device
    ///   - vocabSize: Vocabulary size (fixed for compilation)
    init(device: MTLDevice, vocabSize: Int) throws {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.vocabSize = vocabSize

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
        completion: @escaping (Int32) -> Void
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
            if let error = error {
                print("MPSGraph argmax error: \(error)")
                completion(0)
                return
            }

            // Read result from output buffer
            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result)
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
        completion: @escaping (Int32) -> Void
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
            completion(0)
            return
        }

        // Step 1: Create and commit blit command buffer separately
        guard let blitCmdBuffer = queue.makeCommandBuffer() else {
            completion(0)
            return
        }
        blitCmdBuffer.label = "MPSGraph Argmax Blit"

        guard let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder() else {
            completion(0)
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

        // Set up execution descriptor with completion handler
        let execDescriptor = MPSGraphExecutableExecutionDescriptor()
        execDescriptor.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if let error = error {
                print("MPSGraph argmax error: \(error)")
                completion(0)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result)
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
    private let execDescriptor: MPSGraphExecutableExecutionDescriptor

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
    init(device: MTLDevice, vocabSize: Int, k: Int = 40, temperature: Float = 1.0, topP: Float = 1.0, minP: Float = 0.0)
        throws
    {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.vocabSize = vocabSize
        self.k = k
        self.temperature = temperature
        self.topP = topP
        self.minP = minP

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

        // Step 1: Get Top-K values and indices
        let topKResult = graph.topK(logitsFloat32, k: k, name: "topk")
        let topKValues = topKResult[0]  // [1, k] sorted descending
        let topKIndices = topKResult[1]  // [1, k] as Int32

        // Step 2: Apply temperature: values / temperature
        let scaledValues = graph.division(topKValues, temperaturePlaceholder, name: "scaled")

        // Step 3: Softmax over the K dimension (axis 1)
        let probabilities = graph.softMax(with: scaledValues, axis: 1, name: "probs")

        // Step 4: MinP filtering
        // max_prob is the first element (topK returns sorted descending)
        let maxProb = graph.sliceTensor(probabilities, dimension: 1, start: 0, length: 1, name: "max_prob")
        // threshold = minP * max_prob
        let minPThreshold = graph.multiplication(minPPlaceholder, maxProb, name: "minp_threshold")
        // mask: probs >= threshold (broadcasts [1,1] to [1,k])
        let minPMask = graph.greaterThanOrEqualTo(probabilities, minPThreshold, name: "minp_mask")

        // Step 5: TopP filtering via exclusive cumulative sum
        // exclusive_cumsum[i] = sum of probs[0..i-1], so position 0 always has value 0
        let exclusiveCumsum = graph.cumulativeSum(
            probabilities, axis: 1, exclusive: true, reverse: false, name: "excl_cumsum")
        // mask: exclusive_cumsum < topP (includes all tokens before cumsum reaches topP)
        let topPMask = graph.lessThan(exclusiveCumsum, topPPlaceholder, name: "topp_mask")

        // Step 6: Combined mask = minP AND topP
        let combinedMask = graph.logicalAND(minPMask, topPMask, name: "combined_mask")
        let maskFloat = graph.cast(combinedMask, to: .float32, name: "mask_float")

        // Step 7: Apply mask and re-normalize
        let maskedProbs = graph.multiplication(probabilities, maskFloat, name: "masked_probs")
        let sumMasked = graph.reductionSum(with: maskedProbs, axis: 1, name: "sum_masked")
        // Avoid division by zero: use max(sum, epsilon)
        let epsilon = graph.constant(1e-10, dataType: .float32)
        let safeDenominator = graph.maximum(sumMasked, epsilon, name: "safe_denom")
        let normalizedProbs = graph.division(maskedProbs, safeDenominator, name: "normalized_probs")

        // Step 8: Multinomial sampling via cumulative sum + random comparison
        let cumsum = graph.cumulativeSum(normalizedProbs, axis: 1, exclusive: false, reverse: false, name: "cumsum")
        let selectionMask = graph.greaterThanOrEqualTo(cumsum, randomPlaceholder, name: "selection_mask")
        let selectionMaskFloat = graph.cast(selectionMask, to: .float32, name: "selection_mask_float")
        let selectedIdx = graph.reductionArgMaximum(with: selectionMaskFloat, axis: 1, name: "selected_idx")

        // Step 9: Gather the token index from topKIndices
        let selectedIdxInt32 = graph.cast(selectedIdx, to: .int32, name: "selected_idx_i32")
        let indicesFlat = graph.reshape(topKIndices, shape: [k as NSNumber], name: "indices_flat")
        let selectedIdxFlat = graph.reshape(selectedIdxInt32, shape: [1 as NSNumber], name: "selected_flat")

        let outputTensor = graph.gatherAlongAxis(
            0,
            updates: indicesFlat,
            indices: selectedIdxFlat,
            name: "token_id"
        )
        self.outputTensor = outputTensor

        // Compile to executable
        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            logitsPlaceholder: MPSGraphShapedType(shape: [1, vocabSize as NSNumber], dataType: .float16),
            temperaturePlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            randomPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            topPPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
            minPPlaceholder: MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32),
        ]

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
        self.execDescriptor = MPSGraphExecutableExecutionDescriptor()
    }

    /// Encode composite sampling asynchronously (protocol conformance).
    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32) -> Void
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

        execDescriptor.completionHandler = { [outputBuffer, outputOffset] (_, error) in
            if let error = error {
                print("MPSGraph composite sampler error: \(error)")
                completion(0)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result)
        }

        executable.runAsync(
            with: queue,
            inputs: [logitsData, temperatureData, randomData, topPData, minPData],
            results: [outputData],
            executionDescriptor: execDescriptor
        )
    }

    /// Encode composite sampling with slice support for prefill scenarios.
    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32) -> Void
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
            completion(0)
            return
        }

        guard let blitCmdBuffer = queue.makeCommandBuffer() else {
            completion(0)
            return
        }
        blitCmdBuffer.label = "MPSGraph Composite Blit"

        guard let blitEncoder = blitCmdBuffer.makeBlitCommandEncoder() else {
            completion(0)
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
            if let error = error {
                print("MPSGraph composite sampler error: \(error)")
                completion(0)
                return
            }

            let result = outputBuffer.contents()
                .advanced(by: outputOffset)
                .assumingMemoryBound(to: Int32.self)
                .pointee
            completion(result)
        }

        executable.runAsync(
            with: queue,
            inputs: [logitsData, temperatureData, randomData, topPData, minPData],
            results: [outputData],
            executionDescriptor: prefillExecDescriptor
        )
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
