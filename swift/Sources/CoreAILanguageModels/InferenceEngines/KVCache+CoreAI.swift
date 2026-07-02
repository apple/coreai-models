// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
@preconcurrency import Metal
import MetalPerformanceShaders

// MARK: - KV Cache Protocol & Types

/// Binding-ready tensor reference for Core AI inference.
/// Stores the Metal buffer alongside its shape/strides for RawView construction.
struct TensorBinding {
    let metalBuffer: MTLBuffer
    private(set) var shape: [Int]
    private(set) var strides: [Int]
    let scalarType: NDArray.ScalarType

    init(metalBuffer: MTLBuffer, shape: [Int], strides: [Int], scalarType: NDArray.ScalarType) {
        self.metalBuffer = metalBuffer
        self.shape = shape
        self.strides = strides
        self.scalarType = scalarType
    }

    /// Convenience for pipelined engine's MutableRawView construction.
    struct Layout {
        let shape: [Int]
        let strides: [Int]
    }

    var layout: Layout {
        Layout(shape: shape, strides: strides)
    }
}

/// Protocol for KV cache strategies in Core AI engines.
/// Minimal interface - implementations handle their own complexity.
///
/// Conforming types manage key and value cache buffers for transformer inference.
/// The protocol supports both static (fixed-size) and dynamic (growing) strategies.
protocol CoreAIKVCache {
    /// Current allocated capacity (sequence length dimension).
    var currentCapacity: Int { get }

    /// Get binding-ready tensor for key cache.
    var keyBinding: TensorBinding { get }

    /// Get binding-ready tensor for value cache.
    var valueBinding: TensorBinding { get }

    /// Ensure capacity for context length. Returns true if reallocation occurred.
    /// When true is returned, executables must be rebound to the new buffers.
    ///
    /// - Parameters:
    ///   - contextLength: Required capacity in tokens
    ///   - queue: Metal command queue for blit operations (if needed)
    /// - Returns: true if buffers were reallocated and need rebinding
    /// - Throws: `KVCacheError.capacityExceeded` if static cache cannot accommodate contextLength
    mutating func ensureCapacity(forContextLength contextLength: Int, queue: MTLCommandQueue) throws -> Bool

    /// Encode pipelined expansion onto a command buffer WITHOUT committing.
    /// Returns the old key/value buffers that must be kept alive until GPU completes.
    ///
    /// This is the pipelined alternative to `ensureCapacity()` — it encodes the copy
    /// operation onto the provided command buffer so it can be pipelined with other GPU work.
    /// No `waitUntilCompleted` is called.
    ///
    /// - Parameters:
    ///   - contextLength: Required capacity in tokens
    ///   - commandBuffer: Command buffer to encode onto (not committed)
    /// - Returns: Old buffers (key, value) that must be retained until GPU completes, or nil if no growth needed
    /// - Throws: `KVCacheError.capacityExceeded` if growth is not possible
    mutating func encodePipelinedExpansion(
        forContextLength contextLength: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (oldKeyBuffer: MTLBuffer, oldValueBuffer: MTLBuffer)?
}

// MARK: - CoreAIKVCache Factory

/// Factory for creating KV cache instances based on strategy.
enum KVCacheFactory {
    /// Detect if the model supports dynamic KV cache sizing.
    ///
    /// Dynamic sizing is supported when the sequence dimension in the KV cache
    /// output shape is `-1` (dynamic). This is set when the model is exported
    /// with `--dynamic-sized-kvcache-gpu` flag.
    ///
    /// - Parameter keyReqs: Tensor requirements for key cache from graph descriptor
    /// - Returns: true if the model supports dynamic KV cache sizing
    static func isDynamicKVCache(keyReqs: NDArrayDescriptor) -> Bool {
        let seqDim = detectSequenceDim(shape: keyReqs.shape)
        let isDynamic = keyReqs.shape[seqDim] < 0
        return isDynamic
    }

    /// Auto-detect sequence dimension from tensor rank.
    /// - KV cache shape [L, B, H, S, D] (5D with layers) → seqDim = 3
    /// - KV cache shape [B, H, S, D] (4D per-layer) → seqDim = 2
    static func detectSequenceDim(shape: [Int]) -> Int {
        shape.count == 5 ? 3 : 2
    }

    /// Decode KV cache shape into human-readable structure description.
    static func describeKVCacheStructure(shape: [Int]) -> String {
        switch shape.count {
        case 2:
            return "batch=\(shape[0]) × features=\(shape[1])"
        case 3:
            return "batch=\(shape[0]) × context=\(shape[1]) × features=\(shape[2])"
        case 4:
            return "batch=\(shape[0]) × heads=\(shape[1]) × context=\(shape[2]) × head_dim=\(shape[3])"
        case 5:
            return
                "\(shape[0]) layers × \(shape[1]) batch × \(shape[2]) heads × \(shape[3]) context × \(shape[4]) head_dim"
        default:
            return shape.map(String.init).joined(separator: "×")
        }
    }

    /// Create a KV cache based on the specified options.
    ///
    /// For `.auto` strategy, this factory auto-selects:
    /// - `GrowingKVCache` for models with dynamic seqDim (exported with `--dynamic-sized-kvcache-gpu`)
    /// - `StaticKVCache` for legacy models with fixed seqDim
    ///
    /// - Parameters:
    ///   - options: Engine options containing strategy and optional size override
    ///   - device: Metal device for buffer allocation
    ///   - keyReqs: Tensor requirements for key cache
    ///   - valueReqs: Tensor requirements for value cache
    ///   - maxContextLength: Maximum context length from model config
    /// - Returns: A KV cache conforming to CoreAIKVCache protocol
    static func make(
        options: EngineOptions,
        device: MTLDevice,
        keyReqs: NDArrayDescriptor,
        valueReqs: NDArrayDescriptor,
        maxContextLength: Int
    ) throws -> any CoreAIKVCache {
        let isDynamic = isDynamicKVCache(keyReqs: keyReqs)

        // Resolve strategy: auto → growing (if dynamic) or fixedSize (if legacy)
        let resolvedStrategy: KVCacheStrategy
        switch options.kvCacheStrategy {
        case .auto:
            resolvedStrategy = isDynamic ? .growing : .fixedSize
            CLILogger.log(
                "KVCache auto-selected '\(resolvedStrategy.rawValue)' (model \(isDynamic ? "supports" : "does not support") dynamic KV cache)"
            )
        case .fixedSize, .growing, .chunked:
            resolvedStrategy = options.kvCacheStrategy
        }

        // Get size from explicit override, or use resolved strategy's default
        let size = options.kvCacheSize ?? resolvedStrategy.defaultSize(maxContextLength: maxContextLength)!

        // Validate strategy compatibility with model (only for explicit non-auto strategies)
        if options.kvCacheStrategy != .auto && options.kvCacheStrategy != .fixedSize && !isDynamic {
            throw KVCacheError.unsupportedStrategy(
                "Strategy '\(options.kvCacheStrategy.rawValue)' requires dynamic KV cache support. "
                    + "Model has fixed seqDim. Re-export with --dynamic-sized-kvcache-gpu flag."
            )
        }

        switch resolvedStrategy {
        case .auto:
            fatalError("auto should have been resolved above")
        case .fixedSize:
            return try StaticKVCache(
                device: device,
                keyReqs: keyReqs,
                valueReqs: valueReqs,
                capacity: size
            )
        case .growing:
            return try GrowingKVCache(
                device: device,
                keyReqs: keyReqs,
                valueReqs: valueReqs,
                initialCapacity: size
            )
        case .chunked:
            // Chunked not yet implemented - fall back to static
            return try StaticKVCache(
                device: device,
                keyReqs: keyReqs,
                valueReqs: valueReqs,
                capacity: size
            )
        }
    }
}

// MARK: - Static KV Cache

/// Static KV cache - allocates fixed capacity at startup.
/// This is the current default behavior, preserved for compatibility.
///
/// Use this when:
/// - Memory is not a concern
/// - Predictable allocation is required
/// - Maximum throughput is needed (no growth stalls)
struct StaticKVCache: CoreAIKVCache {
    let currentCapacity: Int

    private(set) var keyBinding: TensorBinding
    private(set) var valueBinding: TensorBinding

    /// Initialize static KV cache from tensor requirements.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation
    ///   - keyReqs: Tensor requirements for key cache (from graph descriptor)
    ///   - valueReqs: Tensor requirements for value cache (from graph descriptor)
    ///   - capacity: Override capacity (nil = use max from requirements)
    /// - Throws: If buffer allocation fails
    init(
        device: MTLDevice,
        keyReqs: NDArrayDescriptor,
        valueReqs: NDArrayDescriptor,
        capacity: Int? = nil
    ) throws {
        let seqDim = KVCacheFactory.detectSequenceDim(shape: keyReqs.shape)

        // Note: maxCapacity may be -1 for models with dynamic sequence dimension.
        // In that case, we must use the provided capacity parameter.
        let maxCapacityFromModel = keyReqs.shape[seqDim]

        // If model has dynamic seq dim (-1), use capacity directly; otherwise respect model's max
        if maxCapacityFromModel < 0 {
            guard let cap = capacity else {
                throw KVCacheError.layoutCreationFailed
            }
            self.currentCapacity = cap
        } else {
            self.currentCapacity = min(capacity ?? maxCapacityFromModel, maxCapacityFromModel)
        }

        // Build concrete shapes with adjusted sequence dimension.
        var keyShape = keyReqs.shape
        var valueShape = valueReqs.shape
        keyShape[seqDim] = self.currentCapacity
        valueShape[seqDim] = self.currentCapacity

        let keyResolved = keyReqs.resolvingDynamicDimensions(keyShape)
        let valueResolved = valueReqs.resolvingDynamicDimensions(valueShape)

        let keyByteCount = keyResolved.minimumByteCount
        let valueByteCount = valueResolved.minimumByteCount

        guard let keyBuf = device.makeBuffer(length: keyByteCount, options: .storageModeShared),
            let valueBuf = device.makeBuffer(length: valueByteCount, options: .storageModeShared)
        else {
            throw KVCacheError.allocationFailed(keyByteCount + valueByteCount)
        }

        self.keyBinding = TensorBinding(
            metalBuffer: keyBuf, shape: keyShape,
            strides: keyResolved.preferredStrides, scalarType: keyReqs.scalarType)
        self.valueBinding = TensorBinding(
            metalBuffer: valueBuf, shape: valueShape,
            strides: valueResolved.preferredStrides, scalarType: valueReqs.scalarType)

        // Log final allocation summary
        let fmt = ByteCountFormatter()
        fmt.countStyle = .memory
        let shapeDesc = KVCacheFactory.describeKVCacheStructure(shape: keyShape)
        CLILogger.log(
            "StaticKVCache allocated: \(shapeDesc), Total: \(fmt.string(fromByteCount: Int64(keyByteCount + valueByteCount)))"
        )
    }

    mutating func ensureCapacity(forContextLength contextLength: Int, queue: MTLCommandQueue) throws -> Bool {
        guard contextLength <= currentCapacity else {
            throw KVCacheError.capacityExceeded(
                needed: contextLength,
                available: currentCapacity
            )
        }
        return false  // Static cache never grows
    }

    mutating func encodePipelinedExpansion(
        forContextLength contextLength: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (oldKeyBuffer: MTLBuffer, oldValueBuffer: MTLBuffer)? {
        guard contextLength <= currentCapacity else {
            throw KVCacheError.capacityExceeded(
                needed: contextLength,
                available: currentCapacity
            )
        }
        return nil  // Static cache never grows
    }
}

// MARK: - Growing KV Cache

/// Growing KV cache - starts small, grows exponentially with blit-copy.
///
/// Memory-efficient for short conversations while supporting long contexts.
/// Growth causes a synchronous pipeline stall (~20ms) but is amortized O(log₂ N).
///
/// Use this when:
/// - Memory efficiency is important
/// - Most conversations are shorter than max context
/// - Occasional stalls are acceptable
struct GrowingKVCache: CoreAIKVCache {
    private(set) var currentCapacity: Int

    private(set) var keyBinding: TensorBinding
    private(set) var valueBinding: TensorBinding

    private let device: MTLDevice
    private let maxCapacity: Int
    private let sequenceDim: Int
    /// Store the original requirements so we can resolve preferredStrides for proper MLIR alignment
    private var keyReqsTemplate: NDArrayDescriptor
    private var valueReqsTemplate: NDArrayDescriptor
    /// Pre-allocated identity kernel for pipelined expansion (reused across growth events).
    /// Saves ~24µs per growth event by avoiding MPSNDArrayIdentity allocation.
    private let identityKernel: MPSNDArrayIdentity

    /// Initialize growing KV cache with small initial capacity.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation
    ///   - keyReqs: Tensor requirements for key cache (from graph descriptor)
    ///   - valueReqs: Tensor requirements for value cache (from graph descriptor)
    ///   - initialCapacity: Starting capacity in tokens (default 256)
    /// - Throws: If initial buffer allocation fails
    init(
        device: MTLDevice,
        keyReqs: NDArrayDescriptor,
        valueReqs: NDArrayDescriptor,
        initialCapacity: Int = 256
    ) throws {
        self.device = device
        self.keyReqsTemplate = keyReqs
        self.valueReqsTemplate = valueReqs
        self.sequenceDim = KVCacheFactory.detectSequenceDim(shape: keyReqs.shape)
        self.identityKernel = MPSNDArrayIdentity(device: device)

        // Note: maxCapacity may be -1 for models with dynamic sequence dimension.
        // In that case, we use Int.max to allow unbounded growth.
        let maxCapacityFromModel = keyReqs.shape[sequenceDim]

        // If model has dynamic seq dim (-1), allow "unlimited" growth
        self.maxCapacity = maxCapacityFromModel > 0 ? maxCapacityFromModel : Int.max
        self.currentCapacity = initialCapacity

        // Build concrete shapes with initial capacity.
        var keyShape = keyReqs.shape
        var valueShape = valueReqs.shape
        keyShape[sequenceDim] = self.currentCapacity
        valueShape[sequenceDim] = self.currentCapacity

        let keyResolved = keyReqs.resolvingDynamicDimensions(keyShape)
        let valueResolved = valueReqs.resolvingDynamicDimensions(valueShape)

        let keyByteCount = keyResolved.minimumByteCount
        let valueByteCount = valueResolved.minimumByteCount

        guard let keyBuf = device.makeBuffer(length: keyByteCount, options: .storageModeShared),
            let valueBuf = device.makeBuffer(length: valueByteCount, options: .storageModeShared)
        else {
            throw KVCacheError.allocationFailed(keyByteCount + valueByteCount)
        }

        self.keyBinding = TensorBinding(
            metalBuffer: keyBuf, shape: keyShape,
            strides: keyResolved.preferredStrides, scalarType: keyReqs.scalarType)
        self.valueBinding = TensorBinding(
            metalBuffer: valueBuf, shape: valueShape,
            strides: valueResolved.preferredStrides, scalarType: valueReqs.scalarType)

        // Log final allocation summary
        let fmt = ByteCountFormatter()
        fmt.countStyle = .memory
        let shapeDesc = KVCacheFactory.describeKVCacheStructure(shape: keyShape)
        CLILogger.log(
            "GrowingKVCache allocated (initial): \(shapeDesc), Total: \(fmt.string(fromByteCount: Int64(keyByteCount + valueByteCount)))"
        )
    }

    /// Ensure capacity for context length (asynchronous convenience wrapper).
    ///
    /// Wraps `encodePipelinedExpansion()` + commit
    /// Used by warmup and prompt pre-growth which has Core AI's compute stream.
    ///
    mutating func ensureCapacity(forContextLength contextLength: Int, queue: MTLCommandQueue) throws -> Bool {
        guard contextLength > currentCapacity else { return false }

        guard let cmd = queue.makeCommandBuffer() else {
            throw KVCacheError.allocationFailed(0)
        }

        guard (try encodePipelinedExpansion(forContextLength: contextLength, commandBuffer: cmd)) != nil else {
            return false
        }
        return true
    }

    /// Encode pipelined expansion onto a command buffer WITHOUT committing.
    ///
    /// Uses MPSNDArrayIdentity with sliced array views to encode the strided copy
    /// directly onto the command buffer. No waitUntilCompleted is called.
    ///
    /// Approach: Flattens [L,B,H] into 3D `[L*B*H, S, D]`, slices destination's S dimension,
    /// then uses a cached identity kernel to copy source → sliced destination in one operation.
    ///
    /// Returns the old buffers that must be retained (via ARC capture in completion handler)
    /// until GPU completes the copy.
    mutating func encodePipelinedExpansion(
        forContextLength contextLength: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (oldKeyBuffer: MTLBuffer, oldValueBuffer: MTLBuffer)? {
        guard contextLength > currentCapacity else { return nil }

        // Exponential 2× growth
        var newCapacity = currentCapacity
        while newCapacity < contextLength { newCapacity *= 2 }
        newCapacity = min(newCapacity, maxCapacity)

        guard newCapacity >= contextLength else {
            throw KVCacheError.capacityExceeded(needed: contextLength, available: maxCapacity)
        }
        guard newCapacity > currentCapacity else { return nil }

        // Build concrete shapes with new capacity.
        var keyShape = keyReqsTemplate.shape
        var valueShape = valueReqsTemplate.shape
        keyShape[sequenceDim] = newCapacity
        valueShape[sequenceDim] = newCapacity

        let keyResolved = keyReqsTemplate.resolvingDynamicDimensions(keyShape)
        let valueResolved = valueReqsTemplate.resolvingDynamicDimensions(valueShape)

        let newKeyByteCount = keyResolved.minimumByteCount
        let newValueByteCount = valueResolved.minimumByteCount

        guard let newKeyBuf = device.makeBuffer(length: newKeyByteCount, options: .storageModeShared),
            let newValueBuf = device.makeBuffer(length: newValueByteCount, options: .storageModeShared)
        else {
            throw KVCacheError.allocationFailed(newKeyByteCount + newValueByteCount)
        }

        // Save old buffers before updating bindings
        let oldKeyBuf = keyBinding.metalBuffer
        let oldValueBuf = valueBinding.metalBuffer

        // Extract shape dimensions: [L, B, H, S, D]
        let l = keyShape[0]
        let b = keyShape[1]
        let h = keyShape[2]
        let d = keyShape[4]
        let oldS = currentCapacity
        let newS = newCapacity
        let mpsDataType = keyReqsTemplate.scalarType.mpsDataType

        // Identity copy: MPSNDArrayIdentity with sliced 3D view
        encodePipelinedIdentity(
            commandBuffer: commandBuffer,
            oldKeyBuf: oldKeyBuf, newKeyBuf: newKeyBuf,
            oldValueBuf: oldValueBuf, newValueBuf: newValueBuf,
            l: l, b: b, h: h, oldS: oldS, newS: newS, d: d,
            mpsDataType: mpsDataType
        )
        commandBuffer.commit()

        // Update bindings to new buffers (CPU metadata only — safe before GPU executes)
        keyBinding = TensorBinding(
            metalBuffer: newKeyBuf, shape: keyShape,
            strides: keyResolved.preferredStrides, scalarType: keyReqsTemplate.scalarType)
        valueBinding = TensorBinding(
            metalBuffer: newValueBuf, shape: valueShape,
            strides: valueResolved.preferredStrides, scalarType: valueReqsTemplate.scalarType)

        let fmt = ByteCountFormatter()
        fmt.countStyle = .memory
        let shapeDesc = KVCacheFactory.describeKVCacheStructure(shape: keyShape)
        CLILogger.log(
            "GrowingKVCache pipelined grow: \(currentCapacity) → \(newCapacity), \(shapeDesc), Total: \(fmt.string(fromByteCount: Int64(newKeyByteCount + newValueByteCount)))"
        )

        currentCapacity = newCapacity
        return (oldKeyBuffer: oldKeyBuf, oldValueBuffer: oldValueBuf)
    }

    // MARK: - Private Helpers

    /// Encodes the pipelined KV cache copy using MPSNDArrayIdentity with sliced destination view.
    ///
    /// Creates a full MPSNDArray for the new buffer, slices its S dimension to [oldS],
    /// then uses the cached identityKernel to copy from old (contiguous) to sliced destination.
    /// Data lands at correct positions with gaps for the expanded S dimension.
    ///
    /// Flattens [L, B, H] into a single dimension since they're contiguous
    /// in both old and new buffers. This reduces the array from 5D to 3D
    /// `[L*B*H, S, D]`, lowering MPS dimension handling overhead.
    private func encodePipelinedIdentity(
        commandBuffer: any MTLCommandBuffer,
        oldKeyBuf: MTLBuffer, newKeyBuf: MTLBuffer,
        oldValueBuf: MTLBuffer, newValueBuf: MTLBuffer,
        l: Int, b: Int, h: Int, oldS: Int, newS: Int, d: Int,
        mpsDataType: MPSDataType
    ) {
        // Flatten L×B×H into one dimension — valid because L,B,H are contiguous in both layouts.
        // This holds when the buffer is C-order (allocated as shape.reduce * elementSize).
        // Assert the total element count matches to catch any alignment padding.
        assert(
            l * b * h * oldS * d == oldKeyBuf.length / (keyReqsTemplate.scalarType.byteSize),
            "L,B,H flattening requires C-order contiguous buffer (no alignment padding)")
        let lbh = l * b * h
        let oldShape: [NSNumber] = [lbh as NSNumber, oldS as NSNumber, d as NSNumber]
        let newShape: [NSNumber] = [lbh as NSNumber, newS as NSNumber, d as NSNumber]

        // Key cache
        let srcKeyDesc = MPSNDArrayDescriptor(dataType: mpsDataType, shape: oldShape)
        let srcKeyArray = MPSNDArray(buffer: oldKeyBuf, offset: 0, descriptor: srcKeyDesc)
        let dstKeyDesc = MPSNDArrayDescriptor(dataType: mpsDataType, shape: newShape)
        let dstKeyFullArray = MPSNDArray(buffer: newKeyBuf, offset: 0, descriptor: dstKeyDesc)

        // Slice destination S dimension: MPS reverses to [D=0, S=1, LBH=2], so S is dim 1
        let keyViewDesc = dstKeyFullArray.descriptor()
        keyViewDesc.sliceDimension(1, withSubrange: MPSDimensionSlice(start: 0, length: oldS))
        if let dstKeySlice = dstKeyFullArray.arrayView(with: keyViewDesc) {
            identityKernel.encode(
                to: commandBuffer,
                sourceArray: srcKeyArray, destinationArray: dstKeySlice)
        }

        // Value cache
        let srcValDesc = MPSNDArrayDescriptor(dataType: mpsDataType, shape: oldShape)
        let srcValArray = MPSNDArray(buffer: oldValueBuf, offset: 0, descriptor: srcValDesc)
        let dstValDesc = MPSNDArrayDescriptor(dataType: mpsDataType, shape: newShape)
        let dstValFullArray = MPSNDArray(buffer: newValueBuf, offset: 0, descriptor: dstValDesc)

        let valViewDesc = dstValFullArray.descriptor()
        valViewDesc.sliceDimension(1, withSubrange: MPSDimensionSlice(start: 0, length: oldS))
        if let dstValSlice = dstValFullArray.arrayView(with: valViewDesc) {
            identityKernel.encode(
                to: commandBuffer,
                sourceArray: srcValArray, destinationArray: dstValSlice)
        }
    }
}

// MARK: - ScalarType Extension

extension NDArray.ScalarType {
    /// Byte size for Core AI scalar types.
    var byteSize: Int {
        switch self {
        case .float16, .bfloat16: return 2
        case .float32: return 4
        case .int32, .uint32: return 4
        case .int8, .uint8, .bool: return 1
        case .int16, .uint16: return 2
        case .int64, .uint64, .float64: return 8
        case .float8e5m2, .float8e4m3fn, .float8e8m0fn: return 1
        case .cfloat16: return 4
        case .cfloat32: return 8
        case .cfloat64: return 16
        case .int128, .uint128: return 16
        // Sub-byte types - estimate 1 byte for safety
        case .int2, .int3, .int4, .int5, .int6, .int7,
            .uint1, .uint2, .uint3, .uint4, .uint5, .uint6, .uint7,
            .float4e2m1fn:
            return 1
        @unknown default: return 4
        }
    }

    /// Convert Core AI ScalarType to MPSDataType for MPSNDArray operations.
    /// Only supports types actually used in KV caches.
    fileprivate var mpsDataType: MPSDataType {
        switch self {
        case .float16: return .float16
        case .bfloat16: return .bFloat16
        case .float32: return .float32
        case .int8: return .int8
        case .float8e5m2, .float8e4m3fn, .float8e8m0fn, .float4e2m1fn:
            return .float16
        default:
            fatalError("Unsupported KV cache scalar type for MPSNDArray: \(self)")
        }
    }
}
