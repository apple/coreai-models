// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Darwin

// MARK: - Fixed NDArray State

/// Fixed-size state for non-truncatable persistent states.
/// Allocated at full size on init, zero-initialized. No capacity management needed.
public final class FixedNDArrayState: SyncStateHandler {
    public let stateNames: [String]
    public let supportsTruncation: Bool = false
    public let currentCapacity: Int = .max
    public var stateCount: Int { arrays.count }

    private var arrays: [String: NDArray]

    public init(states: [(name: String, descriptor: NDArrayDescriptor)]) {
        var arrays: [String: NDArray] = [:]
        for (name, desc) in states {
            var array = NDArray(descriptor: desc)
            zeroFillNDArray(&array)
            arrays[name] = array
        }
        self.arrays = arrays
        self.stateNames = states.map(\.name)
    }

    public func ensureCapacity(forContextLength contextLength: Int) throws -> Bool {
        false
    }

    public subscript(stateIndex index: Int) -> (name: String, array: NDArray) {
        get { (stateNames[index], arrays[stateNames[index]]!) }
        set { arrays[stateNames[index]] = newValue.array }
    }

    @_lifetime(views: borrow self)
    public func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    public func reset() {
        for name in stateNames {
            zeroFillNDArray(&arrays[name]!)
        }
    }

    public func truncate(to tokenCount: Int) {
        preconditionFailure("truncate(to:) called on non-truncatable FixedNDArrayState")
    }
}

// MARK: - Growing NDArray State

/// A KV cache that can move one batch row's entry between sequence slots.
///
/// A ragged decode step writes every row's new token at slot `max(cursors)`, then moves each
/// shorter row down to its own cursor. A byte copy is enough, because K was already rotated by
/// RoPE with the row's real position.
///
/// ```swift
/// for b in 0..<n where cursors[b] < writeSlot {
///     cache.copyRegion(row: b, fromSeq: writeSlot, toSeq: cursors[b])
/// }
/// ```
protocol RegionCopyable {
    /// Copies row `row`'s K and V at slot `fromSeq` to slot `toSeq` in every state array. Does
    /// nothing when the slots are equal.
    func copyRegion(row: Int, fromSeq: Int, toSeq: Int)
}

/// Dynamically-growing KV cache state. Starts small and doubles capacity.
public final class GrowingNDArrayState: SyncStateHandler, RegionCopyable {
    public let stateNames: [String]
    public let supportsTruncation: Bool = true
    public private(set) var currentCapacity: Int
    public var stateCount: Int { arrays.count }

    private var arrays: [String: NDArray]
    private let descriptors: [NDArrayDescriptor]
    private let maxCapacity: Int
    private let sequenceDimIndex: Int
    private let batchDimIndex: Int
    private let batchSize: Int

    public init(
        states: [(name: String, descriptor: NDArrayDescriptor)],
        initialCapacity: Int,
        maxCapacity: Int,
        batchSize: Int = 1
    ) {
        self.descriptors = states.map(\.descriptor)
        self.stateNames = states.map(\.name)
        self.batchSize = batchSize

        let firstDesc = states[0].descriptor
        // The layout is [..., seq, head_dim]. The first dynamic dim would be wrong here, because it
        // is the batch dim when batch is also dynamic.
        self.sequenceDimIndex = max(0, firstDesc.shape.count - 2)
        // Batch is the first dynamic dim other than seq, for example dim 1 in [G, B, H, seq, D].
        // A static batch dim falls back to 0, so a static [G, B, H, seq, D] cache is not supported.
        let seqDim = self.sequenceDimIndex
        self.batchDimIndex =
            firstDesc.shape.enumerated()
            .first { $0.offset != seqDim && $0.element < 0 }?.offset ?? 0

        // A static seq dim means the cache was exported at full context and cannot grow. A dynamic
        // batch dim does not make it growable.
        let sequenceExtent = firstDesc.shape[sequenceDimIndex]
        let staticCapacity = sequenceExtent >= 0 ? sequenceExtent : nil
        self.maxCapacity = staticCapacity ?? maxCapacity

        let capacity = staticCapacity ?? min(initialCapacity, maxCapacity)
        self.currentCapacity = capacity

        var arrays: [String: NDArray] = [:]
        for (name, desc) in states {
            let resolved = desc.resolvingDynamicDimensions(
                Self.resolveKVShape(
                    desc.shape, sequenceDimIndex: sequenceDimIndex, sequenceLength: capacity,
                    batchSize: batchSize))
            arrays[name] = NDArray(descriptor: resolved)
        }
        self.arrays = arrays
    }

    /// Replaces the dynamic seq dim with `sequenceLength` and any other dynamic dim with `batchSize`.
    static func resolveKVShape(
        _ shape: [Int], sequenceDimIndex: Int, sequenceLength: Int, batchSize: Int
    ) -> [Int] {
        shape.enumerated().map { index, dim in
            guard dim < 0 else { return dim }
            return index == sequenceDimIndex ? sequenceLength : batchSize
        }
    }

    public func ensureCapacity(forContextLength contextLength: Int) throws -> Bool {
        guard contextLength > currentCapacity else { return false }
        guard contextLength <= maxCapacity else {
            throw InferenceRuntimeError.invalidState(
                "Context length \(contextLength) exceeds maximum \(maxCapacity)")
        }

        var newCapacity = max(currentCapacity, 1)
        while newCapacity < contextLength {
            newCapacity = min(newCapacity * 2, maxCapacity)
        }

        let previousCapacity = currentCapacity
        for (i, name) in stateNames.enumerated() {
            let desc = descriptors[i]
            let newShape = Self.resolveKVShape(
                desc.shape, sequenceDimIndex: sequenceDimIndex, sequenceLength: newCapacity,
                batchSize: batchSize)
            let resolvedDesc = desc.resolvingDynamicDimensions(newShape)
            var newArray = NDArray(descriptor: resolvedDesc)
            _ = newArray.mutableRawView()
            copyCache(from: arrays[name]!, to: &newArray, sequenceDim: sequenceDimIndex)
            arrays[name] = newArray
        }

        currentCapacity = newCapacity
        CLILogger.log("KV cache grew: \(previousCapacity) -> \(newCapacity)")
        return true
    }

    public subscript(stateIndex index: Int) -> (name: String, array: NDArray) {
        get { (stateNames[index], arrays[stateNames[index]]!) }
        set { arrays[stateNames[index]] = newValue.array }
    }

    @_lifetime(views: borrow self)
    public func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    public func reset() {
        for name in stateNames {
            zeroFillNDArray(&arrays[name]!)
        }
    }

    public func truncate(to tokenCount: Int) {}

    // MARK: - Private

    private func copyCache(from source: NDArray, to destination: inout NDArray, sequenceDim: Int) {
        let srcShape = source.shape
        let dstShape = destination.shape
        guard let headDim = srcShape.last else { return }

        let numBlocks = srcShape[..<sequenceDim].reduce(1, *)
        let oldSeqLen = srcShape[sequenceDim]
        let copyElements = oldSeqLen * headDim
        let srcBlockStride = srcShape[sequenceDim...].reduce(1, *)
        let dstBlockStride = dstShape[sequenceDim...].reduce(1, *)

        copyBlockPrefixes(
            from: source, to: &destination, blockCount: numBlocks, sourceBlockStride: srcBlockStride,
            destinationBlockStride: dstBlockStride, runElements: copyElements)
    }

    // MARK: - Per-row copy

    func copyRegion(row: Int, fromSeq: Int, toSeq: Int) {
        guard fromSeq != toSeq else { return }
        for name in stateNames {
            Self.copyRegion(
                in: &arrays[name]!, sequenceDimIndex: sequenceDimIndex, batchDimIndex: batchDimIndex,
                row: row, fromSeq: fromSeq, toSeq: toSeq)
        }
    }

    /// Array-level form of `copyRegion(row:fromSeq:toSeq:)`. The batch axis can be at any index, for
    /// example 1 in `[G, B, H, seq, D]`.
    static func copyRegion(
        in array: inout NDArray, sequenceDimIndex: Int, batchDimIndex: Int,
        row: Int, fromSeq: Int, toSeq: Int
    ) {
        guard fromSeq != toSeq else { return }
        let shape = array.shape
        guard let headDim = shape.last else { return }
        let bases = blockBases(
            shape: shape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex, row: row)

        // A raw byte copy, so every byte-addressable scalar type works, 8-bit KV caches included.
        guard let width = stateElementByteWidth(array.scalarType) else {
            preconditionFailure("Packed sub-byte scalar type for region copy: \(array.scalarType)")
        }
        let runBytes = headDim * width
        array.mutableRawView().withUnsafeMutableBytes { raw, _, _ in
            let ptr = raw.assumingMemoryBound(to: UInt8.self)
            for base in bases {
                ptr.advanced(by: (base + toSeq * headDim) * width).update(
                    from: ptr.advanced(by: (base + fromSeq * headDim) * width), count: runBytes)
            }
        }
    }

    /// Flat offsets of the `[seq, head_dim]` blocks of batch row `row`, one per index combination of
    /// the other axes. For `[G, B, H, seq, D] = [2, 3, 2, 5, 2]` and row 1: `[20, 30, 80, 90]`.
    private static func blockBases(
        shape: [Int], batchDimIndex: Int, sequenceDimIndex: Int, row: Int
    ) -> [Int] {
        let outerCount = shape[0..<batchDimIndex].reduce(1, *)
        let innerCount = shape[(batchDimIndex + 1)..<sequenceDimIndex].reduce(1, *)
        let outerBlock = shape[batchDimIndex...].reduce(1, *)  // elements per outer index
        let batchStride = shape[(batchDimIndex + 1)...].reduce(1, *)  // elements per batch index
        let blockStride = shape[sequenceDimIndex...].reduce(1, *)  // seq * head_dim
        var bases: [Int] = []
        bases.reserveCapacity(outerCount * innerCount)
        for o in 0..<outerCount {
            let outerBase = o * outerBlock + row * batchStride
            for n in 0..<innerCount {
                bases.append(outerBase + n * blockStride)
            }
        }
        return bases
    }

    /// Copies slots `[0, count)` of row 0 of the batch-1 cache `source` into row `toRow`, leaving
    /// the other rows untouched. `source` can have a different sequence capacity, but this cache
    /// must already hold `count` slots.
    ///
    /// Admitting a request into a running batch:
    ///
    /// ```swift
    /// // Prefill the prompt alone into a batch-1 scratch cache first.
    /// _ = try batchCache.ensureCapacity(forContextLength: promptLen)
    /// batchCache.copyRowPrefix(from: scratchCache, toRow: row, count: promptLen)
    /// ```
    func copyRowPrefix(from source: GrowingNDArrayState, toRow: Int, count: Int) {
        guard count > 0 else { return }
        for name in stateNames {
            guard let src = source.arrays[name] else {
                preconditionFailure("copyRowPrefix: source missing state \(name)")
            }
            Self.copyRowPrefix(
                from: src, sourceRow: 0, to: &arrays[name]!, destRow: toRow,
                sequenceDimIndex: sequenceDimIndex, batchDimIndex: batchDimIndex, count: count)
        }
    }

    /// Array-level form of `copyRowPrefix(from:toRow:count:)`. The arrays can differ in sequence
    /// capacity and batch size, but must match in scalar type and every other dim.
    static func copyRowPrefix(
        from source: NDArray, sourceRow: Int, to destination: inout NDArray, destRow: Int,
        sequenceDimIndex: Int, batchDimIndex: Int, count: Int
    ) {
        guard count > 0 else { return }
        let srcShape = source.shape
        let dstShape = destination.shape
        guard let headDim = srcShape.last else { return }
        precondition(
            source.scalarType == destination.scalarType,
            "copyRowPrefix: scalar type mismatch \(source.scalarType) vs \(destination.scalarType)")
        let srcBases = blockBases(
            shape: srcShape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex,
            row: sourceRow)
        let dstBases = blockBases(
            shape: dstShape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex,
            row: destRow)
        precondition(srcBases.count == dstBases.count, "copyRowPrefix: block-count mismatch")
        let copyElements = count * headDim  // slots [0,count) are contiguous from each block's start

        guard let width = stateElementByteWidth(destination.scalarType) else {
            preconditionFailure("Packed sub-byte scalar type for row-prefix copy: \(destination.scalarType)")
        }
        let copyBytes = copyElements * width
        source.rawView().withUnsafeBytes { srcRaw, _, _ in
            let srcPtr = srcRaw.assumingMemoryBound(to: UInt8.self)
            destination.mutableRawView().withUnsafeMutableBytes { dstRaw, _, _ in
                let dstPtr = dstRaw.assumingMemoryBound(to: UInt8.self)
                for i in srcBases.indices {
                    dstPtr.advanced(by: dstBases[i] * width).update(
                        from: srcPtr.advanced(by: srcBases[i] * width), count: copyBytes)
                }
            }
        }
    }
}

// MARK: - Shared Utilities

/// Copies the leading `runElements` elements of each of `blockCount` blocks from
/// `source` to `destination`, whose blocks start every `sourceBlockStride` and
/// `destinationBlockStride` elements respectively.
///
/// A raw byte copy, so it works for any byte-addressable scalar type (fp16/bf16/fp32
/// and 8-bit KV caches alike); a typed view would trap on a mismatched type.
func copyBlockPrefixes(
    from source: NDArray, to destination: inout NDArray,
    blockCount: Int, sourceBlockStride: Int, destinationBlockStride: Int, runElements: Int
) {
    guard let width = stateElementByteWidth(source.scalarType) else {
        preconditionFailure("Packed sub-byte scalar type for state copy: \(source.scalarType)")
    }
    source.rawView().withUnsafeBytes { sourceBytes, _, _ in
        destination.mutableRawView().withUnsafeMutableBytes { destinationBytes, _, _ in
            for block in 0..<blockCount {
                (destinationBytes + block * destinationBlockStride * width).copyMemory(
                    from: sourceBytes + block * sourceBlockStride * width,
                    byteCount: runElements * width)
            }
        }
    }
}

func zeroFillNDArray(_ array: inout NDArray) {
    guard let width = stateElementByteWidth(array.scalarType) else {
        preconditionFailure("Packed sub-byte scalar type for state: \(array.scalarType)")
    }
    // Every supported type zeroes to an all-zero bit pattern, so a raw memset covers them
    // without the scalar-type trap of a typed view.
    let byteCount = array.shape.reduce(1, *) * width
    array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
        memset(ptr, 0, byteCount)
    }
}

/// Bytes held by `array`, from its shape and element size. Packed sub-byte arrays count as 0.
func estimatedArrayBytes(_ array: NDArray) -> Int {
    array.shape.reduce(1, *) * (stateElementByteWidth(array.scalarType) ?? 0)
}

extension SyncStateHandler {
    /// Bytes held by this handler's state arrays at their current capacity, estimated from their
    /// shapes and element sizes.
    package var estimatedBytes: Int {
        (0..<stateCount).reduce(0) { total, index in total + estimatedArrayBytes(self[stateIndex: index].array) }
    }
}

/// Bytes per element of a state, or nil for packed sub-byte types, whose element counts
/// don't map to byte ranges.
func stateElementByteWidth(_ type: NDArray.ScalarType) -> Int? {
    switch type {
    case .int2, .int3, .int4, .int5, .int6, .int7,
        .uint1, .uint2, .uint3, .uint4, .uint5, .uint6, .uint7,
        .float4e2m1fn:
        return nil
    default:
        return type.byteSize
    }
}
