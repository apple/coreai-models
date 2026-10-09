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

/// A KV-cache state whose per-row new-token slice can be moved between sequence slots. Ragged
/// batching writes every row's new token at the longest row's slot `max(cursor)`, then the
/// runner blits each shorter row's slice down to its own `cursor[b]` — a pure byte copy (the slot is
/// only storage; K is already RoPE-rotated by `position_ids` before the write).
public protocol RegionCopyable {
    /// Copy one sequence position's `[·, head_dim]` slice (all inner/head blocks) of a single batch
    /// `row`, from `fromSeq` to `toSeq`, in place for every state array. No-op when equal.
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
        self.maxCapacity = maxCapacity
        self.descriptors = states.map(\.descriptor)
        self.stateNames = states.map(\.name)
        self.batchSize = batchSize

        let firstDesc = states[0].descriptor
        // KV caches are [..., seq, head_dim]; the sequence dim is always second-to-last. A
        // dynamic-*batch* cache also has a dynamic batch dim, so "first dynamic dim" would wrongly
        // pick the batch dim — pin to the layout convention instead.
        self.sequenceDimIndex = max(0, firstDesc.shape.count - 2)
        // Batch is NOT the leading dim in the verified [G, B, H, seq, D] layout — it sits at a
        // non-sequence index (dim 1). Detect it as the dynamic non-seq dim so a per-row blit
        // targets the right slab. KNOWN LIMITATION: a pinned-static batch dim has no
        // dynamic marker, so the probe falls back to 0 — correct only for a leading-batch layout,
        // not a static [G,B,H,seq,D] asset (unsupported here).
        let seqDim = self.sequenceDimIndex
        self.batchDimIndex =
            firstDesc.shape.enumerated()
            .first { $0.offset != seqDim && $0.element < 0 }?.offset ?? 0

        let capacity = min(initialCapacity, maxCapacity)
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

    /// Resolve a KV-cache descriptor's dynamic dims: grow the sequence dim to `sequenceLength`, and
    /// pin every other dynamic dim (e.g. a dynamic request-batch dim) to `batchSize`. Static dims
    /// pass through unchanged. Pure/table-testable.
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

    // MARK: - Per-row blit

    /// Move one batch row's new-token K/V slice between sequence slots, in place, for every state
    /// array (after a shared write at `max(cursor)`, blit each shorter row down to its own
    /// cursor). A pure byte copy — the slot is only storage (K is RoPE-rotated by `position_ids`
    /// before the write), so this is bit-identical to writing the row at `toSeq` directly. No-op when
    /// `fromSeq == toSeq` (the longest row, and every row at equal length → zero blits).
    public func copyRegion(row: Int, fromSeq: Int, toSeq: Int) {
        guard fromSeq != toSeq else { return }
        for name in stateNames {
            Self.copyRegion(
                in: &arrays[name]!, sequenceDimIndex: sequenceDimIndex, batchDimIndex: batchDimIndex,
                row: row, fromSeq: fromSeq, toSeq: toSeq)
        }
    }

    /// Copy the `[·, head_dim]` slice at sequence position `fromSeq` to `toSeq`, for the given batch
    /// `row` only, across every outer×inner block (every axis outside batch and sequence — e.g. a
    /// leading group/layer dim and the head dims). Handles any batch-dim position (the KV layout puts
    /// batch at `batchDimIndex`, not necessarily leading). Pure/testable; `fromSeq`/`toSeq` distinct.
    static func copyRegion(
        in array: inout NDArray, sequenceDimIndex: Int, batchDimIndex: Int,
        row: Int, fromSeq: Int, toSeq: Int
    ) {
        guard fromSeq != toSeq else { return }
        let shape = array.shape
        guard let headDim = shape.last else { return }
        let bases = blockBases(
            shape: shape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex, row: row)

        switch array.scalarType {
        case .float16, .bfloat16:
            // Raw 16-bit copy (a typed Float16 view traps on a BFloat16 array); values move verbatim.
            array.mutableRawView().withUnsafeMutableBytes { raw, _, _ in
                let ptr = raw.assumingMemoryBound(to: UInt16.self)
                for base in bases {
                    ptr.advanced(by: base + toSeq * headDim).update(
                        from: ptr.advanced(by: base + fromSeq * headDim), count: headDim)
                }
            }
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for base in bases {
                    ptr.advanced(by: base + toSeq * headDim).update(
                        from: ptr.advanced(by: base + fromSeq * headDim), count: headDim)
                }
            }
        default:
            preconditionFailure("Unsupported scalar type for region copy: \(array.scalarType)")
        }
    }

    /// Flat element offsets of each `[seq, head_dim]` block (outer × inner — every axis outside the
    /// batch and sequence axes) for batch `row`. Works for any layout: `[B, H, seq, D]` (batch dim 0),
    /// `[G, B, H, seq, D]` (batch dim 1), etc.
    static func blockBases(
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

    /// Copy a freshly-prefilled row's KV prefix (sequence slots `[0, count)`, all head blocks) from a
    /// batch-1 scratch cache into batch slab `toRow` of this cache, for every state array. Used by the
    /// continuous scheduler to admit a request: prefill it alone, then blit its KV into its slab
    /// without disturbing the other live rows. Source and destination may have different sequence
    /// capacities; the batch dims differ (scratch is batch-1).
    public func copyRowPrefix(from source: GrowingNDArrayState, toRow: Int, count: Int) {
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

    /// Copy sequence slots `[0, count)` (all head blocks) of batch row `sourceRow` in `source` into
    /// row `destRow` of `destination`, in place. Source/destination can have different sequence
    /// capacities (different block strides); scalar types must match. Pure/testable.
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
        // Per (outer × inner) block, copy the contiguous [0,count) seq prefix (count*head_dim elems)
        // from source row `sourceRow` to destination row `destRow`. Source/destination may differ in
        // seq capacity (different block strides) and batch size, but share outer/inner dims.
        let srcBases = blockBases(
            shape: srcShape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex,
            row: sourceRow)
        let dstBases = blockBases(
            shape: dstShape, batchDimIndex: batchDimIndex, sequenceDimIndex: sequenceDimIndex,
            row: destRow)
        precondition(srcBases.count == dstBases.count, "copyRowPrefix: block-count mismatch")
        let copyElements = count * headDim  // slots [0,count) are contiguous from each block's start

        switch destination.scalarType {
        case .float16, .bfloat16:
            source.rawView().withUnsafeBytes { srcRaw, _, _ in
                let srcPtr = srcRaw.assumingMemoryBound(to: UInt16.self)
                destination.mutableRawView().withUnsafeMutableBytes { dstRaw, _, _ in
                    let dstPtr = dstRaw.assumingMemoryBound(to: UInt16.self)
                    for i in srcBases.indices {
                        dstPtr.advanced(by: dstBases[i]).update(
                            from: srcPtr.advanced(by: srcBases[i]), count: copyElements)
                    }
                }
            }
        case .float32:
            source.view(as: Float.self).withUnsafePointer { srcPtr, _, _ in
                var dstView = destination.mutableView(as: Float.self)
                dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                    for i in srcBases.indices {
                        dstPtr.advanced(by: dstBases[i]).update(
                            from: srcPtr.advanced(by: srcBases[i]), count: copyElements)
                    }
                }
            }
        default:
            preconditionFailure("Unsupported scalar type for row-prefix copy: \(destination.scalarType)")
        }
    }
}

// MARK: - Shared Utilities

/// Copies the leading `runElements` elements of each of `blockCount` blocks from
/// `source` to `destination`, whose blocks start every `sourceBlockStride` and
/// `destinationBlockStride` elements respectively.
///
/// A raw byte copy: a typed `view(as: Float16.self)` traps on a BFloat16 array (the
/// scalar types must match), and the element values are copied verbatim either way.
func copyBlockPrefixes(
    from source: NDArray, to destination: inout NDArray,
    blockCount: Int, sourceBlockStride: Int, destinationBlockStride: Int, runElements: Int
) {
    let width: Int
    switch source.scalarType {
    case .float16, .bfloat16: width = 2
    case .float32: width = 4
    default: preconditionFailure("Unsupported scalar type for state copy: \(source.scalarType)")
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
    let count = array.shape.reduce(1, *)
    switch array.scalarType {
    case .float16, .bfloat16:
        // Both 16-bit types zero to an all-zero bit pattern, so a raw view avoids the
        // scalar-type trap a typed `mutableView(as: Float16.self)` hits on a BFloat16 array.
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            memset(ptr, 0, count * MemoryLayout<UInt16>.stride)
        }
    case .float32:
        var view = array.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            memset(ptr, 0, count * MemoryLayout<Float>.size)
        }
    default:
        preconditionFailure("Unsupported scalar type for state: \(array.scalarType)")
    }
}
