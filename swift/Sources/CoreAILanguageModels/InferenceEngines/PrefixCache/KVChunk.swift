// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Metal

/// Copies a span of sequence positions between a KV cache tensor and a dense chunk of bytes.
///
/// A KV state is laid out `[outer..., sequence, inner...]` — `[layers, batch, heads, positions, head_dim]`
/// for the usual 5-D cache — with strides that may be padded. A chunk holds positions
/// `[start, start + count)` densely, in the same order: every outer index, then the positions, then the
/// inner elements. So a chunk is the tensor's slice along the sequence axis, without its padding, and it
/// can go back into a cache of any capacity.
enum KVChunk {
    /// The tensor a chunk is read from or written to: its memory and how to walk it.
    struct Layout {
        let shape: [Int]
        /// In elements, as `NDArray` strides are.
        let strides: [Int]
        let elementBytes: Int
        let sequenceAxis: Int

        /// Bytes of a dense chunk of `count` positions.
        func chunkBytes(count: Int) -> Int {
            outerCount * count * innerCount * elementBytes
        }

        var outerCount: Int { shape[..<sequenceAxis].reduce(1, *) }
        var innerCount: Int { shape[(sequenceAxis + 1)...].reduce(1, *) }

        /// Whether the inner axes are dense, so one position's elements are a single contiguous run.
        var innerIsDense: Bool {
            var expected = 1
            for axis in stride(from: shape.count - 1, to: sequenceAxis, by: -1) {
                if shape[axis] > 1, strides[axis] != expected { return false }
                expected *= shape[axis]
            }
            return true
        }
    }

    /// The byte offset of each outer index's first element, in order.
    private static func outerOffsets(_ layout: Layout) -> [Int] {
        var offsets = [0]
        for axis in 0..<layout.sequenceAxis {
            let step = layout.strides[axis] * layout.elementBytes
            offsets = offsets.flatMap { base in (0..<layout.shape[axis]).map { base + $0 * step } }
        }
        return offsets
    }

    /// The byte offset of each inner element of one position, in order; one entry when they are dense.
    private static func innerOffsets(_ layout: Layout) -> [Int] {
        var offsets = [0]
        for axis in (layout.sequenceAxis + 1)..<layout.shape.count {
            let step = layout.strides[axis] * layout.elementBytes
            offsets = offsets.flatMap { base in (0..<layout.shape[axis]).map { base + $0 * step } }
        }
        return offsets
    }

    /// Positions `[start, start + count)` of the tensor at `base`, as a dense chunk.
    static func read(
        from base: UnsafeRawPointer, layout: Layout, start: Int, count: Int
    ) -> Data {
        let size = layout.chunkBytes(count: count)
        guard size > 0 else { return Data() }
        // Every byte is copied over, so skip zeroing it first.
        let destination = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16)
        copy(layout: layout, start: start, count: count) { tensorOffset, chunkOffset, bytes in
            destination.advanced(by: chunkOffset).copyMemory(
                from: base.advanced(by: tensorOffset), byteCount: bytes)
        }
        return Data(bytesNoCopy: destination, count: size, deallocator: .custom { pointer, _ in pointer.deallocate() })
    }

    /// Writes a dense chunk into positions `[start, start + count)` of the tensor at `base`.
    static func write(
        _ chunk: Data, to base: UnsafeMutableRawPointer, layout: Layout, start: Int, count: Int
    ) {
        precondition(chunk.count == layout.chunkBytes(count: count), "chunk does not fit this tensor")
        chunk.withUnsafeBytes { raw in
            guard let source = raw.baseAddress else { return }
            copy(layout: layout, start: start, count: count) { tensorOffset, chunkOffset, bytes in
                base.advanced(by: tensorOffset).copyMemory(
                    from: source.advanced(by: chunkOffset), byteCount: bytes)
            }
        }
    }

    /// Calls `move` once per contiguous run, with its tensor offset, chunk offset and length in bytes.
    private static func copy(
        layout: Layout, start: Int, count: Int, move: (Int, Int, Int) -> Void
    ) {
        precondition(
            start >= 0 && count >= 0 && start + count <= layout.shape[layout.sequenceAxis],
            "positions [\(start), \(start + count)) are outside the tensor's \(layout.shape[layout.sequenceAxis])")
        let element = layout.elementBytes
        let positionStride = layout.strides[layout.sequenceAxis] * element
        let dense = layout.innerIsDense
        let row = layout.innerCount * element
        var chunkOffset = 0
        if dense, positionStride == row {
            // Positions follow each other with no padding: one run per outer index.
            for outer in outerOffsets(layout) {
                move(outer + start * positionStride, chunkOffset, count * row)
                chunkOffset += count * row
            }
            return
        }
        let inner = dense ? [0] : innerOffsets(layout)
        let run = dense ? row : element
        for outer in outerOffsets(layout) {
            for position in start..<(start + count) {
                let rowOffset = outer + position * positionStride
                for offset in inner {
                    move(rowOffset + offset, chunkOffset, run)
                    chunkOffset += run
                }
            }
        }
    }
}

extension KVChunk.Layout {
    /// A pipelined KV binding's layout, with the sequence axis where ``KVCacheFactory`` finds it.
    init(_ binding: TensorBinding) {
        self.init(
            shape: binding.shape, strides: binding.strides,
            elementBytes: binding.scalarType.byteSize,
            sequenceAxis: KVCacheFactory.detectSequenceDim(shape: binding.shape))
    }
}
