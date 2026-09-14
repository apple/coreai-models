// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// The seven fixed-shape memory tensors `tracker_step` reads, reused across calls.
struct PackedMemory {
    var spatialMemory: NDArray
    var spatialMemoryPosition: NDArray
    var spatialTemporalIndex: NDArray
    var spatialValid: NDArray
    var objectPointers: NDArray
    var pointerTemporalPosition: NDArray
    var pointerValid: NDArray
}

/// Selects each object's eligible memories and packs them into the graph's fixed slots.
///
/// HF builds the tracker's memory with a variable-length `torch.cat`, so its length changes
/// every frame and per object. A traced graph takes one shape, so this uses a fixed bank plus
/// an additive key mask. Packing valid entries at the front in any order is safe because
/// cross-attention is permutation-invariant over keys, RoPE repeats the same per-slot pattern
/// across spatial slots, and masked slots contribute nothing.
///
/// `gatherMemoryFrames` and `objectPointers` are ports of `_gather_memory_frame_outputs` and
/// `_get_object_pointers` in `modeling_sam3_tracker_video.py`, so which memories are eligible
/// cannot drift from upstream even though the layout does.
@VideoSegmentationActor
final class MemoryBankPacker {
    private let shapes: VideoSegmentationEngine.Shapes
    private let parameters: VideoSegmentationParameters
    private var packed: PackedMemory
    /// Slots written on the previous call, so only those need clearing on this one.
    private var previousSpatialSlots = 0
    private var previousPointerSlots = 0
    /// Emitted at most once per video; the condition is a property of the export, not of
    /// the frame, so repeating it per frame per object would bury the log.
    private var warnedAboutPointerOverflow = false

    /// Slot capacities and per-slot element counts, all derived from the traced shapes.
    private var spatialSlotElements: Int { shapes.memoryTokenCount * shapes.memoryDim }
    private var pointerSlotElements: Int { shapes.hiddenDim }

    init(
        engine: VideoSegmentationEngine,
        shapes: VideoSegmentationEngine.Shapes,
        parameters: VideoSegmentationParameters,
        packed: PackedMemory
    ) throws {
        self.shapes = shapes
        self.parameters = parameters
        self.packed = packed

        // The spatial half of the bank is exact, not a budget: HF can populate at most
        // `max_cond_frame_num` conditioning memories plus `num_maskmem - 1` recent ones.
        // An asset exported with fewer slots would silently drop real memories, so refuse.
        let required = parameters.maxCondFrameNum + parameters.numMaskmem - 1
        guard shapes.spatialSlots >= required else {
            throw VideoSegmentationError.unsupportedGeometry(
                "Asset has \(shapes.spatialSlots) spatial memory slots but this configuration "
                    + "can produce \(required) (max_cond_frame_num \(parameters.maxCondFrameNum) + "
                    + "num_maskmem \(parameters.numMaskmem) - 1). Re-export with "
                    + "--spatial-slots \(required).")
        }
    }

    /// Allocate the bank once, at the shapes the graph was traced with.
    static func makePacked(engine: VideoSegmentationEngine) async throws -> PackedMemory {
        let step = VideoSegmentationEngine.Function.trackerStep
        return PackedMemory(
            spatialMemory: try await engine.makeInput(for: step, named: "spatial_memory"),
            spatialMemoryPosition: try await engine.makeInput(
                for: step, named: "spatial_memory_pos"),
            spatialTemporalIndex: try await engine.makeInput(for: step, named: "spatial_tpos_idx"),
            spatialValid: try await engine.makeInput(for: step, named: "spatial_valid"),
            objectPointers: try await engine.makeInput(for: step, named: "object_pointers"),
            pointerTemporalPosition: try await engine.makeInput(for: step, named: "ptr_tpos"),
            pointerValid: try await engine.makeInput(for: step, named: "ptr_valid"))
    }

    /// Fill the bank for one object on one frame and hand back the shared tensors.
    ///
    /// The returned `PackedMemory` aliases this packer's storage, so it is only valid until
    /// the next `pack` call, which is exactly the lifetime `tracker_step` needs.
    func pack(
        history: ObjectOutputHistory,
        objectIndex: Int,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool
    ) throws -> PackedMemory {
        try packSpatial(history: history, objectIndex: objectIndex, frameIndex: frameIndex, reverse: reverse)
        packPointers(
            history: history, objectIndex: objectIndex, frameIndex: frameIndex,
            totalFrames: totalFrames, reverse: reverse)
        return packed
    }

    // MARK: - Spatial memory

    private func packSpatial(
        history: ObjectOutputHistory, objectIndex: Int, frameIndex: Int, reverse: Bool
    ) throws {
        let entries = Self.gatherMemoryFrames(
            history: history, frameIndex: frameIndex, reverse: reverse, parameters: parameters)

        var temporalIndices = [Int32](repeating: 0, count: shapes.spatialSlots)
        var valid = [Float](repeating: 0, count: shapes.spatialSlots)
        var slot = 0
        for (offset, output) in entries {
            guard let output,
                let features = output.memoryFeatures,
                let position = output.memoryPositionEncoding
            else { continue }
            guard slot < shapes.spatialSlots else {
                throw VideoSegmentationError.unsupportedGeometry(
                    "Object \(objectIndex) has more than \(shapes.spatialSlots) spatial memories "
                        + "on frame \(frameIndex). Re-export with a larger --spatial-slots.")
            }
            write(features, into: &packed.spatialMemory, slot: slot)
            write(position, into: &packed.spatialMemoryPosition, slot: slot)
            temporalIndices[slot] = Int32(
                Self.temporalIndex(forOffset: offset, numMaskmem: parameters.numMaskmem))
            valid[slot] = 1
            slot += 1
        }

        // Clear whatever the previous object left in the slots this one does not use.
        if slot < previousSpatialSlots {
            let range = (slot * spatialSlotElements)..<(previousSpatialSlots * spatialSlotElements)
            clearHalfRegion(&packed.spatialMemory, range)
            clearHalfRegion(&packed.spatialMemoryPosition, range)
        }
        previousSpatialSlots = slot

        fillNDArray(&packed.spatialTemporalIndex, as: Int32.self, with: temporalIndices)
        fillFloatNDArray(&packed.spatialValid, with: valid)
    }

    /// Row of `memory_temporal_positional_encoding` a memory at `offset` should use.
    ///
    /// HF indexes it as `[relative_temporal_offset - 1]`. Conditioning frames arrive with
    /// offset 0, so that is Python's `[-1]`, the last row rather than an error. Swift has no
    /// negative indexing and the graph gathers with a plain index, so wrap into range here.
    nonisolated static func temporalIndex(forOffset offset: Int, numMaskmem: Int) -> Int {
        ((offset - 1) % numMaskmem + numMaskmem) % numMaskmem
    }

    /// Port of `Sam3TrackerVideoModel._gather_memory_frame_outputs`.
    ///
    /// Returns `(relativeTemporalOffset, output)` pairs. Conditioning frames carry offset 0;
    /// recent frames carry their distance. A `nil` output is a gap the caller skips, kept in
    /// the list rather than filtered so the offsets stay attached to the right entries.
    static func gatherMemoryFrames(
        history: ObjectOutputHistory,
        frameIndex: Int,
        reverse: Bool,
        parameters: VideoSegmentationParameters
    ) -> [(offset: Int, output: StoredFrameOutput?)] {
        let (selected, unselected) = selectClosestConditioningFrames(
            history: history, frameIndex: frameIndex, limit: parameters.maxCondFrameNum)

        var entries: [(offset: Int, output: StoredFrameOutput?)] = selected.map {
            (0, history.conditioning[$0])
        }
        // Most recent last, matching upstream's `range(num_maskmem - 1, 0, -1)`. Order is
        // irrelevant to the packed bank but is preserved so a slot-by-slot comparison
        // against a reference lines up.
        for offset in stride(from: parameters.numMaskmem - 1, to: 0, by: -1) {
            let previousFrame = reverse ? frameIndex + offset : frameIndex - offset
            let output =
                history.nonConditioning[previousFrame]
                ?? (unselected.contains(previousFrame) ? history.conditioning[previousFrame] : nil)
            entries.append((offset, output))
        }
        return entries
    }

    /// Port of `_select_closest_cond_frames`: the nearest conditioning frame before the
    /// current one, the nearest at or after it, then the next-closest until the cap.
    static func selectClosestConditioningFrames(
        history: ObjectOutputHistory, frameIndex: Int, limit: Int
    ) -> (selected: [Int], unselected: Set<Int>) {
        let all = history.conditioningOrder
        if limit == -1 || all.count <= limit {
            return (all, [])
        }
        var selected: [Int] = []
        if let before = all.filter({ $0 < frameIndex }).max() { selected.append(before) }
        if let after = all.filter({ $0 >= frameIndex }).min() { selected.append(after) }
        let remaining =
            all
            .filter { !selected.contains($0) }
            .sorted { abs($0 - frameIndex) < abs($1 - frameIndex) }
            .prefix(max(0, limit - selected.count))
        selected.append(contentsOf: remaining)
        let unselected = Set(all).subtracting(selected)
        return (selected, unselected)
    }

    // MARK: - Object pointers

    private func packPointers(
        history: ObjectOutputHistory,
        objectIndex: Int,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool
    ) {
        var (offsets, pointers, maxPointers) = Self.objectPointers(
            history: history, frameIndex: frameIndex, totalFrames: totalFrames,
            reverse: reverse, parameters: parameters)

        if pointers.count > shapes.ptrSlots {
            // The spatial half of the bank is exactly sized, but this half is a budget:
            // HF's conditioning-frame pointer branch has no cap, so a long video with
            // frequent reconditioning can exceed it. Keep the temporally closest.
            if !warnedAboutPointerOverflow {
                warnedAboutPointerOverflow = true
                CLILogger.log(
                    "Object \(objectIndex) has \(pointers.count) object pointers on frame "
                        + "\(frameIndex) but the asset has \(shapes.ptrSlots) slots; dropping the "
                        + "furthest. Re-export with a larger --ptr-slots to restore parity. "
                        + "(Reported once per video.)")
            }
            let keep =
                offsets.indices
                .sorted { abs(offsets[$0]) < abs(offsets[$1]) }
                .prefix(shapes.ptrSlots)
                .sorted()
            offsets = keep.map { offsets[$0] }
            pointers = keep.map { pointers[$0] }
        }

        // Upstream divides by `max_object_pointers_to_use - 1`, which is zero on a
        // single-frame video. Clamp rather than emit infinities into the graph.
        let maxTemporalDifference = Float(max(1, maxPointers - 1))
        var temporalPositions = [Float](repeating: 0, count: shapes.ptrSlots)
        var valid = [Float](repeating: 0, count: shapes.ptrSlots)
        for (slot, offset) in offsets.enumerated() {
            writePointer(pointers[slot], slot: slot)
            temporalPositions[slot] = Float(offset) / maxTemporalDifference
            valid[slot] = 1
        }

        if offsets.count < previousPointerSlots {
            let range = (offsets.count * pointerSlotElements)..<(previousPointerSlots * pointerSlotElements)
            clearHalfRegion(&packed.objectPointers, range)
        }
        previousPointerSlots = offsets.count

        fillFloatNDArray(&packed.pointerTemporalPosition, with: temporalPositions)
        fillFloatNDArray(&packed.pointerValid, with: valid)
    }

    /// Port of `Sam3TrackerVideoModel._get_object_pointers` in non-streaming mode.
    static func objectPointers(
        history: ObjectOutputHistory,
        frameIndex: Int,
        totalFrames: Int,
        reverse: Bool,
        parameters: VideoSegmentationParameters
    ) -> (offsets: [Int], pointers: [NDArray], maxPointers: Int) {
        let sign = reverse ? -1 : 1
        let maxPointers = min(totalFrames, parameters.maxObjectPointers)

        var offsets: [Int] = []
        var pointers: [NDArray] = []

        // Conditioning frames, in registration order, restricted to the past (or the
        // future when tracking backwards). HF does this whenever `not self.training`.
        for frame in history.conditioningOrder {
            let eligible = reverse ? frame >= frameIndex : frame <= frameIndex
            guard eligible, let output = history.conditioning[frame] else { continue }
            offsets.append((frameIndex - frame) * sign)
            pointers.append(output.objectPointer)
        }

        // Then a contiguous look-back over non-conditioning frames. The `break` on an
        // out-of-range index is upstream's, and it stops the scan rather than skipping, so
        // a video boundary truncates the pointer set.
        for difference in 1..<max(1, maxPointers) {
            let reference = reverse ? frameIndex + difference : frameIndex - difference
            if reference < 0 || reference >= totalFrames { break }
            guard let output = history.nonConditioning[reference] else { continue }
            offsets.append(difference)
            pointers.append(output.objectPointer)
        }
        return (offsets, pointers, maxPointers)
    }

    // MARK: - Slot writes

    private func write(_ payload: MemoryPayload, into array: inout NDArray, slot: Int) {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        copyIntoNDArray(
            &array, as: Float16.self, elementOffset: slot * spatialSlotElements,
            from: payload.values)
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    private func writePointer(_ pointer: NDArray, slot: Int) {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let values = readNDArray(pointer, as: Float16.self, count: pointerSlotElements)
        copyIntoNDArray(
            &packed.objectPointers, as: Float16.self,
            elementOffset: slot * pointerSlotElements, from: values)
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    private func clearHalfRegion(_ array: inout NDArray, _ range: Range<Int>) {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        clearNDArrayRegion(&array, as: Float16.self, elementRange: range, value: 0)
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }
}
