// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Removes tracks that never settle: ones the detector stops confirming and duplicates of an
/// earlier track.
///
/// Port of `Sam3VideoModel._process_hotstart`. Both rules fire only for tracks that first
/// appeared inside the hotstart window. Older tracks are exempt.
///
/// The keep-alive counter is a separate, always-on mechanism over the same bookkeeping. A
/// track whose counter runs out is suppressed for the frame but stays in the registry.
@VideoSegmentationActor
enum HotstartHeuristics {
    /// Mutates the session's hotstart bookkeeping and returns the ids removed on this
    /// frame.
    static func process(
        session: VideoInferenceSession,
        frameIndex: Int,
        reverse: Bool,
        detectionToMatchedTrackIDs: [Int: [Int]],
        newObjectIDs: [Int],
        emptyTrackIDs: [Int],
        unmatchedTrackIDs: [Int],
        parameters: VideoSegmentationParameters
    ) -> Set<Int> {
        var newlyRemoved: Set<Int> = []
        var suppressed = session.suppressedObjectIDsByFrame[frameIndex] ?? []
        // Everything before this frame is "outside the hotstart window".
        let hotstartBoundary =
            reverse ? frameIndex + parameters.hotstartDelay : frameIndex - parameters.hotstartDelay
        // Upstream's `obj_first_frame_idx[x]` raises for an object with no record. Here it
        // defaults to the current frame.
        func firstSeen(_ id: Int) -> Int { session.firstFrameByObjectID[id] ?? frameIndex }
        func withinHotstart(_ id: Int) -> Bool {
            reverse ? firstSeen(id) < hotstartBoundary : firstSeen(id) > hotstartBoundary
        }
        func isGone(_ id: Int) -> Bool {
            session.removedObjectIDs.contains(id) || newlyRemoved.contains(id)
        }

        // Record first sightings and seed keep-alive.
        for id in newObjectIDs {
            if session.firstFrameByObjectID[id] == nil {
                session.firstFrameByObjectID[id] = frameIndex
            }
            session.keepAliveByObjectID[id] = parameters.initTrkKeepAlive
        }

        func adjustKeepAlive(_ id: Int, by delta: Int) {
            let current = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
            session.keepAliveByObjectID[id] =
                delta > 0
                ? min(parameters.maxTrkKeepAlive, current + delta)
                : max(parameters.minTrkKeepAlive, current + delta)
        }

        // A track counts as matched if any detection claimed it. `emptyTrackIDs` already
        // separates "occluded" from "gone".
        var matched: Set<Int> = []
        for tracks in detectionToMatchedTrackIDs.values { matched.formUnion(tracks) }
        for id in matched { adjustKeepAlive(id, by: 1) }
        for id in unmatchedTrackIDs {
            session.unmatchedFramesByObjectID[id, default: []].append(frameIndex)
            adjustKeepAlive(id, by: -1)
        }
        if parameters.decreaseTrkKeepAliveForEmptyMasklets {
            for id in emptyTrackIDs { adjustKeepAlive(id, by: -1) }
        }

        // Drop tracks unmatched for too long inside the window, and suppress tracks whose
        // keep-alive has run out. Sorted for deterministic removal.
        for id in session.unmatchedFramesByObjectID.keys.sorted() {
            guard let frames = session.unmatchedFramesByObjectID[id], !isGone(id) else { continue }
            if frames.count >= parameters.hotstartUnmatchThresh, withinHotstart(id) {
                newlyRemoved.insert(id)
                CLILogger.log(
                    "Removing object \(id) at frame \(frameIndex): unmatched on frames \(frames)")
            }
            let keepAlive = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
            // The flag reads as a restriction but upstream uses it as an off switch:
            // `modeling_sam3_video.py` gates on `not suppress_unmatched_only_within_hotstart`.
            if keepAlive <= 0, !parameters.suppressUnmatchedOnlyWithinHotstart, !isGone(id) {
                suppressed.insert(id)
            }
        }

        // Record overlaps. Two tracks overlap when the same detection claims both, and the
        // later-appearing one is the candidate duplicate.
        //
        // Walked in detection-id order. Ids are unique, so the traversal cannot depend on
        // dictionary layout.
        let bySeenOrder: (Int, Int) -> Bool = { firstSeen($0) < firstSeen($1) }
        for detection in detectionToMatchedTrackIDs.keys.sorted() {
            guard let tracks = detectionToMatchedTrackIDs[detection], tracks.count >= 2,
                let anchor = reverse
                    ? tracks.max(by: bySeenOrder) : tracks.min(by: bySeenOrder)
            else { continue }
            for id in tracks where id != anchor {
                let pair = VideoInferenceSession.OverlapPair(firstAppearing: anchor, duplicate: id)
                session.overlapFramesByPair[pair, default: []].append(frameIndex)
            }
        }

        // Drop duplicates that have overlapped for long enough, again only inside the
        // hotstart window.
        for pair in session.overlapFramesByPair.keys.sorted(by: {
            ($0.firstAppearing, $0.duplicate) < ($1.firstAppearing, $1.duplicate)
        }) {
            let id = pair.duplicate
            guard let frames = session.overlapFramesByPair[pair], !isGone(id),
                withinHotstart(id), frames.count >= parameters.hotstartDupThresh
            else { continue }
            newlyRemoved.insert(id)
            CLILogger.log(
                "Removing object \(id) at frame \(frameIndex): overlaps object "
                    + "\(pair.firstAppearing) on frames \(frames)")
        }

        session.removedObjectIDs.formUnion(newlyRemoved)
        session.suppressedObjectIDsByFrame[frameIndex] = suppressed
        return newlyRemoved
    }
}
