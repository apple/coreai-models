// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Removes tracks that never settle: ones the detector stops confirming, and duplicates
/// of an earlier track.
///
/// Port of `Sam3VideoModel._process_hotstart`. Both rules only fire for tracks that first
/// appeared inside the hotstart window; a track older than that has earned the benefit of the
/// doubt. The window is why output lags by `hotstartDelay` frames: a track removed on frame 20
/// must never have been shown on frame 8.
///
/// The keep-alive counter is a separate, always-on mechanism layered on the same
/// bookkeeping: matched frames raise it toward `maxTrkKeepAlive`, unmatched ones lower it
/// toward `minTrkKeepAlive`, and a track at or below zero is suppressed for the frame
/// rather than removed outright.
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

        // Step 1: record first sightings and seed keep-alive.
        for id in newObjectIDs {
            if session.firstFrameByObjectID[id] == nil {
                session.firstFrameByObjectID[id] = frameIndex
            }
            session.keepAliveByObjectID[id] = parameters.initTrkKeepAlive
        }

        // A track counts as matched if any detection claimed it. Using the
        // detection-to-track map avoids recomputing areas to distinguish "occluded" from
        // "gone", which is the distinction `emptyTrackIDs` already carries.
        var matched: Set<Int> = []
        for tracks in detectionToMatchedTrackIDs.values { matched.formUnion(tracks) }
        for id in matched {
            let current = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
            session.keepAliveByObjectID[id] = min(parameters.maxTrkKeepAlive, current + 1)
        }
        for id in unmatchedTrackIDs {
            session.unmatchedFramesByObjectID[id, default: []].append(frameIndex)
            let current = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
            session.keepAliveByObjectID[id] = max(parameters.minTrkKeepAlive, current - 1)
        }
        if parameters.decreaseTrkKeepAliveForEmptyMasklets {
            for id in emptyTrackIDs {
                let current = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
                session.keepAliveByObjectID[id] = max(parameters.minTrkKeepAlive, current - 1)
            }
        }

        // Step 2: drop tracks unmatched for too long inside the window, and suppress
        // tracks whose keep-alive has run out.
        //
        // Sorted so removal is deterministic. Upstream iterates a `defaultdict`, whose
        // order is insertion order and therefore also deterministic, but is not the same
        // order; the two only differ when two objects would be removed on the same frame,
        // and both are removed either way.
        for id in session.unmatchedFramesByObjectID.keys.sorted() {
            guard let frames = session.unmatchedFramesByObjectID[id] else { continue }
            if session.removedObjectIDs.contains(id) || newlyRemoved.contains(id) { continue }
            if frames.count >= parameters.hotstartUnmatchThresh {
                let firstFrame = session.firstFrameByObjectID[id] ?? frameIndex
                let withinHotstart =
                    reverse ? firstFrame < hotstartBoundary : firstFrame > hotstartBoundary
                if withinHotstart {
                    newlyRemoved.insert(id)
                    CLILogger.log(
                        "Removing object \(id) at frame \(frameIndex): unmatched on frames \(frames)")
                }
            }
            let keepAlive = session.keepAliveByObjectID[id] ?? parameters.initTrkKeepAlive
            if keepAlive <= 0 && !parameters.suppressUnmatchedOnlyWithinHotstart
                && !session.removedObjectIDs.contains(id) && !newlyRemoved.contains(id)
            {
                suppressed.insert(id)
            }
        }

        // Step 3: record overlaps. Two tracks overlap when the same detection claims both;
        // the later-appearing one is the candidate duplicate.
        for tracks in detectionToMatchedTrackIDs.values.sorted(by: { ($0.first ?? -1) < ($1.first ?? -1) }) {
            guard tracks.count >= 2 else { continue }
            let firstAppearing =
                reverse
                ? tracks.max(by: { firstFrame(session, $0, frameIndex) < firstFrame(session, $1, frameIndex) })
                : tracks.min(by: { firstFrame(session, $0, frameIndex) < firstFrame(session, $1, frameIndex) })
            guard let anchor = firstAppearing else { continue }
            for id in tracks where id != anchor {
                let pair = VideoInferenceSession.OverlapPair(firstAppearing: anchor, duplicate: id)
                session.overlapFramesByPair[pair, default: []].append(frameIndex)
            }
        }

        // Step 4: drop duplicates that have overlapped for long enough, again only inside
        // the hotstart window.
        for pair in session.overlapFramesByPair.keys.sorted(by: {
            ($0.firstAppearing, $0.duplicate) < ($1.firstAppearing, $1.duplicate)
        }) {
            guard let frames = session.overlapFramesByPair[pair] else { continue }
            let id = pair.duplicate
            if session.removedObjectIDs.contains(id) || newlyRemoved.contains(id) { continue }
            let firstFrame = session.firstFrameByObjectID[id] ?? frameIndex
            let withinHotstart =
                reverse ? firstFrame < hotstartBoundary : firstFrame > hotstartBoundary
            guard withinHotstart, frames.count >= parameters.hotstartDupThresh else { continue }
            newlyRemoved.insert(id)
            CLILogger.log(
                "Removing object \(id) at frame \(frameIndex): overlaps object "
                    + "\(pair.firstAppearing) on frames \(frames)")
        }

        session.removedObjectIDs.formUnion(newlyRemoved)
        session.suppressedObjectIDsByFrame[frameIndex] = suppressed
        return newlyRemoved
    }

    /// First frame an object was seen on. Upstream's `obj_first_frame_idx[x]` would raise for
    /// an object with no record; this defaults to the current frame instead of failing the
    /// video.
    private static func firstFrame(
        _ session: VideoInferenceSession, _ id: Int, _ frameIndex: Int
    ) -> Int {
        session.firstFrameByObjectID[id] ?? frameIndex
    }
}
