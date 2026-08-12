// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreAISpeech
import Foundation

// MARK: - Terminal rendering

/// Usable terminal columns, falling back to 80 when stdout is not a terminal.
private func terminalWidth() -> Int {
    var size = winsize()
    if ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0, size.ws_col > 0 {
        return Int(size.ws_col)
    }
    if let columns = ProcessInfo.processInfo.environment["COLUMNS"], let value = Int(columns),
        value > 0
    {
        return value
    }
    return 80
}

/// The last `width` characters, marked with a leading ellipsis when truncated.
///
/// The tail rather than the head: in a live partial the head is settled text the reader
/// has already seen, and the newest words are what they are waiting for.
private func fittedTail(_ text: String, width: Int) -> String {
    guard width > 1 else { return "" }
    guard text.count > width else { return text }
    return "…" + String(text.suffix(width - 1))
}

/// `--stream` mode: replay a file through the push-based streaming API.
///
/// This repo does not capture audio — a host app owns `AVAudioEngine` and pushes PCM in.
/// Replaying a file through the same `append(pcm:)` entry point is what makes streaming
/// testable and lets it be compared against the offline path.
func runStreaming(
    bundleURL: URL,
    audioPath: String,
    chunkFrames: Int?,
    rightContextFrames: Int?,
    leftContextFrames: Int?,
    endpointFrames: Int?,
    realtime: Bool,
    simulated: Bool = false,
    verbose: Bool
) async throws {
    let start = ContinuousClock.now
    let model = try await SpeechRecognitionModel(resourcesAt: bundleURL)
    let loadMs = (ContinuousClock.now - start).inMilliseconds
    let sampleRate = await model.sampleRate

    let base = StreamingConfig.balanced
    let config = StreamingConfig(
        leftContextFrames: leftContextFrames ?? base.leftContextFrames,
        chunkFrames: chunkFrames ?? base.chunkFrames,
        rightContextFrames: rightContextFrames ?? base.rightContextFrames,
        endpointSilenceFrames: endpointFrames ?? base.endpointSilenceFrames,
        maxSegmentFrames: base.maxSegmentFrames)

    let url = URL(fileURLWithPath: audioPath)
    let pcm = try MelSpectrogram.loadAndResample(url, targetSampleRate: sampleRate)

    let updates = try await model.startStream(config: config, simulated: simulated)
    // Report what the session adopted, not what was asked for: a bundle not exported for
    // streaming has its own traced window, and left context is widened to fill it.
    let active = await model.activeStreamingConfig ?? config

    print("Model loaded in \(String(format: "%.0f", loadMs)) ms")
    if active.windowMelFrames != config.windowMelFrames {
        print(
            "Bundle traced for \(active.windowMelFrames) mel frames — left context widened "
                + "from \(config.leftContextFrames) to \(active.leftContextFrames) frames")
    }
    print(
        "Streaming: left \(active.leftContextFrames) / chunk \(active.chunkFrames) / "
            + "right \(active.rightContextFrames) frames "
            + "(\(String(format: "%.2f", active.seconds(frames: active.leftContextFrames)))s / "
            + "\(String(format: "%.2f", active.seconds(frames: active.chunkFrames)))s / "
            + "\(String(format: "%.2f", active.seconds(frames: active.rightContextFrames)))s)")
    print(
        "Window \(active.windowSampleCount) samples "
            + "(\(String(format: "%.2f", Double(active.windowSampleCount) / sampleRate))s, "
            + "\(active.windowMelFrames) mel, \(active.windowEncoderFrames) enc) · "
            + "theoretical latency \(String(format: "%.2f", active.theoreticalLatency))s")
    print(
        "Audio: \(pcm.count) samples "
            + "(\(String(format: "%.2f", Double(pcm.count) / sampleRate))s)"
            + (realtime ? " · paced at 1x" : " · as fast as possible")
            + (simulated ? " · SIMULATED (decode deferred to end)" : ""))
    print("-- Streaming ----------------------------------------------------------")

    // Collect finalized segments off the update stream while the pusher runs.
    //
    // Partials are rendered in place with `\r`, which only works on a terminal and only
    // while the line fits on one physical row: `\r` returns to the start of the last
    // *wrapped* row, so a partial wider than the terminal leaves its earlier rows on
    // screen and the rewrite becomes an append. So clamp to the width and keep the tail,
    // where the new words are — and when stdout is redirected, drop partials entirely
    // rather than emit a wall of prefixes that makes piped output undiffable.
    let showPartials = isatty(STDOUT_FILENO) != 0
    let width = terminalWidth()
    let collector = Task { () -> [TranscriptSegment] in
        var finals: [TranscriptSegment] = []
        var lastPartialLength = 0
        for await update in updates {
            switch update {
            case .partial(let segment):
                guard showPartials else { break }
                // Two columns of indent, and leave the last column free so the cursor
                // never wraps on its own.
                let line = "  " + fittedTail(segment.text, width: width - 3)
                let pad = max(0, lastPartialLength - line.count)
                FileHandle.standardOutput.write(
                    Data(("\r" + line + String(repeating: " ", count: pad)).utf8))
                lastPartialLength = line.count
            case .finalized(let segment):
                if lastPartialLength > 0 {
                    FileHandle.standardOutput.write(
                        Data(("\r" + String(repeating: " ", count: lastPartialLength)).utf8))
                }
                print(
                    (showPartials ? "\r" : "")
                        + "[\(String(format: "%6.2f", segment.startTime)) –"
                        + "\(String(format: "%7.2f", segment.endTime))]  \(segment.text)")
                lastPartialLength = 0
                finals.append(segment)
            }
        }
        return finals
    }

    // Push in chunk-sized slices, which is what a host app's tap would deliver.
    let pushSize = active.chunkFrames * active.samplesPerEncoderFrame
    let wallStart = ContinuousClock.now
    var offset = 0
    while offset < pcm.count {
        let end = min(offset + pushSize, pcm.count)
        if realtime {
            // Pace to the audio's own timeline so the run reflects real latency.
            let target = Double(offset) / sampleRate
            let elapsed = (ContinuousClock.now - wallStart).inMilliseconds / 1000
            if target > elapsed {
                try await Task.sleep(for: .milliseconds(Int((target - elapsed) * 1000)))
            }
        }
        try await model.append(pcm: Array(pcm[offset..<end]))
        offset = end
    }
    _ = try await model.finishStream()
    let segments = await collector.value
    let wallMs = (ContinuousClock.now - wallStart).inMilliseconds

    let full = segments.map(\.text).joined(separator: " ")
    print("── Transcription ──────────────────────────────────────────────────────")
    print("  \(full)")
    let audioSeconds = Double(pcm.count) / sampleRate
    print(
        "  segments: \(segments.count)  wall: \(String(format: "%.0f", wallMs)) ms  "
            + "RTF: \(String(format: "%.3f", (wallMs / 1000) / audioSeconds))"
            + (realtime ? " (paced — not a throughput measure)" : ""))
    if verbose {
        for segment in segments {
            print(
                "    [\(segment.segmentIndex)] \(String(format: "%.2f", segment.startTime))–"
                    + "\(String(format: "%.2f", segment.endTime))s  "
                    + "\(segment.tokens.count) tokens")
        }
    }
}
