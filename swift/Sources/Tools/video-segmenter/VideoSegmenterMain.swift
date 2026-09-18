// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIShared
import CoreAIVideoSegmenter
import CoreGraphics
import Foundation

@main
struct VideoSegmenterCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video-segmenter",
        abstract: "Segment and track objects through a video using text prompts (SAM 3 video)."
    )

    // MARK: - Options

    @Option(name: .long, help: "Path to the video_segmenter model bundle directory.")
    var model: String

    @Option(name: .long, help: "Path to the input video (mp4, mov, ...).")
    var inputVideo: String?

    @Option(name: .long, help: "Text prompt describing an object to track. Repeatable.")
    var prompt: [String] = []

    @Option(name: .long, help: "Output video path. Defaults to <input>_segmented.mp4.")
    var output: String?

    @Option(name: .long, help: "Stop after this many frames. Default: the whole video.")
    var maxFrames: Int?

    @Option(name: .long, help: "Write per-frame object ids, scores, and boxes to this JSON path.")
    var outputJson: String?

    // MARK: - Rendering

    @Option(name: .long, help: "Mask fill opacity, 0–1.")
    var maskOpacity: Float = 0.5

    @Flag(name: .long, inversion: .prefixedNo, help: "Draw mask-derived bounding boxes.")
    var boxes: Bool = true

    @Flag(name: .long, inversion: .prefixedNo, help: "Draw '#id prompt score' captions.")
    var labels: Bool = true

    @Flag(name: .long, help: "Run inference only; skip writing an annotated video.")
    var noRender: Bool = false

    // MARK: - Tracking overrides

    @Option(name: .long, help: "Detection probability threshold.")
    var scoreThreshold: Float?

    @Option(
        name: .long,
        help: """
            Mask-IoU threshold for detection NMS. 0 disables it, matching a Python run \
            without the kernels-community/cv-utils kernel.
            """)
    var detNmsThresh: Float?

    @Option(
        name: .long,
        help: """
            Max component area for hole filling and sprinkle removal. 0 disables both, \
            matching a Python run without the kernels-community/cv-utils kernel.
            """)
    var fillHoleArea: Int?

    @Option(name: .long, help: "Frames of output held back by the hotstart heuristics. 0 disables.")
    var hotstartDelay: Int?

    @Option(name: .long, help: "Recondition every Nth frame. 0 disables reconditioning.")
    var reconditionEvery: Int?

    // MARK: - Diagnostics

    @Flag(name: .long, help: "Run a warmup pass before the first frame.")
    var warmup: Bool = false

    @Flag(name: .long, help: "Print per-frame progress and per-entrypoint timing.")
    var verbose: Bool = false

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear the Core AI specialization cache for this bundle before loading.")
    var clearCoreAICache: Bool = false

    @Option(
        name: .long,
        help: """
            Path to a reference directory, in the layout `ParityReference` documents. \
            Replays the same clip through the Swift stack and reports per-frame agreement \
            instead of rendering.
            """)
    var parity: String?

    @Option(name: .long, help: "Minimum per-frame mask IoU in --parity mode.")
    var iouFloor: Float = 0.98

    @Option(name: .long, help: "Maximum per-frame score delta in --parity mode.")
    var scoreTol: Float = 0.02

    @Option(name: .long, help: "Maximum per-frame box delta in pixels, in --parity mode.")
    var boxTol: Float = 2.0

    // MARK: - Validation

    func validate() throws {
        if parity == nil {
            guard inputVideo != nil else {
                throw ValidationError("--input-video is required (unless --parity is set).")
            }
            guard !prompt.isEmpty else {
                throw ValidationError("At least one --prompt is required.")
            }
        }
        guard (0...1).contains(maskOpacity) else {
            throw ValidationError("--mask-opacity must be between 0 and 1.")
        }
        if let maxFrames, maxFrames <= 0 {
            throw ValidationError("--max-frames must be positive.")
        }
    }

    // MARK: - Run

    func run() async throws {
        if verbose { CLILogger.level = 1 }

        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: URL(fileURLWithPath: expand(model)))
            print("Cleared the specialization cache for \(cleared.count) asset(s).")
        }

        let segmenter = try await VideoSegmenter(
            resourcesAt: expand(model), parameters: overrides(), pinning: applyTrackingFlags)

        print("Preparing asset...", terminator: "")
        fflush(stdout)
        let loadStart = ContinuousClock.now
        try await segmenter.loadResources()
        print(" done in \(format(ContinuousClock.now - loadStart))")

        if warmup {
            print("Warming up...", terminator: "")
            fflush(stdout)
            let warmupStart = ContinuousClock.now
            try await segmenter.warmup()
            print(" done in \(format(ContinuousClock.now - warmupStart))")
        }

        if let parityDirectory = parity {
            try await runParity(
                segmenter: segmenter, directory: URL(fileURLWithPath: expand(parityDirectory)))
            return
        }
        try await runSegmentation(segmenter: segmenter)
    }

    /// Tracking flags the user passed explicitly. Applied after the bundle's `tracking`
    /// block, since metadata.json declares most of these and would otherwise win.
    private func applyTrackingFlags(to parameters: inout VideoSegmentationParameters) {
        scoreThreshold.map { parameters.scoreThresholdDetection = $0 }
        detNmsThresh.map { parameters.detNmsThresh = $0 }
        fillHoleArea.map { parameters.fillHoleArea = $0 }
        hotstartDelay.map { parameters.hotstartDelay = $0 }
        reconditionEvery.map { parameters.reconditionEveryNthFrame = $0 }
    }

    /// Parameters assembled from the flags. Anything left `nil` keeps the bundle's value.
    private func overrides() -> VideoSegmentationParameters {
        var parameters = VideoSegmentationParameters.default
        applyTrackingFlags(to: &parameters)
        parameters.maskOpacity = maskOpacity
        parameters.drawBoxes = boxes
        parameters.drawLabels = labels
        return parameters
    }

    // MARK: - Segmentation

    private func runSegmentation(segmenter: VideoSegmenter) async throws {
        let sourceURL = URL(fileURLWithPath: expand(inputVideo!))
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw ValidationError("No video at \(sourceURL.path)")
        }
        let metadata = try await SequentialVideoReader.metadata(of: sourceURL)
        let expected = min(metadata.estimatedFrameCount, maxFrames ?? .max)
        print(
            "Video: \(metadata.width)×\(metadata.height), "
                + "\(String(format: "%.2f", metadata.nominalFrameRate)) fps, ~\(expected) frames")
        print("Prompts: \(prompt.joined(separator: ", "))")

        let collector = FrameCollector()
        let started = ContinuousClock.now

        if noRender {
            for try await frame in segmenter.segment(
                videoAt: sourceURL, prompts: prompt, maxFrames: maxFrames)
            {
                await collector.record(frame, verbose: verbose)
            }
        } else {
            let destination = URL(fileURLWithPath: expand(output ?? defaultOutputPath(for: sourceURL)))
            let verbose = self.verbose
            _ = try await segmenter.renderAnnotatedVideo(
                from: sourceURL, prompts: prompt, to: destination, maxFrames: maxFrames,
                onFrame: { frame in
                    await collector.record(frame, verbose: verbose)
                })
            print("Wrote \(destination.path)")
        }

        let elapsed = ContinuousClock.now - started
        let records = await collector.records
        print("")
        print("Segmented \(records.count) frames in \(format(elapsed))", terminator: "")
        if !records.isEmpty {
            let perFrame = elapsed.inSeconds / Double(records.count)
            print(" (\(String(format: "%.0f", perFrame * 1000)) ms/frame)")
        } else {
            print("")
        }
        printTimings(await segmenter.lastRunTimings, frames: records.count)
        printTrackSummary(records)

        if let jsonPath = outputJson {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(records).write(
                to: URL(fileURLWithPath: expand(jsonPath)))
            print("Wrote \(expand(jsonPath))")
        }
    }

    private func defaultOutputPath(for source: URL) -> String {
        source.deletingPathExtension().lastPathComponent + "_segmented.mp4"
    }

    private func printTimings(_ timings: [String: Double], frames: Int) {
        guard !timings.isEmpty, frames > 0 else { return }
        print("")
        print("  \(pad("entrypoint", 20))\(rpad("total", 10))\(rpad("per frame", 12))")
        print("  " + String(repeating: "-", count: 42))
        for (name, seconds) in timings.sorted(by: { $0.value > $1.value }) {
            print(
                "  " + pad(name, 20)
                    + rpad(String(format: "%.2fs", seconds), 10)
                    + rpad(String(format: "%.0f ms", seconds / Double(frames) * 1000), 12))
        }
        let total = timings.values.reduce(0, +)
        print(
            "  " + pad("total in Core AI", 20)
                + rpad(String(format: "%.2fs", total), 10)
                + rpad(String(format: "%.0f ms", total / Double(frames) * 1000), 12))
    }

    private func printTrackSummary(_ records: [FrameRecord]) {
        var frameCount: [Int: Int] = [:]
        var prompts: [Int: String] = [:]
        for record in records {
            for object in record.objects {
                frameCount[object.id, default: 0] += 1
                prompts[object.id] = object.prompt
            }
        }
        guard !frameCount.isEmpty else {
            print("\nNo objects tracked. Try a lower --score-threshold or a different prompt.")
            return
        }
        print("")
        print("  \(pad("track", 8))\(pad("prompt", 20))\(rpad("frames", 8))")
        print("  " + String(repeating: "-", count: 36))
        for id in frameCount.keys.sorted() {
            print(
                "  " + pad("#\(id)", 8) + pad(prompts[id] ?? "", 20)
                    + rpad("\(frameCount[id]!)", 8))
        }
    }

    // MARK: - Parity

    /// Compare against a ``ParityReference`` dump.
    private func runParity(segmenter: VideoSegmenter, directory: URL) async throws {
        let reference = try ParityReference(directory: directory)
        print(
            "Parity reference: \(reference.frameCount) frames, prompts "
                + "\(reference.prompts.joined(separator: ", "))")

        var results: [Int: [TrackedObject]] = [:]
        var lowResolution: [Int: [[Float]]] = [:]
        // Ask for the pre-upsample logits when the reference has them, so a disagreement
        // can be attributed to the tracker or to the upsample.
        var parityParameters = await segmenter.parameters
        parityParameters.emitLowResolutionMasks =
            reference.lowResolutionMasks(at: reference.frameIndices[0]) != nil
        applyTrackingFlags(to: &parityParameters)
        // Prefer the frames PyTorch itself decoded, which holds the decoder constant. See
        // `ParityReference` for the size of the AVFoundation/PyAV difference.
        if let frames = try reference.decodedFrames() {
            print("Frames: \(frames.count) PNGs from the reference (decoder held constant)")
            for try await frame in segmenter.segment(
                frames: frames, prompts: reference.prompts, parameters: parityParameters)
            {
                results[frame.frameIndex] = frame.objects
                lowResolution[frame.frameIndex] = frame.lowResolutionMasks
            }
        } else {
            print("Frames: decoding \(reference.videoURL.path) with AVFoundation")
            print(
                "  The reference has no frames/ directory, so decoder differences are\n"
                    + "  folded into the numbers below. Re-dump with one to remove them.")
            for try await frame in segmenter.segment(
                videoAt: reference.videoURL, prompts: reference.prompts,
                maxFrames: reference.frameCount, parameters: parityParameters)
            {
                results[frame.frameIndex] = frame.objects
                lowResolution[frame.frameIndex] = frame.lowResolutionMasks
            }
        }

        print("")
        print(
            "\(rpad("frame", 6)) \(rpad("objs", 6)) \(rpad("ids", 5)) \(rpad("min IoU", 9)) "
                + "\(rpad("max dScore", 11)) \(rpad("max dBox", 9)) \(rpad("lowres", 9))   result")
        print(String(repeating: "-", count: 72))

        var failures = 0
        var worstLowResDelta: Float = 0
        for frameIndex in reference.frameIndices {
            let expected = try reference.objects(at: frameIndex)
            let actual = results[frameIndex] ?? []
            let comparison = compare(expected: expected, actual: actual)
            let ok =
                comparison.idsMatch && comparison.minIoU >= iouFloor
                && comparison.maxScoreDelta <= scoreTol && comparison.maxBoxDelta <= boxTol
            if !ok { failures += 1 }
            // Max absolute logit difference before the upsample, when both sides have it.
            var lowResColumn = "-"
            if let referenceLowRes = reference.lowResolutionMasks(at: frameIndex),
                let actualLowRes = lowResolution[frameIndex], !actualLowRes.isEmpty
            {
                let flat = actualLowRes.flatMap { $0 }
                if flat.count == referenceLowRes.values.count {
                    var worst: Float = 0
                    for index in flat.indices {
                        worst = max(worst, abs(flat[index] - referenceLowRes.values[index]))
                    }
                    worstLowResDelta = max(worstLowResDelta, worst)
                    lowResColumn = String(format: "%.4f", worst)
                } else {
                    lowResColumn = "n/a"
                }
            }
            print(
                rpad("\(frameIndex)", 6) + " "
                    + rpad("\(expected.count)/\(actual.count)", 6) + " "
                    + rpad(comparison.idsMatch ? "ok" : "DIFF", 5) + " "
                    + rpad(String(format: "%.4f", comparison.minIoU), 9) + " "
                    + rpad(String(format: "%.4f", comparison.maxScoreDelta), 11) + " "
                    + rpad(String(format: "%.1f", comparison.maxBoxDelta), 9) + " "
                    + rpad(lowResColumn, 9) + "   "
                    + (ok ? "ok" : "FAIL"))
        }

        print(String(repeating: "-", count: 72))
        if worstLowResDelta > 0 {
            // The `lowres` column measures the port alone. The final masks add an upsample
            // and a threshold, and a logit near zero crosses that threshold on an
            // arbitrarily small difference, which is how a tiny logit delta becomes a large
            // box delta.
            print(
                "Worst mask-logit difference before upsampling: "
                    + String(format: "%.4f", worstLowResDelta))
        }
        if failures == 0 {
            print(
                "All \(reference.frameCount) frames within tolerance "
                    + "(IoU ≥ \(iouFloor), dScore ≤ \(scoreTol), dBox ≤ \(boxTol)).")
            return
        }
        print("\(failures) of \(reference.frameCount) frames outside tolerance.")
        if detNmsThresh != 0 || fillHoleArea != 0 {
            // The most common cause of a spurious parity failure.
            print(
                "The Python reference no-ops NMS and hole filling when "
                    + "kernels-community/cv-utils is not installed. If that was the case, rerun "
                    + "with --det-nms-thresh 0 --fill-hole-area 0.")
        }
        throw ExitCode.failure
    }

    private struct Comparison {
        var idsMatch = false
        var minIoU: Float = 1
        var maxScoreDelta: Float = 0
        var maxBoxDelta: Float = 0
    }

    private func compare(
        expected: [ParityReference.Object], actual: [TrackedObject]
    ) -> Comparison {
        var comparison = Comparison()
        comparison.idsMatch = expected.map(\.id).sorted() == actual.map(\.id).sorted()
        guard comparison.idsMatch else {
            comparison.minIoU = 0
            return comparison
        }
        let byID = Dictionary(uniqueKeysWithValues: actual.map { ($0.id, $0) })
        for reference in expected {
            guard let got = byID[reference.id] else {
                comparison.minIoU = 0
                continue
            }
            comparison.minIoU = min(comparison.minIoU, reference.mask.iou(got.mask))
            comparison.maxScoreDelta = max(
                comparison.maxScoreDelta, abs(reference.score - got.score))
            comparison.maxBoxDelta = max(
                comparison.maxBoxDelta,
                Float(
                    max(
                        abs(reference.box.minX - got.box.minX),
                        abs(reference.box.minY - got.box.minY),
                        abs(reference.box.maxX - got.box.maxX),
                        abs(reference.box.maxY - got.box.maxY))))
        }
        return comparison
    }

    // MARK: - Formatting

    private func expand(_ path: String) -> String { (path as NSString).expandingTildeInPath }

    private func format(_ duration: Duration) -> String {
        String(format: "%.2fs", duration.inSeconds)
    }

    private func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
    }

    private func rpad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
    }
}

// MARK: - JSON records

struct FrameRecord: Encodable, Sendable {
    struct Object: Encodable, Sendable {
        let id: Int
        let prompt: String
        let score: Float
        let trackerScore: Float
        let box: [Double]
        let maskPixels: Int
    }
    let frameIndex: Int
    let objects: [Object]
    /// Wall clock spent on this frame, in milliseconds.
    let processingMs: Double
}

/// Accumulates per-frame records off the render loop's hot path.
actor FrameCollector {
    private(set) var records: [FrameRecord] = []
    private var totalProcessing: Duration = .zero

    func record(_ frame: VideoSegmentationFrame, verbose: Bool) {
        let objects = frame.objects.map {
            FrameRecord.Object(
                id: $0.id, prompt: $0.prompt, score: $0.score, trackerScore: $0.trackerScore,
                box: [$0.box.minX, $0.box.minY, $0.box.maxX, $0.box.maxY],
                maskPixels: $0.mask.area)
        }
        records.append(
            FrameRecord(
                frameIndex: frame.frameIndex, objects: objects,
                processingMs: frame.processingTime.inSeconds * 1000))
        totalProcessing += frame.processingTime

        if verbose {
            let ids = objects.map { "#\($0.id)" }.joined(separator: " ")
            // This frame's own time, then the running total of those times. `segment()`
            // lets the frame loop run a bounded distance ahead of this consumer, so a
            // timestamp here would report collection time, not compute time.
            print(
                "  frame \(frame.frameIndex): \(objects.count) object(s) \(ids)"
                    .padding(toLength: 46, withPad: " ", startingAt: 0)
                    + String(format: "%7.0f ms", frame.processingTime.inSeconds * 1000)
                    + String(format: "   %7.1fs cumulative", totalProcessing.inSeconds))
        }
    }
}
