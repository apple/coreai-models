// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// Text-promptable video object segmentation and tracking (SAM 3 video).
///
/// ```swift
/// let segmenter = try await VideoSegmenter(resourcesAt: "exports/sam3_video_float16")
/// try await segmenter.renderAnnotatedVideo(
///     from: URL(fileURLWithPath: "clip.mp4"),
///     prompts: ["person", "dog"],
///     to: URL(fileURLWithPath: "out.mp4"))
/// ```
///
/// Or consume the per-frame results directly:
///
/// ```swift
/// for try await frame in segmenter.segment(videoAt: url, prompts: ["person"]) {
///     print(frame.frameIndex, frame.objects.map(\.id))
/// }
/// ```
@VideoSegmentationActor
public final class VideoSegmenter: ResourceManaging {
    private let bundle: VideoSegmenterBundle
    private let engine: VideoSegmentationEngine
    private let tokenizer: CLIPTokenizer
    public let parameters: VideoSegmentationParameters

    private var shapes: VideoSegmentationEngine.Shapes?
    private var packer: MemoryBankPacker?
    private var tracker: TrackerLoop?
    /// In-flight load, so concurrent callers build one packer and tracker between them.
    private var loadTask: Task<Void, Error>?

    /// Cumulative wall clock per Core AI entrypoint from the most recent run.
    public private(set) var lastRunTimings: [String: Double] = [:]

    /// Load a `kind: video_segmenter` bundle directory.
    ///
    /// - Parameters:
    ///   - path: Bundle directory holding `metadata.json`, the `.aimodel`, and `tokenizer/`.
    ///   - parameters: Overrides applied under the bundle's own `tracking` block, so a
    ///     bundle that declares its thresholds wins. For per-run overrides, see
    ///     ``segment(videoAt:prompts:maxFrames:parameters:)``.
    ///   - pinning: Applied after the bundle, for settings that must outrank it such as an
    ///     explicit command-line flag. The packer and tracker are built from the result, so
    ///     this is the only way to override a field the bundle declares.
    public init(
        resourcesAt path: String,
        parameters: VideoSegmentationParameters = .default,
        pinning: ((inout VideoSegmentationParameters) -> Void)? = nil
    ) async throws {
        let bundle = try VideoSegmenterBundle(from: path)
        self.bundle = bundle
        var resolved = try bundle.parameters(overriding: parameters)
        pinning?(&resolved)
        self.parameters = resolved
        self.tokenizer = try CLIPTokenizer(folder: bundle.tokenizerFolder)
        self.engine = VideoSegmentationEngine(modelURL: bundle.modelURL)
    }

    /// Load and specialize the asset. Called implicitly on first use.
    public func loadResources() async throws {
        guard shapes == nil else { return }
        if let loadTask { return try await loadTask.value }
        let task = Task { try await performLoad() }
        loadTask = task
        do {
            try await task.value
        } catch {
            loadTask = nil
            throw error
        }
        loadTask = nil
    }

    private func performLoad() async throws {
        try await engine.loadResources()
        guard let resolved = await engine.shapes else {
            throw VideoSegmentationError.invalidConfiguration(
                "Engine reported no shapes after loading.")
        }
        // The bundle's declared geometry and the traced graph must agree. They come from the
        // same export, so a mismatch means they were paired by hand.
        guard resolved.imageSize == bundle.geometry.imageSize,
            resolved.spatialSlots == bundle.geometry.spatialSlots,
            resolved.ptrSlots == bundle.geometry.ptrSlots,
            resolved.textSequenceLength == bundle.geometry.maxTextSeqLen
        else {
            throw VideoSegmentationError.unsupportedGeometry(
                "metadata.json declares image_size \(bundle.geometry.imageSize), spatial_slots "
                    + "\(bundle.geometry.spatialSlots), ptr_slots \(bundle.geometry.ptrSlots), "
                    + "max_text_seq_len \(bundle.geometry.maxTextSeqLen), but the asset was traced "
                    + "at \(resolved.imageSize) / \(resolved.spatialSlots) / \(resolved.ptrSlots) "
                    + "/ \(resolved.textSequenceLength).")
        }

        let packer = try await makePacker(shapes: resolved, parameters: parameters)
        // A concurrent `unloadResources` cancels this task, and committing afterwards would
        // silently undo it.
        try Task.checkCancellation()
        self.shapes = resolved
        self.packer = packer
        self.tracker = TrackerLoop(
            engine: engine, shapes: resolved, parameters: parameters, packer: packer)
    }

    /// Allocate a memory bank at `shapes` and wrap it in a packer for `parameters`.
    private func makePacker(
        shapes: VideoSegmentationEngine.Shapes, parameters: VideoSegmentationParameters
    ) async throws -> MemoryBankPacker {
        let packed = try await MemoryBankPacker.makePacked(engine: engine)
        return try MemoryBankPacker(shapes: shapes, parameters: parameters, packed: packed)
    }

    /// Release the asset. The next call reloads it.
    public func unloadResources() async {
        loadTask?.cancel()
        loadTask = nil
        await engine.unloadResources()
        shapes = nil
        packer = nil
        tracker = nil
    }

    /// Load the asset, then run ``VideoSegmentationEngine/warmup()``.
    public func warmup() async throws {
        try await loadResources()
        try await engine.warmup()
    }

    /// Segment and track `prompts` through an already-decoded frame sequence.
    ///
    /// The frames are consumed in the order given and their indices are their positions.
    /// Use this to hold the video decoder constant. See ``ParityReference`` for the size of
    /// the AVFoundation/PyAV difference.
    public nonisolated func segment(
        frames: [CGImage],
        prompts: [String],
        parameters overrides: VideoSegmentationParameters? = nil
    ) -> AsyncThrowingStream<VideoSegmentationFrame, Error> {
        stream(source: .images(frames), prompts: prompts, maxFrames: nil, overrides: overrides)
    }

    /// Segment and track `prompts` through the video at `url`, one result per frame.
    ///
    /// Results arrive in frame order but lag the decoder by `hotstartDelay - 1` frames. The
    /// backlog is flushed when the video ends.
    ///
    /// Nonisolated so a caller can start the stream without an `await`.
    public nonisolated func segment(
        videoAt url: URL,
        prompts: [String],
        maxFrames: Int? = nil,
        parameters overrides: VideoSegmentationParameters? = nil
    ) -> AsyncThrowingStream<VideoSegmentationFrame, Error> {
        stream(source: .video(url), prompts: prompts, maxFrames: maxFrames, overrides: overrides)
    }

    private nonisolated func stream(
        source: FrameSource,
        prompts: [String],
        maxFrames: Int?,
        overrides: VideoSegmentationParameters?
    ) -> AsyncThrowingStream<VideoSegmentationFrame, Error> {
        backpressuredStream(depth: Self.streamDepth) { yield in
            try await self.run(
                source: source, prompts: prompts, maxFrames: maxFrames, overrides: overrides,
                emit: yield)
        }
    }

    /// Unconsumed frames the stream will hold. Each carries a decoded `CGImage`. This is the
    /// knob that keeps a slow consumer from growing the backlog without limit.
    private nonisolated static let streamDepth = 4

    /// Where a run's frames come from.
    private enum FrameSource {
        case video(URL)
        case images([CGImage])
    }

    /// Segment the video and write an annotated copy with masks and boxes composited on.
    ///
    /// - Returns: The number of frames written.
    @discardableResult
    public func renderAnnotatedVideo(
        from source: URL,
        prompts: [String],
        to destination: URL,
        maxFrames: Int? = nil,
        parameters overrides: VideoSegmentationParameters? = nil,
        onFrame: (@Sendable (VideoSegmentationFrame) async -> Void)? = nil
    ) async throws -> Int {
        let effective = overrides ?? parameters
        let metadata = try await SequentialVideoReader.metadata(of: source)
        let renderer = VideoOverlayRenderer(parameters: effective)
        let writer = try StreamingVideoWriter(
            url: destination,
            width: metadata.width,
            height: metadata.height,
            frameRate: max(1, Int(metadata.nominalFrameRate.rounded())))

        var written = 0
        // Encoding serializes behind the writer actor while the next frame's inference runs,
        // so the two overlap with no explicit pipeline.
        do {
            for try await frame in segment(
                videoAt: source, prompts: prompts, maxFrames: maxFrames, parameters: effective)
            {
                await onFrame?(frame)
                try await writer.append(renderer.render(frame.objects, onto: frame.image))
                written += 1
            }
        } catch {
            // An unfinished AVAssetWriter leaves a file with no moov atom, so close it before
            // the error propagates. `finish()` is idempotent.
            _ = try? await writer.finish()
            throw error
        }
        try await writer.finish()
        return written
    }

    // MARK: - Frame loop

    /// Port of `Sam3VideoModel.propagate_in_video_iterator`, forward only.
    ///
    /// Reverse propagation is plumbed through the heuristics but has no entry point here,
    /// since it would need the whole video resident.
    private func run(
        source: FrameSource,
        prompts: [String],
        maxFrames: Int?,
        overrides: VideoSegmentationParameters?,
        emit: (VideoSegmentationFrame) async -> Void
    ) async throws {
        guard !prompts.isEmpty else { throw VideoSegmentationError.noPrompts }
        try await loadResources()
        guard let shapes, let tracker = self.tracker else {
            throw VideoSegmentationError.invalidConfiguration("Engine failed to initialize.")
        }
        let parameters = overrides ?? self.parameters

        // `packer` and `tracker` were built from the loaded parameters. Both read tracking
        // knobs, so per-run overrides need their own pair.
        let runTracker: TrackerLoop
        if let overrides {
            let packer = try await makePacker(shapes: shapes, parameters: overrides)
            runTracker = TrackerLoop(
                engine: engine, shapes: shapes, parameters: overrides, packer: packer)
        } else {
            runTracker = tracker
        }

        let width: Int
        let height: Int
        let totalFrames: Int
        switch source {
        case .video(let url):
            let metadata = try await SequentialVideoReader.metadata(of: url)
            width = metadata.width
            height = metadata.height
            totalFrames = min(metadata.estimatedFrameCount, maxFrames ?? .max)
        case .images(let images):
            guard let first = images.first else { return }
            width = first.width
            height = first.height
            totalFrames = images.count
        }

        let session = VideoInferenceSession(videoWidth: width, videoHeight: height)
        for prompt in prompts {
            let id = session.addPrompt(prompt)
            if session.promptTokens[id] == nil {
                session.promptTokens[id] = tokenizer.encodeWithMask(
                    prompt, contextLength: shapes.textSequenceLength)
            }
        }

        let processor = FrameProcessor(
            engine: engine, shapes: shapes, parameters: parameters, tracker: runTracker)
        let postprocessor = MaskPostprocessor(
            lowResolutionSize: shapes.lowResMaskSize, videoWidth: width, videoHeight: height,
            emitLowResolutionMasks: parameters.emitLowResolutionMasks)

        // The hotstart buffer holds decoded frames alongside results, since the renderer
        // needs the source image and re-decoding would mean a second pass over the file.
        var buffer: [(raw: RawFrameOutput, image: CGImage, elapsed: Duration)] = []

        func handle(_ image: CGImage, index: Int) async throws {
            try Task.checkCancellation()
            let started = ContinuousClock.now
            let raw = try await processor.process(
                session: session, image: image, frameIndex: index,
                totalFrames: totalFrames, reverse: false)
            let elapsed = ContinuousClock.now - started

            guard parameters.hotstartEnabled else {
                await emit(finish(raw, image, elapsed))
                return
            }
            buffer.append((raw, image, elapsed))
            if buffer.count >= parameters.hotstartDelay {
                let (oldest, oldestImage, oldestElapsed) = buffer.removeFirst()
                await emit(finish(oldest, oldestImage, oldestElapsed))
            }
        }

        func finish(
            _ raw: RawFrameOutput, _ image: CGImage, _ elapsed: Duration
        ) -> VideoSegmentationFrame {
            let processed = postprocessor.postprocess(raw, session: session)
            return VideoSegmentationFrame(
                frameIndex: raw.frameIndex,
                objects: processed.objects,
                image: image,
                lowResolutionMasks: processed.lowResolutionMasks,
                processingTime: elapsed)
        }

        switch source {
        case .video(let url):
            for try await frame in SequentialVideoReader.frames(of: url, maxFrames: maxFrames) {
                try await handle(frame.image, index: frame.index)
            }
        case .images(let images):
            for (index, image) in images.enumerated() {
                try await handle(image, index: index)
            }
        }

        // Flush whatever the delay is still holding. Postprocessed last, so a track dropped
        // at the very end is hidden in the buffered frames too.
        for (raw, image, elapsed) in buffer {
            await emit(finish(raw, image, elapsed))
        }
        lastRunTimings = processor.timings
    }
}
