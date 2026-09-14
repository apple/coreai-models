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
///
/// ## Hotstart delay
///
/// SAM 3 buffers its first `hotstartDelay` frames (15 by default) before emitting
/// anything, because a track it removes on frame 20 must never have been shown on frame 8.
/// The stream honours that, which means the decoded frames have to be held too, bounded at
/// 15 frames and roughly 124 MB at 1080p. Set `hotstartDelay` to 0 to disable both the
/// delay and the removal rules that need it.
@VideoSegmentationActor
public final class VideoSegmenter: ResourceManaging {
    private let bundle: VideoSegmenterBundle
    private let engine: VideoSegmentationEngine
    private let tokenizer: CLIPTokenizer
    public let parameters: VideoSegmentationParameters

    private var shapes: VideoSegmentationEngine.Shapes?
    private var packer: MemoryBankPacker?
    private var tracker: TrackerLoop?

    /// Cumulative wall clock per Core AI entrypoint from the most recent run.
    public private(set) var lastRunTimings: [String: Double] = [:]

    /// Load a `kind: video_segmenter` bundle directory.
    ///
    /// - Parameters:
    ///   - path: Bundle directory holding `metadata.json`, the `.aimodel`, and `tokenizer/`.
    ///   - parameters: Overrides applied *under* the bundle's own `tracking` block, so a
    ///     bundle that declares its thresholds still wins. Pass per-run overrides to
    ///     ``segment(videoAt:prompts:maxFrames:parameters:)`` instead.
    public init(
        resourcesAt path: String,
        parameters: VideoSegmentationParameters = .default
    ) async throws {
        let bundle = try VideoSegmenterBundle(from: path)
        self.bundle = bundle
        self.parameters = bundle.parameters(overriding: parameters)
        self.tokenizer = try CLIPTokenizer(folder: bundle.tokenizerFolder)
        self.engine = VideoSegmentationEngine(modelURL: bundle.modelURL)
    }

    /// Load and specialize the asset. Called implicitly on first use.
    public func loadResources() async throws {
        guard shapes == nil else { return }
        try await engine.loadResources()
        guard let resolved = await engine.shapes else {
            throw VideoSegmentationError.invalidConfiguration(
                "Engine reported no shapes after loading.")
        }
        // The bundle's declared geometry and the traced graph must agree. They come from
        // the same export, so a mismatch means the metadata and the asset were paired by
        // hand, worth failing loudly rather than packing memory to the wrong layout.
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

        let packed = try await MemoryBankPacker.makePacked(engine: engine)
        let packer = try MemoryBankPacker(
            engine: engine, shapes: resolved, parameters: parameters, packed: packed)
        self.shapes = resolved
        self.packer = packer
        self.tracker = TrackerLoop(
            engine: engine, shapes: resolved, parameters: parameters, packer: packer)
    }

    /// Release the asset. The next call reloads it.
    public func unloadResources() async {
        await engine.unloadResources()
        shapes = nil
        packer = nil
        tracker = nil
    }

    /// Run every entrypoint once on zeros, so the first frame isn't paying for kernel
    /// compilation.
    public func warmup() async throws {
        try await loadResources()
        try await engine.warmup()
    }

    /// Segment and track `prompts` through an already-decoded frame sequence.
    ///
    /// The frames are consumed in the order given and their indices are their positions.
    /// Use this for camera capture, image sequences, or to hold the video decoder constant:
    /// AVFoundation and PyAV disagree by about 0.65 code values on average even on
    /// colour-tagged media, which is enough to move a mask boundary.
    public nonisolated func segment(
        frames: [CGImage],
        prompts: [String],
        parameters overrides: VideoSegmentationParameters? = nil
    ) -> AsyncThrowingStream<VideoSegmentationFrame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(
                        source: .images(frames), prompts: prompts, maxFrames: nil,
                        parameters: overrides ?? self.parameters,
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Segment and track `prompts` through the video at `url`, one result per frame.
    ///
    /// Results arrive in frame order but lag the decoder by `hotstartDelay` frames; the
    /// backlog is flushed when the video ends.
    ///
    /// Nonisolated so a caller can start the stream without an `await`: the frame loop it
    /// spawns hops onto the actor, and everything read here is immutable.
    public nonisolated func segment(
        videoAt url: URL,
        prompts: [String],
        maxFrames: Int? = nil,
        parameters overrides: VideoSegmentationParameters? = nil
    ) -> AsyncThrowingStream<VideoSegmentationFrame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(
                        source: .video(url), prompts: prompts, maxFrames: maxFrames,
                        parameters: overrides ?? self.parameters,
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

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
        // Encoding is serialized behind the writer actor while the next frame's inference
        // runs, so the two overlap naturally without an explicit pipeline.
        for try await frame in segment(
            videoAt: source, prompts: prompts, maxFrames: maxFrames, parameters: effective)
        {
            await onFrame?(frame)
            try await writer.append(renderer.render(frame.objects, onto: frame.image))
            written += 1
        }
        try await writer.finish()
        return written
    }

    // MARK: - Frame loop

    /// Port of `Sam3VideoModel.propagate_in_video_iterator`, forward only.
    ///
    /// Reverse propagation is plumbed through the heuristics but has no entry point here:
    /// it needs the whole video resident so it can start from the last frame, which is the
    /// opposite of how this streams.
    private func run(
        source: FrameSource,
        prompts: [String],
        maxFrames: Int?,
        parameters: VideoSegmentationParameters,
        emit: (VideoSegmentationFrame) -> Void
    ) async throws {
        guard !prompts.isEmpty else { throw VideoSegmentationError.noPrompts }
        try await loadResources()
        guard let shapes, let tracker else {
            throw VideoSegmentationError.invalidConfiguration("Engine failed to initialize.")
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
            engine: engine, shapes: shapes, parameters: parameters, tracker: tracker)
        let postprocessor = MaskPostprocessor(
            lowResolutionSize: shapes.lowResMaskSize, videoWidth: width, videoHeight: height,
            emitLowResolutionMasks: parameters.emitLowResolutionMasks)

        // The hotstart buffer holds decoded frames as well as results: the renderer needs
        // the source image, and re-decoding it later would mean a second pass over the file.
        var buffer: [(raw: RawFrameOutput, image: CGImage, elapsed: Duration)] = []

        func handle(_ image: CGImage, index: Int) async throws {
            try Task.checkCancellation()
            let started = ContinuousClock.now
            let raw = try await processor.process(
                session: session, image: image, frameIndex: index,
                totalFrames: totalFrames, reverse: false)
            let elapsed = ContinuousClock.now - started

            guard parameters.hotstartEnabled else {
                emit(finish(raw, image, elapsed))
                return
            }
            buffer.append((raw, image, elapsed))
            if buffer.count >= parameters.hotstartDelay {
                let (oldest, oldestImage, oldestElapsed) = buffer.removeFirst()
                emit(finish(oldest, oldestImage, oldestElapsed))
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

        // End of video: flush whatever the delay is still holding. These are postprocessed
        // last on purpose, because `hotstartRemovedObjectIDs` has now seen every removal, so
        // a track removed at the very end is hidden in the buffered frames too.
        for (raw, image, elapsed) in buffer {
            emit(finish(raw, image, elapsed))
        }
        lastRunTimings = processor.timings
    }
}
