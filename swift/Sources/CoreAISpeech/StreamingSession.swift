// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Updates

/// One transcription update from a streaming session.
public enum TranscriptionUpdate: Sendable {
    /// Cumulative text for the in-progress segment.
    ///
    /// Append-only: the decoder never revisits a consumed encoder frame, so each partial extends
    /// the previous one and text already shown is never retracted.
    case partial(TranscriptSegment)
    /// A segment closed by an endpoint, the length cap, or end of stream.
    case finalized(TranscriptSegment)
}

public struct TranscriptSegment: Sendable {
    public let text: String
    public let tokens: [Int32]
    public let segmentIndex: Int
    /// Start of the segment's first consumed encoder frame.
    public let startTime: TimeInterval
    /// End of the most recently consumed encoder frame.
    public let endTime: TimeInterval
}

// MARK: - Endpointing

/// Decides when a segment has gone quiet enough to finalize.
///
/// A plain value type so every branch is testable: the streaming path otherwise needs three
/// loaded `AIModel`s.
package struct EndpointDetector {
    package let silenceFrames: Int
    package let maxSegmentFrames: Int
    package private(set) var framesSinceEmission = 0
    package private(set) var segmentFrames = 0

    package init(silenceFrames: Int, maxSegmentFrames: Int) {
        self.silenceFrames = silenceFrames
        self.maxSegmentFrames = maxSegmentFrames
    }

    /// Record a chunk's outcome and report whether to close the segment.
    ///
    /// Two ways to fire, and both land on a pause so the transcript is split at a gap:
    /// `silenceFrames` of quiet, or the first quiet chunk past `maxSegmentFrames`. A hard cut
    /// at the cap would split a word's tokens across segments (`examination` → `exam ination`).
    ///
    /// `silentFrames` is the decoder's duration-weighted count, so the threshold means what it
    /// says at frame resolution. `segmentHasContent` holds the cap at zero until the segment
    /// emits, so a pause cannot spend the next segment's budget before it has any audio.
    package mutating func observe(
        framesAdvanced: Int, silentFrames: Int, segmentHasContent: Bool
    ) -> Bool {
        if segmentHasContent { segmentFrames += framesAdvanced }
        framesSinceEmission = silentFrames
        if framesSinceEmission >= silenceFrames { return true }
        return segmentFrames >= maxSegmentFrames && framesSinceEmission >= 1
    }

    package mutating func reset() {
        framesSinceEmission = 0
        segmentFrames = 0
    }
}

// MARK: - Session state

/// Mutable state for one live streaming session.
///
/// Lives inside the `SpeechRecognitionModel` actor so that `ParakeetTDTDecoder.Stream` and the
/// encoder's `NDArray` outputs never cross an isolation boundary; `@unchecked Sendable` for the
/// same reason as `Stream`.
package final class StreamingSessionState: @unchecked Sendable {
    package let config: StreamingConfig
    let endpointing: EndpointingConfig
    let stream: ParakeetTDTDecoder.Stream
    let decoder: ParakeetTDTDecoder
    var endpoint: EndpointDetector

    /// All PCM pushed so far, trimmed from the front once no window can reach it.
    var pcm: [Float] = []
    /// Absolute sample index of `pcm[0]`, so window arithmetic stays in absolute terms.
    var pcmOrigin = 0
    /// Reused window scratch: the hop's real samples, zero-filled to `windowSampleCount`
    /// while more audio is expected (see `runHopIfReady`).
    var window: [Float]

    /// Diagnostic mode: chunk the encoder but concatenate its outputs and decode once at
    /// the end.
    ///
    /// NeMo's `simulated` flag (`speech_to_text_streaming_infer_rnnt.py:169`) under a name that
    /// says what changes. It shares the encoder path but not the incremental decode, so a
    /// difference against a live stream isolates the state carry or the frame partition.
    let deferredDecode: Bool
    var aggregated: [Float] = []
    var aggregatedFrames = 0

    var hop = 0
    var segmentIndex = 0
    var segmentTokens: [Int32] = []
    var segmentStartFrame = 0
    var lastConsumedFrame = 0
    var finished = false

    let continuation: AsyncThrowingStream<TranscriptionUpdate, Error>.Continuation

    init(
        config: StreamingConfig,
        decoder: ParakeetTDTDecoder,
        tdtConfig: ParakeetTDTConfig,
        endpointing: EndpointingConfig,
        deferredDecode: Bool = false,
        continuation: AsyncThrowingStream<TranscriptionUpdate, Error>.Continuation
    ) {
        self.config = config
        self.endpointing = endpointing
        self.deferredDecode = deferredDecode
        self.decoder = decoder
        self.stream = decoder.makeStream(config: tdtConfig)
        self.endpoint = EndpointDetector(
            silenceFrames: endpointing.silenceFrames,
            maxSegmentFrames: endpointing.maxSegmentFrames)
        self.window = []
        self.window.reserveCapacity(config.windowSampleCount)
        self.continuation = continuation
    }

    var totalSamples: Int { pcmOrigin + pcm.count }
}

// MARK: - Streaming API

extension SpeechRecognitionModel {
    /// Begin a live transcription session and return its update stream.
    ///
    /// Push audio with `append(pcm:)` and close with `finishStream()`. This repo does
    /// not capture audio: a host app owns `AVAudioEngine`, converts to mono float32 at
    /// `sampleRate`, and pushes buffers in.
    ///
    /// Buffered inference over the bundle's traced window — see `StreamingConfig`. Only a
    /// `--streaming` export can stream; `activeStreamingConfig` reports the geometry in use.
    ///
    /// - Parameter endpointing: Where a transcript is cut, and when a long gap resets the
    ///   predictor. Neither changes a tensor shape.
    /// - Parameter deferredDecode: Diagnostic only — chunk the encoder but decode once at the
    ///   end, for diffing against a live stream. It emits no partials, never runs endpointing,
    ///   and holds every consumed frame in memory, so it is not a mode to ship.
    public func startStream(
        endpointing: EndpointingConfig = EndpointingConfig(),
        deferredDecode: Bool = false
    ) throws -> AsyncThrowingStream<TranscriptionUpdate, Error> {
        guard case .parakeetTDT = bundle.kind, let tdtConfig,
            let parakeet = decoder as? ParakeetTDTDecoder
        else {
            throw SpeechError.incompatibleResources(
                "Streaming requires a Parakeet TDT bundle; \(architecture) has no chunked path")
        }
        guard streaming == nil else {
            throw SpeechError.invalidStreamingConfig(
                "a stream is already running — call finishStream() first")
        }
        // Reject `--dynamic` bundles up front. `nFrames` is nil exactly when the encoder's
        // time axis is symbolic, so there is no traced window to derive the geometry from and
        // nothing for `validate(encoderMelFrames:)` to check a config against. The f32 dynamic
        // encoder is separately unreliable on the GPU path at many shapes.
        guard melConfig.nFrames != nil else {
            throw SpeechError.invalidStreamingConfig(
                "this bundle's encoder has a dynamic time axis, so there is no traced window to "
                    + "stream against. Re-export with 'uv run export.py --streaming'.")
        }

        // The traced window *is* the geometry, so a bundle that never recorded one cannot
        // stream. Fitting a caller's preset to whatever window happened to ship would run the
        // session with a left context nobody chose.
        guard let config = bundleStreamingConfig else {
            throw SpeechError.invalidStreamingConfig(
                "this bundle has no streaming geometry. Re-export with "
                    + "'uv run export.py --streaming' — streaming needs a window the export "
                    + "traced and recorded.")
        }
        try config.validate(
            maxDuration: tdtConfig.durations.max() ?? 0, encoderMelFrames: melConfig.nFrames)
        try endpointing.validate(chunkFrames: config.chunkFrames)

        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: TranscriptionUpdate.self, throwing: Error.self)
        streaming = StreamingSessionState(
            config: config, decoder: parakeet, tdtConfig: tdtConfig, endpointing: endpointing,
            deferredDecode: deferredDecode, continuation: continuation)
        return stream
    }

    /// The geometry the running session actually uses, after fitting to the bundle's
    /// traced window. Differs from what was requested whenever the bundle wasn't exported
    /// for streaming.
    public var activeStreamingConfig: StreamingConfig? { streaming?.config }

    /// Push mono float32 PCM at `sampleRate` and run any hops it completes.
    ///
    /// Not realtime-safe — call from a normal `Task`, never from an audio render
    /// callback. A host app should buffer off the audio thread and push from there.
    public func append(pcm samples: [Float]) async throws {
        guard let session = streaming else {
            throw SpeechError.invalidStreamingConfig("no stream is running; call startStream() first")
        }
        session.pcm.append(contentsOf: samples)
        while try await runHopIfReady(session) {}
    }

    /// Flush the tail window, finalize the open segment, and end the update stream.
    ///
    /// The final hop consumes every remaining frame rather than just one chunk, matching
    /// NeMo (`:474-478`). The last word therefore gets no right context — unavoidable at
    /// end of stream.
    @discardableResult
    public func finishStream() async throws -> String {
        guard let session = streaming else {
            throw SpeechError.invalidStreamingConfig("no stream is running")
        }
        session.finished = true
        while try await runHopIfReady(session) {}
        if session.deferredDecode, session.aggregatedFrames > 0 {
            let hidden = tdtConfig?.decoderHiddenSize ?? 0
            let (tokens, _) = try await session.stream.decodeFrames(
                encoderFlat: session.aggregated,
                encoderOutputShape: [1, session.aggregatedFrames, hidden],
                frames: 0..<session.aggregatedFrames,
                windowStartFrame: 0,
                collectStats: false,
                resetAfterSilenceFrames: session.endpointing.resetAfterSilenceFrames)
            session.segmentTokens = tokens
            session.segmentStartFrame = 0
        }
        let text = try emit(session, final: true)
        session.continuation.finish()
        streaming = nil
        return text
    }

    /// Convenience: drive a whole async sequence of buffers to completion.
    public func transcribe<S: AsyncSequence & Sendable>(
        pcmStream: S, endpointing: EndpointingConfig = EndpointingConfig()
    ) throws -> AsyncThrowingStream<TranscriptionUpdate, Error> where S.Element == [Float] {
        let updates = try startStream(endpointing: endpointing)
        Task { [weak self] in
            guard let self else { return }
            do {
                for try await buffer in pcmStream {
                    try await self.append(pcm: buffer)
                }
                _ = try await self.finishStream()
            } catch {
                await self.abandonStream(throwing: error)
            }
        }
        return updates
    }

    /// Drop a failed session and hand the error to the stream's consumer.
    ///
    /// Not `finishStream()`: that flushes and calls `finish()`, and the first finish wins — the
    /// consumer would see a clean end of stream instead of the failure.
    private func abandonStream(throwing error: Error) {
        let continuation = streaming?.continuation
        streaming = nil
        continuation?.finish(throwing: error)
    }

    // MARK: - Internals

    /// Run one hop if enough audio has arrived. Returns false when it needs more.
    private func runHopIfReady(_ session: StreamingSessionState) async throws -> Bool {
        let cfg = session.config
        let windowStartFrame = cfg.windowStartFrame(hop: session.hop)
        let windowStartSample = windowStartFrame * cfg.samplesPerEncoderFrame
        let consume = cfg.consumeRange(hop: session.hop)

        // Wait for the chunk plus its right context, matching NeMo's initial-latency
        // gate (`:429`). At end of stream, take whatever is left.
        let available = session.totalSamples
        if !session.finished, available < cfg.requiredSampleCount(hop: session.hop) {
            return false
        }
        if windowStartSample >= available { return false }

        // How much real audio this window covers. Tail padding (never front) keeps
        // `attention_mask` a prefix, the only shape HF's mask and the subsampling channel mask
        // can express, so a window is always `[audio | zeros]`.
        let validSamples = min(cfg.windowSampleCount, available - windowStartSample)
        if validSamples <= 0 { return false }
        let localStart = windowStartSample - session.pcmOrigin
        guard localStart >= 0, localStart + validSamples <= session.pcm.count else {
            throw SpeechError.invalidStreamingConfig(
                "window [\(windowStartSample), +\(validSamples)) is outside the retained "
                    + "buffer starting at \(session.pcmOrigin)")
        }
        session.window.removeAll(keepingCapacity: true)
        session.window.append(contentsOf: session.pcm[localStart..<localStart + validSamples])

        let isLastHop = session.finished && windowStartSample + validSamples >= available

        // Zero-fill to the traced size while more audio is expected: the encoder is non-causal,
        // so a growing window decodes the session's opening under a different regime than steady
        // state. Never on the final hop, where the zeros mean "no more speech" and masking them
        // honestly is what cues the sentence-final token.
        let padWindow = !isLastHop && validSamples < cfg.windowSampleCount
        if padWindow {
            session.window.append(
                contentsOf: repeatElement(0, count: cfg.windowSampleCount - validSamples))
        }

        // The encoder's own count now covers the padding, so recompute what real audio backs:
        // frames are consumed only where they are, padded window or not.
        let (encOut, encShape, encoderValidEnc) = try await runEncoder(pcm: session.window)
        let validEnc =
            padWindow
            ? Self.validEncoderFrames(
                pcmCount: validSamples, tEnc: encShape[1], config: melConfig,
                subsamplingFactor: tdtConfig?.encoderSubsamplingFactor ?? 1)
            : encoderValidEnc

        // Per-chunk accounting against the frames real audio backs, never a proportion of the
        // window: a boundary-frame rounding error would lose or invent a frame every hop.
        let frameCeiling = windowStartFrame + validEnc
        let upper = isLastHop ? frameCeiling : min(consume.upperBound, frameCeiling)
        let lower = min(consume.lowerBound, upper)

        if lower < upper, session.deferredDecode {
            aggregateChunk(
                session, encOut: encOut, frames: lower..<upper,
                windowStartFrame: windowStartFrame)
        } else if lower < upper {
            try await consumeChunk(
                session, encOut: encOut, encShape: encShape, frames: lower..<upper,
                windowStartFrame: windowStartFrame)
        }

        session.hop += 1
        // Release audio no future window can reach.
        let keepFrom = cfg.windowStartSample(hop: session.hop)
        if keepFrom > session.pcmOrigin {
            let drop = min(keepFrom - session.pcmOrigin, session.pcm.count)
            session.pcm.removeFirst(drop)
            session.pcmOrigin += drop
        }
        return !isLastHop
    }

    /// Keep only the chunk's frames, exactly as the streaming path consumes them, and defer
    /// all decoding to `finishStream()`.
    private func aggregateChunk(
        _ session: StreamingSessionState, encOut: NDArray, frames: Range<Int>,
        windowStartFrame: Int
    ) {
        let hidden = tdtConfig?.decoderHiddenSize ?? 0
        let lo = (frames.lowerBound - windowStartFrame) * hidden
        let hi = (frames.upperBound - windowStartFrame) * hidden
        session.aggregated.append(contentsOf: floatElements(encOut, in: lo..<hi))
        session.aggregatedFrames += frames.count
        session.lastConsumedFrame = frames.upperBound
    }

    /// Decode the chunk's frames, then endpoint and emit on what came back.
    private func consumeChunk(
        _ session: StreamingSessionState, encOut: NDArray, encShape: [Int], frames: Range<Int>,
        windowStartFrame: Int
    ) async throws {
        let (tokens, _) = try await session.stream.decodeFrames(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            frames: frames,
            windowStartFrame: windowStartFrame,
            collectStats: false,
            resetAfterSilenceFrames: session.endpointing.resetAfterSilenceFrames)

        if session.segmentTokens.isEmpty && !tokens.isEmpty {
            session.segmentStartFrame = frames.lowerBound
        }
        session.segmentTokens.append(contentsOf: tokens)
        session.lastConsumedFrame = frames.upperBound

        let shouldEndpoint = session.endpoint.observe(
            framesAdvanced: frames.count, silentFrames: session.stream.silentFrames,
            segmentHasContent: !session.segmentTokens.isEmpty)

        if shouldEndpoint, !session.segmentTokens.isEmpty {
            _ = try emit(session, final: true)
            // A display boundary only: the transducer state carries straight through.
            // Zeroing the LSTM at every endpoint instead cost ~3.8 s of dropped audio while
            // it re-established context.
            session.endpoint.reset()
            session.segmentIndex += 1
            session.segmentTokens = []
            session.segmentStartFrame = frames.upperBound
        } else if !tokens.isEmpty {
            _ = try emit(session, final: false)
        }
    }

    private func emit(_ session: StreamingSessionState, final: Bool) throws -> String {
        let text = try detokenize(session.segmentTokens)
        guard !session.segmentTokens.isEmpty || final else { return text }
        let segment = TranscriptSegment(
            text: text,
            tokens: session.segmentTokens,
            segmentIndex: session.segmentIndex,
            startTime: session.config.seconds(frame: session.segmentStartFrame),
            endTime: session.config.seconds(frame: session.lastConsumedFrame))
        if !text.isEmpty {
            session.continuation.yield(final ? .finalized(segment) : .partial(segment))
        }
        return text
    }
}
