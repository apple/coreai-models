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
    /// Append-only within a segment: because the decoder never revisits a consumed
    /// encoder frame, each partial is a prefix-extension of the previous one and text
    /// already shown is never retracted.
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
/// Extracted as a plain value type so every branch is testable: the streaming path
/// otherwise needs three loaded `AIModel`s, the same reasoning behind
/// `ParakeetTDTDecoder.validate` being `package static`.
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
    ///
    /// - `silenceFrames` of quiet — the ordinary endpoint.
    /// - past `maxSegmentFrames`, the *first* quiet chunk. A hard cut at the cap would
    ///   land mid-word: it splits a token sequence that detokenizes as one word into two
    ///   segments, and joining them reinserts a space (`examination` became `exam
    ///   ination`). Waiting for any pause keeps the bound without corrupting a word.
    ///
    /// `silentFrames` is the decoder's own duration-weighted count of frames since its last
    /// emission, so the threshold means what it says at frame resolution. Accumulating a
    /// whole chunk per hop instead made the effective rule "one hop emitted nothing" — with
    /// the default 12-frame chunk against a 10-frame threshold, any single quiet hop fired an
    /// endpoint mid-utterance.
    package mutating func observe(framesAdvanced: Int, silentFrames: Int) -> Bool {
        segmentFrames += framesAdvanced
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
/// Lives inside the `SpeechRecognitionModel` actor rather than in a separate actor so
/// that `ParakeetTDTDecoder.Stream` and the encoder's `NDArray` outputs never cross an
/// isolation boundary. `@unchecked Sendable` for the same reason as `Stream`: this is
/// only ever reached from the owning actor's isolated state.
package final class StreamingSessionState: @unchecked Sendable {
    package let config: StreamingConfig
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
    /// This is NeMo's `simulated` flag under a name that says what changes
    /// (`speech_to_text_streaming_infer_rnnt.py:169`, aggregation at `:480-490`, the single
    /// decode at `:529-554`), whose own comment reads "encoder is evaluated on chunks, output
    /// is concatenated and decoded at one step / expected to provide the same results".
    ///
    /// That expectation is the point: this shares the encoder path but not the incremental
    /// decode, so a difference against a live stream isolates the state carry, the
    /// duration-overshoot carry, or the frame partition — no quality judgement needed.
    let deferredDecode: Bool
    var aggregated: [Float] = []
    var aggregatedFrames = 0

    var hop = 0
    var segmentIndex = 0
    var segmentTokens: [Int32] = []
    var segmentStartFrame = 0
    var lastConsumedFrame = 0
    var finished = false

    let continuation: AsyncStream<TranscriptionUpdate>.Continuation

    init(
        config: StreamingConfig,
        decoder: ParakeetTDTDecoder,
        tdtConfig: ParakeetTDTConfig,
        deferredDecode: Bool = false,
        continuation: AsyncStream<TranscriptionUpdate>.Continuation
    ) {
        self.config = config
        self.deferredDecode = deferredDecode
        self.decoder = decoder
        self.stream = decoder.makeStream(config: tdtConfig)
        self.endpoint = EndpointDetector(
            silenceFrames: config.endpointSilenceFrames,
            maxSegmentFrames: config.maxSegmentFrames)
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
    /// The encoder runs over a bounded `[left | chunk | right]` window each hop and only
    /// the chunk's frames are decoded, with the transducer state carried across hops —
    /// NVIDIA's buffered-inference algorithm (NeMo
    /// `speech_to_text_streaming_infer_rnnt.py:446-527`). Parakeet TDT v3 has no
    /// cache-aware encoder, so this is how streaming is done for it.
    ///
    /// - Parameter config: Window geometry. Defaults to `.balanced` (1.92 s theoretical
    ///   latency). If the bundle's encoder was traced for a different window, the left
    ///   context is widened to fit it, keeping `chunk` and `right` exact.
    /// - Parameter deferredDecode: Diagnostic only — chunk the encoder but decode once at the
    ///   end, for diffing against a live stream. It emits no partials, never runs endpointing,
    ///   and holds every consumed frame in memory,, so it is not a mode to ship.
    public func startStream(
        config requested: StreamingConfig = .balanced,
        deferredDecode: Bool = false
    ) throws -> AsyncStream<TranscriptionUpdate> {
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
                "this bundle's encoder has a dynamic time axis. Streaming needs a fixed "
                    + "traced window: export with --streaming (preferred) or plain static, "
                    + "not --dynamic.")
        }

        // A bundle exported with --streaming already carries the geometry it was traced
        // for; that is authoritative, and only the endpointing knobs come from the
        // caller. Otherwise fit the requested preset to whatever window shipped.
        var config = requested
        if let bundled = bundleStreamingConfig {
            config = StreamingConfig(
                windowMelFrames: bundled.windowMelFrames,
                chunkFrames: bundled.chunkFrames,
                rightContextFrames: bundled.rightContextFrames,
                endpointSilenceFrames: requested.endpointSilenceFrames,
                maxSegmentFrames: requested.maxSegmentFrames,
                hopLength: bundled.hopLength,
                subsamplingFactor: bundled.subsamplingFactor,
                sampleRate: bundled.sampleRate)
        }
        if let melFrames = melConfig.nFrames, melFrames != config.windowMelFrames {
            guard let fitted = requested.fitting(encoderMelFrames: melFrames) else {
                throw SpeechError.invalidStreamingConfig(
                    "encoder traced for \(melFrames) mel frames, which cannot hold chunk "
                        + "\(requested.chunkFrames) + right \(requested.rightContextFrames) "
                        + "encoder frames with at least that much left context")
            }
            config = fitted
            CLILogger.log(
                "Streaming: fitted left context to \(config.leftContextFrames) frames "
                    + "(\(String(format: "%.2f", config.seconds(frames: config.leftContextFrames))) s) "
                    + "for this bundle's \(melFrames)-frame window", level: 1)
        }
        try config.validate(
            maxDuration: tdtConfig.durations.max() ?? 0, encoderMelFrames: melConfig.nFrames)

        let (stream, continuation) = AsyncStream.makeStream(of: TranscriptionUpdate.self)
        streaming = StreamingSessionState(
            config: config, decoder: parakeet, tdtConfig: tdtConfig, deferredDecode: deferredDecode,
            continuation: continuation)
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
                windowStartFrame: 0)
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
        pcmStream: S, config: StreamingConfig = .balanced
    ) throws -> AsyncStream<TranscriptionUpdate> where S.Element == [Float] {
        let updates = try startStream(config: config)
        Task { [weak self] in
            guard let self else { return }
            do {
                for try await buffer in pcmStream {
                    try await self.append(pcm: buffer)
                }
                _ = try await self.finishStream()
            } catch {
                _ = try? await self.finishStream()
            }
        }
        return updates
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
        // `attention_mask` a *prefix*, the only shape HF's mask and the subsampling channel
        // mask can express, so a window is always `[audio | zeros]`.
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

        // Zero-fill the window up to the size it was traced for while more audio is still
        // expected. The encoder is full-attention and non-causal, so a frame's representation
        // depends on how much audio surrounds it: through ramp-up a growing window decodes the
        // opening of a session under a different regime than steady state, and the transducer
        // then consumes a sequence stitched from mismatched representations.
        //
        // Never on the final hop: there the zeros mean "no more speech" rather than "audio not
        // yet received", and masking them honestly is what cues the sentence-final token —
        // padding it dropped the closing period.
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
            // Keep only the chunk's frames, exactly as the streaming path consumes them,
            // and defer all decoding to finishStream().
            let hidden = tdtConfig?.decoderHiddenSize ?? 0
            let flat = flattenAsFloat(encOut)
            let lo = (lower - windowStartFrame) * hidden
            let hi = (upper - windowStartFrame) * hidden
            session.aggregated.append(contentsOf: flat[lo..<hi])
            session.aggregatedFrames += upper - lower
            session.lastConsumedFrame = upper
        } else if lower < upper {
            let (tokens, stats) = try await session.stream.decodeFrames(
                encoderOutput: encOut,
                encoderOutputShape: encShape,
                frames: lower..<upper,
                windowStartFrame: windowStartFrame)

            if session.segmentTokens.isEmpty && !tokens.isEmpty {
                session.segmentStartFrame = lower
            }
            session.segmentTokens.append(contentsOf: tokens)
            session.lastConsumedFrame = upper

            let shouldEndpoint = session.endpoint.observe(
                framesAdvanced: upper - lower, silentFrames: session.stream.silentFrames)
            _ = stats

            if shouldEndpoint, !session.segmentTokens.isEmpty {
                _ = try emit(session, final: true)
                // A segment boundary is a *display* boundary: the transducer state carries
                // straight through it. Zeroing the LSTM here instead cost ~48 encoder frames
                // (3.8 s) of dropped audio while it resynced from SOS — measured as three
                // segments and 105 tokens where carrying the state gives one continuous
                // decode and 119. Callers wanting a hard reset can still use `resetSegment`.
                session.endpoint.reset()
                session.segmentIndex += 1
                session.segmentTokens = []
                session.segmentStartFrame = upper
            } else if !tokens.isEmpty {
                _ = try emit(session, final: false)
            }
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
