// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - StreamingConfig

/// Window geometry for buffered ("streaming") Parakeet TDT inference.
///
/// Parakeet TDT v3 is an offline full-context FastConformer — its attention is
/// bidirectional over the whole utterance and there is no cache-aware variant in
/// `transformers`. So live transcription re-runs the *whole* encoder over a bounded
/// window `[left | chunk | right]` each hop, consumes only the chunk's encoder
/// frames, and carries the transducer state across hops.
///
/// This ports NVIDIA's own buffered-inference algorithm (NeMo
/// `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`,
/// main loop at :446-527), which its own example applies to non-cache-aware
/// checkpoints — so this is a supported upstream mode, not a workaround.
///
/// Everything is expressed in **encoder frames**, because that is the only unit in
/// which the chunk boundary is exact. One encoder frame is
/// `hopLength * subsamplingFactor` samples (1280 = 80 ms for Parakeet).
public struct StreamingConfig: Sendable, Equatable {
    /// Frames consumed per hop. Sets the emission cadence. Must be exact — the sliding
    /// arithmetic depends on it.
    public let chunkFrames: Int
    /// Frames of *future* audio the encoder sees but does not decode. The single most
    /// valuable knob for a non-causal encoder, and it is what costs latency.
    public let rightContextFrames: Int
    /// Mel frames the encoder graph is traced for — the primary quantity, because it is
    /// the one the exported graph fixes. Everything else derives from it.
    public let windowMelFrames: Int
    /// Consecutive silent frames before a segment is finalized.
    public let endpointSilenceFrames: Int
    /// Soft cap on segment length. Splits the transcript for display without
    /// resetting the transducer, and keeps a segment inside one offline encoder run for
    /// any later rescoring pass.
    public let maxSegmentFrames: Int

    public let samplesPerEncoderFrame: Int
    public let hopLength: Int
    public let sampleRate: Double

    public var subsamplingFactor: Int { samplesPerEncoderFrame / hopLength }

    /// PCM samples per window: the most audio whose mel length is still exactly
    /// `windowMelFrames`, since `frameCount` is `1 + N/hop`.
    public var windowSampleCount: Int { (windowMelFrames - 1) * hopLength }

    /// Mel frames carrying real audio, i.e. all but the zero-padded remainder.
    public var validWindowMelFrames: Int { windowMelFrames - 1 }

    /// Encoder frames the window can trust.
    ///
    /// Not simply `windowMelFrames / subsampling`: the graph emits
    /// `ceil(windowMelFrames / 8)`, of which the last covers padding, so this is the
    /// count over the *valid* mel frames.
    public var usableEncoderFrames: Int {
        encoderFrameCount(melFrames: validWindowMelFrames, subsamplingFactor: subsamplingFactor)
    }

    /// Encoder frames the graph actually emits.
    public var windowEncoderFrames: Int {
        encoderFrameCount(melFrames: windowMelFrames, subsamplingFactor: subsamplingFactor)
    }

    /// Frames of past audio the encoder sees but does not decode. Improves quality
    /// without affecting latency.
    ///
    /// Derived rather than stored: `chunk` and `right` must be exact, so left context is
    /// what absorbs whatever window the bundle was actually traced for.
    public var leftContextFrames: Int {
        usableEncoderFrames - chunkFrames - rightContextFrames
    }

    /// Theoretical latency before a word can be emitted: `chunk + right`. Inference
    /// time is on top. Matches NeMo's definition (`:26`, `:362`).
    public var theoreticalLatency: TimeInterval {
        seconds(frames: chunkFrames + rightContextFrames)
    }

    /// Build from a desired left context, sizing the window to hold it.
    public init(
        leftContextFrames: Int,
        chunkFrames: Int,
        rightContextFrames: Int,
        endpointSilenceFrames: Int = 10,
        maxSegmentFrames: Int = 375,
        hopLength: Int = 160,
        subsamplingFactor: Int = 8,
        sampleRate: Double = 16_000
    ) {
        self.init(
            windowMelFrames: (leftContextFrames + chunkFrames + rightContextFrames) * subsamplingFactor + 1,
            chunkFrames: chunkFrames,
            rightContextFrames: rightContextFrames,
            endpointSilenceFrames: endpointSilenceFrames,
            maxSegmentFrames: maxSegmentFrames,
            hopLength: hopLength,
            subsamplingFactor: subsamplingFactor,
            sampleRate: sampleRate)
    }

    /// Build from a traced window, which is what a loaded bundle gives us.
    public init(
        windowMelFrames: Int,
        chunkFrames: Int,
        rightContextFrames: Int,
        endpointSilenceFrames: Int = 10,
        maxSegmentFrames: Int = 375,
        hopLength: Int = 160,
        subsamplingFactor: Int = 8,
        sampleRate: Double = 16_000
    ) {
        self.windowMelFrames = windowMelFrames
        self.chunkFrames = chunkFrames
        self.rightContextFrames = rightContextFrames
        self.endpointSilenceFrames = endpointSilenceFrames
        self.maxSegmentFrames = maxSegmentFrames
        self.hopLength = hopLength
        self.samplesPerEncoderFrame = hopLength * subsamplingFactor
        self.sampleRate = sampleRate
    }

    // MARK: Presets

    /// 10.08 s left / 0.96 s chunk / 0.96 s right — 1.92 s theoretical latency,
    /// 12.0 s window (1201 mel frames, 151 encoder frames).
    ///
    /// Upstream recommends 10-2-2 (`:31`), but measurement on this checkpoint found
    /// 10-1-1 no worse at half the latency, and the encoder costs ~54 ms per hop
    /// against a 960 ms budget — so throughput is not what should set this.
    public static let balanced = StreamingConfig(
        leftContextFrames: 126, chunkFrames: 12, rightContextFrames: 12)

    /// 10.0 s left / 2.0 s chunk / 2.0 s right — 4.0 s latency, 14.0 s window
    /// (1401 mel frames, 176 encoder frames). Upstream's recommended setting.
    public static let accuracy = StreamingConfig(
        leftContextFrames: 125, chunkFrames: 25, rightContextFrames: 25)

    // MARK: Bundle metadata

    /// Decode the `streaming` block a `--streaming` export writes into `metadata.json`.
    ///
    /// Returns nil for bundles without the block — every bundle exported before
    /// streaming existed — so `startStream` falls back to fitting a preset to the
    /// traced window.
    package static func decode(fromMetadata raw: Data) -> StreamingConfig? {
        guard let payload = try? JSONDecoder().decode(MetadataPayload.self, from: raw),
            let block = payload.streaming
        else { return nil }
        return StreamingConfig(
            windowMelFrames: block.windowMelFrames,
            chunkFrames: block.chunkEncoderFrames,
            rightContextFrames: block.rightContextEncoderFrames,
            hopLength: block.hopLength,
            subsamplingFactor: block.subsamplingFactor,
            sampleRate: Double(block.sampleRate))
    }

    fileprivate struct MetadataPayload: Decodable {
        let streaming: StreamingBlock?
    }

    fileprivate struct StreamingBlock: Decodable {
        let chunkEncoderFrames: Int
        let rightContextEncoderFrames: Int
        let windowMelFrames: Int
        let hopLength: Int
        let subsamplingFactor: Int
        let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case chunkEncoderFrames = "chunk_encoder_frames"
            case rightContextEncoderFrames = "right_context_encoder_frames"
            case windowMelFrames = "window_mel_frames"
            case hopLength = "hop_length"
            case subsamplingFactor = "subsampling_factor"
            case sampleRate = "sample_rate"
        }
    }

    // MARK: Validation

    /// Preconditions the hop arithmetic and the `timeJump` carry depend on.
    ///
    /// `static`-friendly and pure so every rejection is reachable from a test — the
    /// streaming path otherwise needs three loaded `AIModel`s, mirroring the reasoning
    /// behind `ParakeetTDTDecoder.validate`.
    public func validate(maxDuration: Int, encoderMelFrames: Int?) throws {
        guard chunkFrames >= 1 else {
            throw SpeechError.invalidStreamingConfig("chunkFrames must be >= 1, got \(chunkFrames)")
        }
        guard rightContextFrames >= 0, leftContextFrames >= 0 else {
            throw SpeechError.invalidStreamingConfig("context frame counts must be non-negative")
        }
        // A TDT duration can advance the frame pointer past the chunk end; the debt is
        // carried into the next hop (see ParakeetTDTDecoder.Stream). For the debt-carrying
        // frame to land at a non-negative local index in the *next* window, the window
        // must start no later than the frame we stopped on — which needs left >= chunk.
        guard leftContextFrames >= chunkFrames else {
            throw SpeechError.invalidStreamingConfig(
                "leftContextFrames (\(leftContextFrames)) must be >= chunkFrames "
                    + "(\(chunkFrames)) so a duration overshoot stays inside the next window")
        }
        // An overshoot must not point past the window's valid frames while the loop is
        // still running, so the right context has to cover the largest single jump.
        guard rightContextFrames >= maxDuration else {
            throw SpeechError.invalidStreamingConfig(
                "rightContextFrames (\(rightContextFrames)) must be >= the largest TDT "
                    + "duration (\(maxDuration))")
        }
        guard endpointSilenceFrames >= 1, maxSegmentFrames >= chunkFrames else {
            throw SpeechError.invalidStreamingConfig(
                "endpointSilenceFrames must be >= 1 and maxSegmentFrames >= chunkFrames")
        }
        // The load-bearing one: the graph the bundle actually shipped must match the
        // geometry this config claims. A one-frame mismatch is 80 ms of audio and would
        // drop or duplicate words at every boundary.
        if let melFrames = encoderMelFrames, melFrames != windowMelFrames {
            throw SpeechError.invalidStreamingConfig(
                "encoder traced for \(melFrames) mel frames but this config is built for "
                    + "\(windowMelFrames). Build the config with "
                    + "StreamingConfig(windowMelFrames:) or let startStream() fit it.")
        }
    }

    /// The same config resized to an encoder traced for `melFrames` mel frames.
    ///
    /// Lets a streaming session run against a plain `_static` bundle whose window was
    /// never chosen with streaming in mind: `chunk` and `right` stay exact and left
    /// context takes the remainder. Returns nil if the window cannot even hold
    /// `chunk + right` with that much left context.
    public func fitting(encoderMelFrames melFrames: Int) -> StreamingConfig? {
        let resized = StreamingConfig(
            windowMelFrames: melFrames,
            chunkFrames: chunkFrames,
            rightContextFrames: rightContextFrames,
            endpointSilenceFrames: endpointSilenceFrames,
            maxSegmentFrames: maxSegmentFrames,
            hopLength: hopLength,
            subsamplingFactor: subsamplingFactor,
            sampleRate: sampleRate)
        guard resized.leftContextFrames >= chunkFrames else { return nil }
        return resized
    }

    // MARK: Hop geometry

    /// First global encoder frame of the window at `hop`.
    ///
    /// Clamps to 0 during ramp-up, which is what makes early hops see *more* left
    /// context than steady state rather than less.
    public func windowStartFrame(hop: Int) -> Int {
        max(0, hop * chunkFrames - leftContextFrames)
    }

    /// First PCM sample of the window at `hop`.
    public func windowStartSample(hop: Int) -> Int {
        windowStartFrame(hop: hop) * samplesPerEncoderFrame
    }

    /// Global encoder frames the hop consumes — exactly the chunk, no heuristic.
    public func consumeRange(hop: Int) -> Range<Int> {
        let start = hop * chunkFrames
        return start..<(start + chunkFrames)
    }

    /// Index of the chunk's first frame *within* the hop's encoder output.
    public func localConsumeStart(hop: Int) -> Int {
        consumeRange(hop: hop).lowerBound - windowStartFrame(hop: hop)
    }

    /// Samples that must have arrived before `hop` can run: the chunk plus its right
    /// context. Matches NeMo's initial-latency gate (`:429`).
    public func requiredSampleCount(hop: Int) -> Int {
        (consumeRange(hop: hop).upperBound + rightContextFrames) * samplesPerEncoderFrame
    }

    /// Start time of global encoder frame `frame`.
    ///
    /// Encoder frame `j` is centred on mel frame `8j`, so frame 0 is at t=0.
    public func seconds(frame: Int) -> TimeInterval {
        Double(frame) * Double(samplesPerEncoderFrame) / sampleRate
    }

    public func seconds(frames: Int) -> TimeInterval { seconds(frame: frames) }
}

// MARK: - Subsampling

/// Encoder frames emitted for `melFrames` mel frames.
///
/// The FastConformer subsampling stack is three stride-2, kernel-3, pad-1 convs, each
/// `floor((L + 2·1 − 3)/2) + 1 = floor((L−1)/2) + 1 = ceil(L/2)`, which composes to
/// `ceil(L/8)`. Mirrors HF `ParakeetPreTrainedModel._get_subsampling_output_length`
/// rather than hardcoding the closed form, so an unusual `subsamplingFactor` still
/// computes the right answer.
public func encoderFrameCount(melFrames: Int, subsamplingFactor: Int) -> Int {
    guard melFrames > 0, subsamplingFactor > 1 else { return max(0, melFrames) }
    var length = melFrames
    var factor = subsamplingFactor
    while factor > 1 {
        length = (length - 1) / 2 + 1
        factor /= 2
    }
    return length
}
