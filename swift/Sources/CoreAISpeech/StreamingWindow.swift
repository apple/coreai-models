// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - EndpointingConfig

/// When a streaming segment is closed for display.
///
/// Separate from `StreamingConfig` because it changes no tensor shape: the window is fixed by
/// the traced graph, but where a transcript is cut is a caller's policy — dictation wants a long
/// pause tolerance, live captions a short one.
///
/// Counts encoder frames, matching `StreamingConfig`. The defaults are wall-clock judgements
/// calibrated at 80 ms per frame, every Parakeet bundle's frame duration; scale them if a bundle
/// ever ships a different one.
public struct EndpointingConfig: Sendable, Equatable {
    /// Frames of decoder silence, duration-weighted, before a segment is finalized. 10 frames
    /// is 0.8 s at 80 ms per frame.
    public var silenceFrames: Int
    /// Soft cap on segment length, 375 frames being 30 s. Splits the transcript for display
    /// without resetting the transducer, and waits for a pause so no word is cut.
    public var maxSegmentFrames: Int
    /// Silent frames after which the predictor is reset, or 0 to never. 40 frames is 3.2 s.
    ///
    /// Past a long gap the predictor resists re-entering an emitting state, so the resuming
    /// utterance loses its opening words. Well above `silenceFrames`, which closes a segment for
    /// display only: resetting at every endpoint cost ~3.8 s of dropped audio. NeMo never resets
    /// (`speech_to_text_streaming_infer_rnnt.py:426`), so 0 matches upstream.
    ///
    /// Assumes the host pushes audio through pauses: this and `silenceFrames` advance only when
    /// hops run. A host that gates its mic should call `finishStream()` rather than withhold
    /// samples, which freezes the session instead of pausing it.
    public var resetAfterSilenceFrames: Int

    public init(
        silenceFrames: Int = 10, maxSegmentFrames: Int = 375, resetAfterSilenceFrames: Int = 40
    ) {
        self.silenceFrames = silenceFrames
        self.maxSegmentFrames = maxSegmentFrames
        self.resetAfterSilenceFrames = resetAfterSilenceFrames
    }

    /// Checked against the geometry it will run with: a cap below one chunk could never be
    /// reached at a pause, and zero silence frames would endpoint on every hop.
    public func validate(chunkFrames: Int) throws {
        guard silenceFrames >= 1, maxSegmentFrames >= chunkFrames else {
            throw SpeechError.invalidStreamingConfig(
                "silenceFrames must be >= 1 and maxSegmentFrames (\(maxSegmentFrames)) must be "
                    + ">= the bundle's chunk (\(chunkFrames))")
        }
        // At or below the endpoint threshold this fires on every segment boundary, the
        // configuration measured to drop ~3.8 s of audio.
        guard resetAfterSilenceFrames == 0 || resetAfterSilenceFrames > silenceFrames else {
            throw SpeechError.invalidStreamingConfig(
                "resetAfterSilenceFrames (\(resetAfterSilenceFrames)) must be 0 or greater than "
                    + "silenceFrames (\(silenceFrames)) — resetting at every endpoint drops "
                    + "audio while the predictor re-establishes context")
        }
    }
}

// MARK: - StreamingConfig

/// Window geometry for buffered ("streaming") Parakeet TDT inference.
///
/// Parakeet TDT v3 is an offline full-context FastConformer with no cache-aware variant in
/// `transformers`, so live transcription re-runs the whole encoder over a bounded
/// `[left | chunk | right]` window each hop, consumes only the chunk's frames, and carries the
/// transducer state across hops — NVIDIA's own algorithm (NeMo
/// `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py:446-527`).
///
/// Everything is in encoder frames, the only unit in which the chunk boundary is exact. One
/// frame is `hopLength * subsamplingFactor` samples (1280 = 80 ms for Parakeet).
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

    /// Frames of past audio the encoder sees but does not decode. Improves quality without
    /// affecting latency. The steady-state target — NeMo's `expected_context.left`; the effective
    /// left is smaller during ramp-up, because `windowStartFrame` clamps to 0.
    ///
    /// Derived rather than stored, which makes `left + chunk + right <= usable` identically true.
    /// A stored left that disagreed would start the window past the audio or leave part of a
    /// chunk unconsumed; `decode(fromMetadata:)` cross-checks the recorded copy instead.
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
        hopLength: Int = 160,
        subsamplingFactor: Int = 8,
        sampleRate: Double = 16_000
    ) {
        self.init(
            windowMelFrames: (leftContextFrames + chunkFrames + rightContextFrames) * subsamplingFactor + 1,
            chunkFrames: chunkFrames,
            rightContextFrames: rightContextFrames,
            hopLength: hopLength,
            subsamplingFactor: subsamplingFactor,
            sampleRate: sampleRate)
    }

    /// Build from a traced window, which is what a loaded bundle gives us.
    public init(
        windowMelFrames: Int,
        chunkFrames: Int,
        rightContextFrames: Int,
        hopLength: Int = 160,
        subsamplingFactor: Int = 8,
        sampleRate: Double = 16_000
    ) {
        self.windowMelFrames = windowMelFrames
        self.chunkFrames = chunkFrames
        self.rightContextFrames = rightContextFrames
        self.hopLength = hopLength
        self.samplesPerEncoderFrame = hopLength * subsamplingFactor
        self.sampleRate = sampleRate
    }

    // MARK: Bundle metadata

    /// Decode the `streaming` block a `--streaming` export writes into `metadata.json`.
    ///
    /// `nil` for a bundle without the block — it cannot stream, and `startStream` says so. Throws
    /// when the block disagrees with itself, the one place the recorded left context is checked.
    package static func decode(fromMetadata raw: Data) throws -> StreamingConfig? {
        guard let payload = try? JSONDecoder().decode(MetadataPayload.self, from: raw),
            let block = payload.streaming
        else { return nil }
        // Checked before the config is built: every frame count below runs through
        // `encoderFrameCount`, and a zero hop would divide by zero in `subsamplingFactor`.
        guard block.hopLength > 0, isValidSubsamplingFactor(block.subsamplingFactor) else {
            throw SpeechError.invalidStreamingConfig(
                "streaming metadata has hop_length \(block.hopLength) and subsampling_factor "
                    + "\(block.subsamplingFactor); the hop must be positive and the factor a "
                    + "power of two (the encoder's front end is a stack of stride-2 convs)")
        }
        let config = StreamingConfig(
            windowMelFrames: block.windowMelFrames,
            chunkFrames: block.chunkEncoderFrames,
            rightContextFrames: block.rightContextEncoderFrames,
            hopLength: block.hopLength,
            subsamplingFactor: block.subsamplingFactor,
            sampleRate: Double(block.sampleRate))
        guard block.leftContextEncoderFrames == config.leftContextFrames else {
            throw SpeechError.invalidStreamingConfig(
                "metadata records left context \(block.leftContextEncoderFrames) but its window "
                    + "leaves \(config.leftContextFrames) after chunk \(config.chunkFrames) and "
                    + "right \(config.rightContextFrames). The block disagrees with itself; "
                    + "re-export rather than editing it.")
        }
        return config
    }

    fileprivate struct MetadataPayload: Decodable {
        let streaming: StreamingBlock?
    }

    fileprivate struct StreamingBlock: Decodable {
        let leftContextEncoderFrames: Int
        let chunkEncoderFrames: Int
        let rightContextEncoderFrames: Int
        let windowMelFrames: Int
        let hopLength: Int
        let subsamplingFactor: Int
        let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case leftContextEncoderFrames = "left_context_encoder_frames"
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
/// Three stride-2, kernel-3, pad-1 convs, each `ceil(L/2)`, composing to `ceil(L/8)`. Mirrors HF
/// `ParakeetPreTrainedModel._get_subsampling_output_length` rather than hardcoding the closed
/// form, so a bundle with a different power-of-two factor still computes the right answer.
///
/// `subsamplingFactor` must be a power of two — all a stack of stride-2 convs can express. The
/// loop halves, so a factor of 6 would otherwise apply silently as 4. Both metadata decoders
/// reject a bad factor first, so this traps only a programming error.
public func encoderFrameCount(melFrames: Int, subsamplingFactor: Int) -> Int {
    guard melFrames > 0, subsamplingFactor > 1 else { return max(0, melFrames) }
    precondition(
        subsamplingFactor & (subsamplingFactor - 1) == 0,
        "subsamplingFactor must be a power of two, got \(subsamplingFactor)")
    var length = melFrames
    var factor = subsamplingFactor
    while factor > 1 {
        length = (length - 1) / 2 + 1
        factor /= 2
    }
    return length
}

/// Whether `factor` is a subsampling factor `encoderFrameCount` can express.
package func isValidSubsamplingFactor(_ factor: Int) -> Bool {
    factor > 0 && factor & (factor - 1) == 0
}
