// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - Subsampling

@Suite("Encoder frame subsampling")
struct EncoderSubsamplingTests {
    /// Reference port of HF `ParakeetPreTrainedModel._get_subsampling_output_length`
    /// (`transformers/models/parakeet/modeling_parakeet.py:522-537`): three stride-2,
    /// kernel-3, pad-1 convs. Pins our arithmetic to the model rather than to algebra.
    static func reference(_ melFrames: Int, layers: Int = 3) -> Int {
        var length = melFrames
        for _ in 0..<layers {
            length = (length + 2 * 1 - 3) / 2 + 1
        }
        return length
    }

    @Test("Matches the model's own subsampling formula")
    func matchesReference() {
        for mel in 1...5_000 {
            #expect(
                encoderFrameCount(melFrames: mel, subsamplingFactor: 8) == Self.reference(mel),
                "mel=\(mel)")
        }
    }

    @Test("Pinned to the shipped bundle: 2101 mel frames produce 263 encoder frames")
    func pinnedToShippedBundle() {
        #expect(encoderFrameCount(melFrames: 2101, subsamplingFactor: 8) == 263)
        // The intermediate stages, so a regression says which conv drifted.
        #expect(encoderFrameCount(melFrames: 2101, subsamplingFactor: 2) == 1051)
        #expect(encoderFrameCount(melFrames: 2101, subsamplingFactor: 4) == 526)
    }

    @Test("Collapses to ceil(mel / factor)")
    func collapsesToCeil() {
        for mel in 1...500 {
            let expected = (mel + 7) / 8
            #expect(encoderFrameCount(melFrames: mel, subsamplingFactor: 8) == expected, "mel=\(mel)")
        }
    }

    @Test("Degenerate factors pass the length through")
    func degenerateFactors() {
        #expect(encoderFrameCount(melFrames: 100, subsamplingFactor: 1) == 100)
        #expect(encoderFrameCount(melFrames: 0, subsamplingFactor: 8) == 0)
    }
}

// MARK: - Window geometry

@Suite("Streaming window geometry")
struct StreamingFrameMathTests {
    /// The default export geometry, as a bundle would record it. Geometry is no longer a
    /// library preset, so the tests carry their own.
    static let balancedGeometry = StreamingConfig(
        leftContextFrames: 126, chunkFrames: 12, rightContextFrames: 12)

    /// The identity the whole design rests on: a PCM window that is a whole number of
    /// encoder frames gives `8W + 1` mel frames, of which `8W` are real, and the encoder
    /// emits `W + 1` frames of which `W` can be trusted.
    @Test("A frame-aligned window closes in exact integers")
    func gridIdentities() {
        for usable in 1...200 {
            let config = StreamingConfig(
                windowMelFrames: usable * 8 + 1, chunkFrames: 1, rightContextFrames: 0)
            let melConfig = MelConfig.parakeet.withNFrames(config.windowMelFrames)

            #expect(config.windowSampleCount == usable * 1280, "usable=\(usable)")
            #expect(config.usableEncoderFrames == usable, "usable=\(usable)")
            #expect(config.windowEncoderFrames == usable + 1, "usable=\(usable)")

            // Agreement with the mel front end itself, not just with our own formula.
            let dynamicMel = MelConfig.parakeet
            #expect(
                MelSpectrogram.frameCount(
                    forPCMLength: config.windowSampleCount, config: dynamicMel)
                    == config.windowMelFrames, "usable=\(usable)")
            #expect(
                MelSpectrogram.validFrameCount(
                    forPCMLength: config.windowSampleCount, config: melConfig)
                    == config.validWindowMelFrames, "usable=\(usable)")
        }
    }

    @Test("One encoder frame is 1280 samples / 80 ms")
    func frameDuration() {
        let config = StreamingConfig(
            leftContextFrames: 126, chunkFrames: 12, rightContextFrames: 12)
        #expect(config.samplesPerEncoderFrame == 1280)
        #expect(config.subsamplingFactor == 8)
        #expect(abs(config.seconds(frame: 1) - 0.08) < 1e-12)
        // Frame 0 starts at t=0: encoder frame j is centred on mel frame 8j.
        #expect(config.seconds(frame: 0) == 0)
    }

    /// The two geometries the README recommends exporting, pinned so their advertised latency
    /// and window sizes stay true even though the library no longer carries them as presets.
    @Test("The recommended export geometries have the advertised window and latency")
    func recommendedGeometries() {
        let balanced = StreamingConfig(
            leftContextFrames: 126, chunkFrames: 12, rightContextFrames: 12)
        #expect(balanced.chunkFrames == 12)
        #expect(balanced.rightContextFrames == 12)
        #expect(balanced.leftContextFrames == 126)
        #expect(balanced.usableEncoderFrames == 150)
        #expect(balanced.windowMelFrames == 1201)
        #expect(balanced.windowSampleCount == 192_000)
        #expect(abs(balanced.theoreticalLatency - 1.92) < 1e-9)

        let accuracy = StreamingConfig(
            leftContextFrames: 125, chunkFrames: 25, rightContextFrames: 25)
        #expect(accuracy.usableEncoderFrames == 175)
        #expect(accuracy.windowMelFrames == 1401)
        #expect(accuracy.windowSampleCount == 224_000)
        #expect(abs(accuracy.theoreticalLatency - 4.0) < 1e-9)
    }

    @Test("Hop geometry partitions the timeline with no gaps or repeats")
    func consumeRangesPartition() {
        let geometries = [(126, 12, 12), (125, 25, 25), (239, 12, 12), (20, 4, 8), (100, 100, 50)]
        for (left, chunk, right) in geometries {
            let config = StreamingConfig(
                leftContextFrames: left, chunkFrames: chunk, rightContextFrames: right)
            var expectedNext = 0
            for hop in 0..<500 {
                let range = config.consumeRange(hop: hop)
                #expect(range.lowerBound == expectedNext, "\(left)-\(chunk)-\(right) hop=\(hop)")
                #expect(range.count == chunk)
                expectedNext = range.upperBound

                // The chunk must sit inside the window at a non-negative local index, with
                // its right context still on the far side of it.
                let local = config.localConsumeStart(hop: hop)
                #expect(local >= 0, "hop=\(hop)")
                #expect(local + chunk + right <= config.usableEncoderFrames, "hop=\(hop)")

                // Window start is monotonic and frame-aligned.
                #expect(config.windowStartFrame(hop: hop) <= config.windowStartFrame(hop: hop + 1))
                #expect(
                    config.windowStartSample(hop: hop)
                        == config.windowStartFrame(hop: hop) * config.samplesPerEncoderFrame)
            }
        }
    }

    @Test("Ramp-up clamps the window to zero, giving early hops more left context")
    func rampUpClampsToZero() {
        let config = Self.balancedGeometry  // left 126, chunk 12
        // Until hop*chunk exceeds left context the window cannot slide.
        for hop in 0...10 {
            #expect(config.windowStartFrame(hop: hop) == 0, "hop=\(hop)")
            #expect(config.localConsumeStart(hop: hop) == hop * 12)
        }
        // 126 / 12 = 10.5, so hop 11 is the first to slide.
        #expect(config.windowStartFrame(hop: 11) == 11 * 12 - 126)
        #expect(config.localConsumeStart(hop: 11) == 126)
        #expect(config.localConsumeStart(hop: 50) == 126)
    }

    @Test("A hop waits for its chunk plus right context before running")
    func requiredSampleCount() {
        let config = Self.balancedGeometry
        // Matches NeMo's initial-latency gate: chunk + right before the first encode.
        #expect(config.requiredSampleCount(hop: 0) == (12 + 12) * 1280)
        #expect(config.requiredSampleCount(hop: 1) == (24 + 12) * 1280)
        #expect(config.requiredSampleCount(hop: 5) == (72 + 12) * 1280)
    }

    @Test("Validation rejects geometries that would break the timeJump carry")
    func validationRejectsBadGeometry() {
        // left < chunk: a duration overshoot would land at a negative local index next hop.
        #expect(throws: SpeechError.self) {
            try StreamingConfig(leftContextFrames: 4, chunkFrames: 12, rightContextFrames: 12)
                .validate(maxDuration: 4, encoderMelFrames: nil)
        }
        // right < max duration: an overshoot could point past the window's valid frames.
        #expect(throws: SpeechError.self) {
            try StreamingConfig(leftContextFrames: 100, chunkFrames: 12, rightContextFrames: 2)
                .validate(maxDuration: 4, encoderMelFrames: nil)
        }
        #expect(throws: SpeechError.self) {
            try StreamingConfig(leftContextFrames: 100, chunkFrames: 0, rightContextFrames: 12)
                .validate(maxDuration: 4, encoderMelFrames: nil)
        }
        // Negative context. `leftContextFrames` is derived, so a window too small to hold the
        // chunk plus right context makes it negative, which would start the window past the
        // frames it is meant to cover.
        #expect(throws: SpeechError.self) {
            // usable = ceil(80/8) = 10 encoder frames, but chunk + right asks for 13.
            try StreamingConfig(windowMelFrames: 81, chunkFrames: 8, rightContextFrames: 5)
                .validate(maxDuration: 4, encoderMelFrames: 81)
        }
        #expect(throws: SpeechError.self) {
            try StreamingConfig(leftContextFrames: 100, chunkFrames: 12, rightContextFrames: -1)
                .validate(maxDuration: 4, encoderMelFrames: nil)
        }
        // A config built for a different window than the bundle actually shipped.
        #expect(throws: SpeechError.self) {
            try Self.balancedGeometry.validate(maxDuration: 4, encoderMelFrames: 2101)
        }
        // The happy paths.
        #expect(throws: Never.self) {
            try Self.balancedGeometry.validate(maxDuration: 4, encoderMelFrames: 1201)
            try StreamingConfig(leftContextFrames: 125, chunkFrames: 25, rightContextFrames: 25)
                .validate(maxDuration: 4, encoderMelFrames: 1401)
            try Self.balancedGeometry.validate(maxDuration: 4, encoderMelFrames: nil)
        }
    }
}

// MARK: - Endpointing

@Suite("Endpoint detection")
struct EndpointDetectorTests {
    /// The bug this guards: the detector is fed the decoder's duration-weighted silence
    /// count, so a blank with duration 4 contributes 4 frames rather than one step.
    @Test("Silence is measured in frames, not steps")
    func durationWeighted() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 1_000)
        // Three silent chunks of 4 frames each = 12 frames > 10, so it fires on the third.
        #expect(detector.observe(framesAdvanced: 4, silentFrames: 4, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 4, silentFrames: 8, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 4, silentFrames: 12, segmentHasContent: true) == true)
        #expect(detector.framesSinceEmission == 12)
    }

    /// The regression that motivated frame-granular silence: accumulating `chunkFrames` per quiet
    /// hop made any single quiet hop cross a 10-frame threshold, endpointing mid-utterance.
    @Test("A hop that emits late in its chunk is not an endpoint")
    func partialEmissionWithinChunkIsNotSilence() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 1_000)
        // A 12-frame hop that emitted 9 frames in: only 3 frames of trailing silence.
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 3, segmentHasContent: true) == false)
        // Another full chunk of audio, still emitting — silence stays short.
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 2, segmentHasContent: true) == false)
        #expect(detector.segmentFrames == 24)
    }

    @Test("Any emission resets the silence run")
    func emissionResets() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 1_000)
        #expect(detector.observe(framesAdvanced: 8, silentFrames: 8, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 2, silentFrames: 0, segmentHasContent: true) == false)
        #expect(detector.framesSinceEmission == 0)
        #expect(detector.observe(framesAdvanced: 9, silentFrames: 9, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 1, silentFrames: 10, segmentHasContent: true) == true)
    }

    /// A hard cut at the cap lands mid-word: it splits one word's tokens across two
    /// segments, and joining them reinserts a space (`examination` -> `exam ination`).
    /// Past the cap we wait for the first pause instead.
    @Test("The length cap waits for a pause rather than cutting mid-word")
    func lengthCapWaitsForPause() {
        var detector = EndpointDetector(silenceFrames: 1_000, maxSegmentFrames: 20)
        // Well past the cap, but speech is continuous — must not fire.
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 0, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 0, segmentHasContent: true) == false)
        #expect(detector.segmentFrames == 24)
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 0, segmentHasContent: true) == false)
        // The first quiet frame past the cap closes the segment.
        #expect(detector.observe(framesAdvanced: 1, silentFrames: 1, segmentHasContent: true) == true)
    }

    @Test("Under the cap, a single quiet frame is not an endpoint")
    func underCapIgnoresBriefPause() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 1_000)
        #expect(detector.observe(framesAdvanced: 1, silentFrames: 1, segmentHasContent: true) == false)
        #expect(detector.observe(framesAdvanced: 1, silentFrames: 0, segmentHasContent: true) == false)
    }

    /// The bug: a long pause consumes frames while the next segment is still empty, spending its
    /// length budget before it has any audio — an 8 s pause cut the next segment ~7 s early.
    @Test("A pause does not spend the next segment's length budget")
    func silenceDoesNotChargeTheEmptySegment() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 20)
        // Speech, then an endpoint — the streaming path resets here.
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 12, segmentHasContent: true) == true)
        detector.reset()
        // A long pause: hops keep consuming frames with nothing in the open segment.
        for _ in 0..<10 {
            #expect(
                detector.observe(framesAdvanced: 12, silentFrames: 132, segmentHasContent: false)
                    == true)
        }
        #expect(detector.segmentFrames == 0)
        // Speech resumes with the cap's full budget: one chunk in, it is nowhere near it.
        #expect(detector.observe(framesAdvanced: 12, silentFrames: 0, segmentHasContent: true) == false)
        #expect(detector.segmentFrames == 12)
    }

    @Test("Reset clears both counters")
    func resetClears() {
        var detector = EndpointDetector(silenceFrames: 10, maxSegmentFrames: 20)
        _ = detector.observe(framesAdvanced: 5, silentFrames: 5, segmentHasContent: true)
        detector.reset()
        #expect(detector.framesSinceEmission == 0)
        #expect(detector.segmentFrames == 0)
    }

    /// The predictor reset has to sit well above the endpoint threshold: firing it at every
    /// segment boundary is the configuration measured to drop ~3.8 s of audio to a reset.
    @Test("A predictor reset at or below the endpoint threshold is rejected")
    func resetThresholdMustExceedSilence() throws {
        #expect(throws: SpeechError.self) {
            try EndpointingConfig(silenceFrames: 10, resetAfterSilenceFrames: 10)
                .validate(chunkFrames: 12)
        }
        #expect(throws: SpeechError.self) {
            try EndpointingConfig(silenceFrames: 10, resetAfterSilenceFrames: 5)
                .validate(chunkFrames: 12)
        }
        // 0 disables it, matching NeMo's unbroken state carry, and must stay legal.
        try EndpointingConfig(silenceFrames: 10, resetAfterSilenceFrames: 0)
            .validate(chunkFrames: 12)
        try EndpointingConfig(silenceFrames: 10, resetAfterSilenceFrames: 40)
            .validate(chunkFrames: 12)
    }

    /// The shipped default, pinned: 40 frames is 3.2 s at the 80 ms frame every Parakeet
    /// bundle has, chosen from a safe band of 30–150 (see `resetAfterSilenceFrames`).
    @Test("The default endpointing config is internally consistent")
    func defaultsValidate() throws {
        let config = EndpointingConfig()
        #expect(config.silenceFrames == 10)
        #expect(config.maxSegmentFrames == 375)
        #expect(config.resetAfterSilenceFrames == 40)
        try config.validate(chunkFrames: 25)
    }

    /// Zero silence frames would fire an endpoint on every hop, and a cap below one chunk
    /// could never be reached at a pause — the segment would be cut on the first chunk instead.
    @Test("Endpointing rejects a threshold that could never behave as intended")
    func endpointingRejectsUnreachableThresholds() {
        #expect(throws: SpeechError.self) {
            try EndpointingConfig(silenceFrames: 0).validate(chunkFrames: 25)
        }
        #expect(throws: SpeechError.self) {
            try EndpointingConfig(silenceFrames: -1).validate(chunkFrames: 25)
        }
        #expect(throws: SpeechError.self) {
            try EndpointingConfig(silenceFrames: 10, maxSegmentFrames: 24)
                .validate(chunkFrames: 25)
        }
        // A cap of exactly one chunk is the boundary, and is allowed.
        #expect(throws: Never.self) {
            try EndpointingConfig(silenceFrames: 10, maxSegmentFrames: 25)
                .validate(chunkFrames: 25)
        }
    }
}

@Suite("Streaming metadata")
struct StreamingMetadataTests {
    @Test("Decodes the block a --streaming export writes")
    func decodesStreamingBlock() throws {
        let json = """
            {"metadata_version":"0.2","kind":"speech_recognizer","streaming":{
              "left_context_encoder_frames":126,"chunk_encoder_frames":12,
              "right_context_encoder_frames":12,"usable_encoder_frames":150,
              "window_encoder_frames":151,"window_mel_frames":1201,
              "valid_window_mel_frames":1200,"window_sample_count":192000,
              "samples_per_encoder_frame":1280,"seconds_per_encoder_frame":0.08,
              "sample_rate":16000,"hop_length":160,"subsampling_factor":8}}
            """
        let config = try #require(try StreamingConfig.decode(fromMetadata: Data(json.utf8)))
        #expect(config.windowMelFrames == 1201)
        #expect(config.chunkFrames == 12)
        #expect(config.rightContextFrames == 12)
        #expect(config.leftContextFrames == 126)
        #expect(config.usableEncoderFrames == 150)
        #expect(config.windowSampleCount == 192_000)
        try config.validate(maxDuration: 4, encoderMelFrames: 1201)
    }

    /// The block a current export writes carries only what the runtime reads. Older bundles
    /// carry six extra derived keys, which `Decodable` ignores — covered above.
    @Test("A block of only the recorded keys decodes")
    func recordedKeysOnly() throws {
        let json = """
            {"metadata_version":"0.2","kind":"speech_recognizer","streaming":{
              "left_context_encoder_frames":126,"chunk_encoder_frames":12,
              "right_context_encoder_frames":12,"window_mel_frames":1201,
              "sample_rate":16000,"hop_length":160,"subsampling_factor":8}}
            """
        let config = try #require(try StreamingConfig.decode(fromMetadata: Data(json.utf8)))
        #expect(config.leftContextFrames == 126)
        #expect(config.chunkFrames == 12)
        #expect(config.windowMelFrames == 1201)
    }

    /// Left is derived, so the recorded copy is only a cross-check — and it has to bite, or a
    /// hand-edited chunk would leave the file describing a geometry the runtime never runs.
    @Test("A block whose left context disagrees with its window is rejected")
    func inconsistentLeftIsRejected() {
        // chunk raised to 25 without fixing left: 150 - 25 - 12 is 113, not 126.
        let json = """
            {"metadata_version":"0.2","kind":"speech_recognizer","streaming":{
              "left_context_encoder_frames":126,"chunk_encoder_frames":25,
              "right_context_encoder_frames":12,"window_mel_frames":1201,
              "sample_rate":16000,"hop_length":160,"subsampling_factor":8}}
            """
        #expect(throws: SpeechError.self) {
            try StreamingConfig.decode(fromMetadata: Data(json.utf8))
        }
    }

    @Test("Bundles without the block decode to nil rather than failing")
    func absentBlockIsNil() throws {
        // Every bundle exported before streaming existed looks like this. Such a bundle simply
        // cannot stream; `startStream` reports that rather than inventing a geometry.
        let json = #"{"metadata_version":"0.2","kind":"speech_recognizer","config":{}}"#
        #expect(try StreamingConfig.decode(fromMetadata: Data(json.utf8)) == nil)
        #expect(try StreamingConfig.decode(fromMetadata: Data("not json".utf8)) == nil)
    }

    /// `encoderFrameCount` halves, so a factor of 6 would silently act as 4 — one frame is
    /// 80 ms, enough to drop or duplicate words at every boundary. Caught before any frame
    /// count is derived from it.
    @Test("A subsampling factor the frame arithmetic cannot express is rejected")
    func nonPowerOfTwoSubsamplingIsRejected() {
        for factor in [0, 6, 7, 12] {
            let json = """
                {"metadata_version":"0.2","kind":"speech_recognizer","streaming":{
                  "left_context_encoder_frames":126,"chunk_encoder_frames":12,
                  "right_context_encoder_frames":12,"window_mel_frames":1201,
                  "sample_rate":16000,"hop_length":160,"subsampling_factor":\(factor)}}
                """
            #expect(throws: SpeechError.self, "factor=\(factor)") {
                try StreamingConfig.decode(fromMetadata: Data(json.utf8))
            }
        }
    }

    @Test("A zero hop length is rejected rather than dividing by zero")
    func zeroHopLengthIsRejected() {
        let json = """
            {"metadata_version":"0.2","kind":"speech_recognizer","streaming":{
              "left_context_encoder_frames":126,"chunk_encoder_frames":12,
              "right_context_encoder_frames":12,"window_mel_frames":1201,
              "sample_rate":16000,"hop_length":0,"subsampling_factor":8}}
            """
        #expect(throws: SpeechError.self) {
            try StreamingConfig.decode(fromMetadata: Data(json.utf8))
        }
    }
}
