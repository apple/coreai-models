// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

/// Reference-token subsampling and CFG interpolation for FLUX.2.
///
/// Neither had coverage, and both are the kind of index-and-sign arithmetic whose
/// failure mode is a subtly degraded image rather than a crash, so these pin the
/// documented properties instead of golden values.
@Suite("Flux2 token math")
struct Flux2TokenMathTests {
    /// Deterministic filler — a fixed pattern keeps failures reproducible.
    private func ramp(_ count: Int) -> [Float] {
        (0..<count).map { Float(($0 * 37) % 211) / 211.0 - 0.5 }
    }

    @Test("subsampleTokens at stride 1 is an identity copy")
    func subsampleStrideOneIsIdentity() {
        let tokens = ramp(8 * 8 * 4)
        let actual = Flux2Pipeline.subsampleTokens(
            tokens, fromSide: 8, toSide: 8, channels: 4)
        #expect(actual == tokens)
    }

    /// Averaging is per channel because the channel index encodes intra-patch position;
    /// mixing channels would misalign sub-positions. A per-channel constant must survive.
    @Test("subsampleTokens preserves per-channel constants")
    func subsamplePreservesChannelConstants() {
        let channels = 4
        let fromSide = 8
        var tokens = [Float](repeating: 0, count: fromSide * fromSide * channels)
        for token in 0..<(fromSide * fromSide) {
            for c in 0..<channels { tokens[token * channels + c] = Float(c) * 10 }
        }
        let actual = Flux2Pipeline.subsampleTokens(
            tokens, fromSide: fromSide, toSide: 2, channels: channels)
        for token in 0..<4 {
            for c in 0..<channels {
                #expect(abs(actual[token * channels + c] - Float(c) * 10) < 1e-5)
            }
        }
    }

    /// Block means, not point samples: subsampling a grid whose every 2×2 block averages
    /// to a known value must reproduce that value.
    @Test("subsampleTokens area-averages rather than point-sampling")
    func subsampleAreaAverages() {
        // 4×4 grid, 1 channel, value = token index. Each 2×2 block of a 4-wide row-major
        // grid holds {r, r+1, r+4, r+5}, so its mean is r + 2.5.
        let tokens = (0..<16).map { Float($0) }
        let actual = Flux2Pipeline.subsampleTokens(
            tokens, fromSide: 4, toSide: 2, channels: 1)
        #expect(actual.count == 4)
        for (i, origin) in [0, 2, 8, 10].enumerated() {
            #expect(abs(actual[i] - (Float(origin) + 2.5)) < 1e-5, "block \(i)")
        }
    }

    @Test(
        "CFG interpolation matches uncond + g*(cond - uncond)",
        arguments: [Float(0.0), 1.0, 1.5, 3.5, 7.0, 30.0])
    func cfgMatchesFormula(guidanceScale: Float) {
        let count = 512
        let cond = ramp(count)
        let uncond = ramp(count).map { $0 * -0.5 + 0.25 }
        var destination = [Float](repeating: 0, count: count)

        Flux2Pipeline.applyClassifierFreeGuidance(
            cond: cond[0..<count], uncond: uncond[0..<count],
            guidanceScale: guidanceScale, into: &destination)

        for i in 0..<count {
            // Same expression in the same order, so this is exact, not approximate.
            #expect(destination[i] == uncond[i] + guidanceScale * (cond[i] - uncond[i]))
        }
    }

    @Test("CFG at guidance 1 returns the conditional pass unchanged")
    func cfgAtUnitGuidanceIsConditional() {
        let cond = ramp(64)
        let uncond = ramp(64).map { -$0 }
        var destination = [Float](repeating: 0, count: 64)
        Flux2Pipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 1.0,
            into: &destination)
        #expect(destination == cond)
    }

    /// The buffer is reused across denoising steps, so a stale step must not bleed through.
    @Test("CFG buffer reuse does not carry state across steps")
    func cfgBufferReuseIsClean() {
        let cond = ramp(64)
        let uncond = ramp(64).map { -$0 }
        var destination = [Float](repeating: 0, count: 64)

        Flux2Pipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 25.0,
            into: &destination)
        Flux2Pipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 2.0,
            into: &destination)

        for i in 0..<64 {
            #expect(destination[i] == uncond[i] + 2.0 * (cond[i] - uncond[i]))
        }
    }

    /// Slices handed in by the pipeline are prefixes of a longer transformer output;
    /// only the noise-token span may be read.
    @Test("CFG reads only the requested slice")
    func cfgHonorsSliceBounds() {
        let full = ramp(256)
        let half = 128
        var destination = [Float](repeating: 0, count: half)
        Flux2Pipeline.applyClassifierFreeGuidance(
            cond: full[0..<half], uncond: full[0..<half], guidanceScale: 4.0,
            into: &destination)
        // cond == uncond collapses to uncond regardless of g, so any leakage past the
        // slice bound would show up as a mismatch here.
        #expect(destination == Array(full[0..<half]))
    }
}
