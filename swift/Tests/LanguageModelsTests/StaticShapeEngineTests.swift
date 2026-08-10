// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Context-bucket parsing

@Suite("StaticShapeEngine context-bucket parsing")
struct StaticContextParseTests {
    @Test("parses ctx from extend/prompt function names")
    func parsesCtx() {
        #expect(StaticShapeEngine.parseContextLength(functionName: "extend_256_16") == 256)
        #expect(StaticShapeEngine.parseContextLength(functionName: "extend_1024_8") == 1024)
        #expect(StaticShapeEngine.parseContextLength(functionName: "extend_32768_8") == 32768)
        #expect(StaticShapeEngine.parseContextLength(functionName: "prompt_opt_1024_64") == 1024)
        #expect(StaticShapeEngine.parseContextLength(functionName: "prompt_opt_512_8") == 512)
    }

    @Test("ctx is the second-to-last component, not shape.max()")
    func ctxNotShapeMax() {
        // Regression: qwen3 key_cache is [.,1024,.,256] at the 256 bucket, so `shape.max()`
        // would wrongly report 1024. The name is authoritative.
        #expect(StaticShapeEngine.parseContextLength(functionName: "extend_256_64") == 256)
    }

    @Test("returns nil for names without a numeric ctx component")
    func nilOnMalformed() {
        #expect(StaticShapeEngine.parseContextLength(functionName: "load_embeddings") == nil)
        #expect(StaticShapeEngine.parseContextLength(functionName: "gather") == nil)
    }
}

// MARK: - Causal mask fill

@Suite("Static causal mask fill")
struct StaticMaskFillTests {
    /// Build a rank-4 (1, ctx, 1, q) mask, fill it, and read it back as [ctx][q].
    private func filledMask(ctx: Int, q: Int, tokensInBatch: Int, alignedStep: Int) -> [[Float]] {
        var mask = NDArray(shape: [1, ctx, 1, q], scalarType: .float16)
        let view = mask.mutableView(as: LogitsScalarType.self)
        StaticMaskFill.fill(view, tokensInBatch: tokensInBatch, alignedStep: alignedStep)
        let flat = readNDArray(mask, as: LogitsScalarType.self, count: ctx * q)
        var out = [[Float]](repeating: [Float](repeating: 0, count: q), count: ctx)
        for c in 0..<ctx {
            for query in 0..<q {
                out[c][query] = Float(flat[c * q + query])
            }
        }
        return out
    }

    @Test("prefill: query i attends to keys 0...i, masks the rest")
    func prefillCausal() {
        // ctx=4, q=4, all 4 tokens new, block starts at position 0.
        let m = filledMask(ctx: 4, q: 4, tokensInBatch: 4, alignedStep: 0)
        for query in 0..<4 {
            for key in 0..<4 {
                if key <= query {
                    #expect(m[key][query] == 0, "query \(query) should see key \(key)")
                } else {
                    #expect(m[key][query] < -1000, "query \(query) should mask key \(key)")
                }
            }
        }
    }

    @Test("decode: absolute positions via alignedStep")
    func decodeAligned() {
        // A single query at absolute position 5 (alignedStep=5) should see keys 0...5.
        let m = filledMask(ctx: 8, q: 1, tokensInBatch: 1, alignedStep: 5)
        for key in 0..<8 {
            if key <= 5 {
                #expect(m[key][0] == 0)
            } else {
                #expect(m[key][0] < -1000)
            }
        }
    }
}
