// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("InputLayout Tests")
struct InputLayoutTests {
    let standardInputs = ["input_ids", "position_ids"]
    let standardOutputs = ["logits"]

    @Test("Resolves alternate names and throws on unknown")
    func nameResolution() throws {
        let alt = try InputLayout.resolveRequired(
            from: ["in_new_token_ids", "pos_ids"],
            candidates: InputLayout.knownInputIdNames,
            label: "input_ids")
        #expect(alt == "in_new_token_ids")

        #expect(throws: InferenceRuntimeError.self) {
            try InputLayout.resolveRequired(
                from: ["tokens", "positions"],
                candidates: InputLayout.knownInputIdNames,
                label: "input_ids")
        }
    }

    @Test("Dynamic model: standard policies and names")
    func dynamicDefaults() throws {
        let layout = try makeLayout()
        #expect(layout.queryPolicy == .dynamic)
        #expect(layout.logitsPolicy == .dynamic)
        #expect(layout.positionPolicy == .full)
        #expect(layout.prefillPolicy == .chunk(threshold: 1024, chunkSize: 512))
        #expect(layout.inputIdsName == "input_ids")
        #expect(layout.positionIdsName == "position_ids")
    }

    @Test("Static S=1: fixed policies, prefill chunks at 1 (#212)")
    func staticS1() throws {
        let layout = try makeLayout(inputIdsSeqLen: 1, logitsSeqLen: 1)
        #expect(layout.queryPolicy == .fixed(1))
        #expect(layout.logitsPolicy == .fixed(seqLen: 1))
        #expect(layout.prefillPolicy == .chunk(threshold: 1, chunkSize: 1))
    }

    @Test("Prefill graph overrides chunk policy, even for S=1")
    func prefillGraphOverrides() throws {
        let dynamic = try makeLayout(hasPrefillGraph: true)
        #expect(dynamic.prefillPolicy == .prefillGraph)

        let s1 = try makeLayout(inputIdsSeqLen: 1, logitsSeqLen: 1, hasPrefillGraph: true)
        #expect(s1.prefillPolicy == .prefillGraph)
    }

    @Test("Alternate input names resolve correctly")
    func alternateNames() throws {
        let layout = try InputLayout.analyze(
            inputNames: ["in_new_token_ids", "pos_ids"],
            outputNames: ["out_logits"],
            inputIdsSeqLen: nil, logitsSeqLen: nil,
            chunkThreshold: 1024, chunkSize: 512,
            hasPrefillGraph: false, useCompactPositionIds: true)
        #expect(layout.inputIdsName == "in_new_token_ids")
        #expect(layout.positionIdsName == "pos_ids")
        #expect(layout.positionPolicy == .compact)
    }

    @Test("Throws on empty outputs or missing position_ids")
    func throwsOnBadDescriptor() {
        #expect(throws: InferenceRuntimeError.self) {
            try makeLayout(outputNames: [])
        }
        #expect(throws: InferenceRuntimeError.self) {
            try InputLayout.analyze(
                inputNames: ["input_ids"], outputNames: ["logits"],
                inputIdsSeqLen: nil, logitsSeqLen: nil,
                chunkThreshold: 1024, chunkSize: 512,
                hasPrefillGraph: false, useCompactPositionIds: false)
        }
    }

    @Test("inputIdsSeqLen=0 treated as dynamic")
    func zeroSeqLenIsDynamic() throws {
        let layout = try makeLayout(inputIdsSeqLen: 0)
        #expect(layout.queryPolicy == .dynamic)
    }

    @Test("Asymmetric: static input with dynamic logits still chunks at 1")
    func asymmetricStaticInput() throws {
        let layout = try makeLayout(inputIdsSeqLen: 1, logitsSeqLen: nil)
        #expect(layout.queryPolicy == .fixed(1))
        #expect(layout.logitsPolicy == .dynamic)
        #expect(layout.prefillPolicy == .chunk(threshold: 1, chunkSize: 1))
    }

    // MARK: - Helpers

    private func makeLayout(
        inputIdsSeqLen: Int? = nil,
        logitsSeqLen: Int? = nil,
        chunkThreshold: Int = 1024,
        chunkSize: Int = 512,
        hasPrefillGraph: Bool = false,
        useCompactPositionIds: Bool = false,
        outputNames: [String]? = nil
    ) throws -> InputLayout {
        try InputLayout.analyze(
            inputNames: standardInputs,
            outputNames: outputNames ?? standardOutputs,
            inputIdsSeqLen: inputIdsSeqLen,
            logitsSeqLen: logitsSeqLen,
            chunkThreshold: chunkThreshold,
            chunkSize: chunkSize,
            hasPrefillGraph: hasPrefillGraph,
            useCompactPositionIds: useCompactPositionIds)
    }
}
