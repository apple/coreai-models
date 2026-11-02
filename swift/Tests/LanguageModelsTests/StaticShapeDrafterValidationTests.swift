// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The pure checks `StaticShapeDrafter.load` makes before it trusts an exported drafter.
@Suite("Draft head load validation")
struct StaticShapeDrafterValidationTests {
    private func config(
        _ json: String = "{\"full_windows\": [512, 1024]}"
    ) throws -> StaticShapeDrafter.Config {
        let separator = json == "{}" ? "" : ","
        let full =
            json.dropLast()
            + separator + " \"sliding_theta\": 10000, \"full_theta\": 1000000, \"full_partial_rotary_factor\": 0.25}"
        return try JSONDecoder().decode(StaticShapeDrafter.Config.self, from: Data(full.utf8))
    }

    private func message(_ body: () throws -> Void) -> String {
        do {
            try body()
            return ""
        } catch {
            return "\(error)"
        }
    }

    private let functions: Set<String> = ["draft_step_512", "draft_step_1024"]

    private func contract(_ json: String) throws -> StaticShapeDrafter.Config {
        try config("{\"contract\": \(json), \"full_windows\": [512, 1024]}")
    }

    @Test("The contract this runtime provides is the one the exporter writes")
    func exporterContract() throws {
        let written = try contract(
            """
            {"hidden_states_output": "hidden_states", "token_embedding": "unscaled",
             "kv_source": "last_storing_layer"}
            """)
        #expect(written.contract == StaticShapeDrafter.Contract.supported)
        try StaticShapeDrafter.checkContract(written.contract, name: "draft_step.json")
    }

    @Test("An export from before the contract existed is accepted")
    func noContract() throws {
        #expect(try config().contract == nil)
        try StaticShapeDrafter.checkContract(try config().contract, name: "draft_step.json")
    }

    @Test(
        "An expectation the runtime does not implement is rejected, naming it and the remedy",
        arguments: [
            ("hidden_states_output", "final_hidden", "hidden-state output"),
            ("token_embedding", "scaled", "token embedding"),
            ("kv_source", "first_storing_layer", "shared K/V"),
        ])
    func unsupportedExpectation(field: String, value: String, mentions: String) throws {
        let c = try contract("{\"\(field)\": \"\(value)\"}")
        let text = message { try StaticShapeDrafter.checkContract(c.contract, name: "draft_step.json") }
        #expect(text.contains(mentions) && text.contains(value) && text.contains("re-export"))
    }

    @Test("An expectation the export leaves out is not checked")
    func partialContract() throws {
        let c = try contract("{\"kv_source\": \"last_storing_layer\"}")
        try StaticShapeDrafter.checkContract(c.contract, name: "draft_step.json")
    }

    @Test("Windows resolve to the listed full_windows of a current export")
    func currentExport() throws {
        let r = try StaticShapeDrafter.resolveWindows(
            config: try config(), name: "draft_step.json", functionNames: functions)
        #expect(r.windows == [512, 1024])
        #expect(!r.legacy)
    }

    @Test("An export from before per-window draft steps resolves to its one window")
    func legacyExport() throws {
        let r = try StaticShapeDrafter.resolveWindows(
            config: try config("{\"window\": 512}"), name: "draft_step.json", functionNames: ["draft_step"])
        #expect(r.windows == [512])
        #expect(r.legacy)
    }

    @Test("A config with neither full_windows nor window is rejected")
    func noWindows() throws {
        let text = message {
            _ = try StaticShapeDrafter.resolveWindows(
                config: try config("{}"), name: "draft_step.json", functionNames: functions)
        }
        #expect(text.contains("neither full_windows nor window"))
    }

    @Test("Windows that are unsorted, duplicated or not positive are rejected")
    func badWindowOrder() throws {
        for list in ["[1024, 512]", "[512, 512]", "[0, 512]", "[-5]", "[]"] {
            let text = message {
                _ = try StaticShapeDrafter.resolveWindows(
                    config: try config("{\"full_windows\": \(list)}"), name: "draft_step.json",
                    functionNames: functions.union(["draft_step_0", "draft_step_-5"]))
            }
            #expect(!text.isEmpty, "\(list)")
        }
    }

    @Test("Every listed window needs its function in the program, not only the first")
    func missingWindowFunction() throws {
        let text = message {
            _ = try StaticShapeDrafter.resolveWindows(
                config: try config("{\"full_windows\": [512, 99999]}"), name: "draft_step.json",
                functionNames: ["draft_step_512"])
        }
        #expect(text.contains("draft_step_99999"))
    }

    @Test("A sidecar version newer than the reader's is rejected, an absent one accepted")
    func sidecarVersion() throws {
        #expect(try config().version == nil)
        let newer = try config("{\"version\": 2, \"full_windows\": [512]}")
        let text = message {
            _ = try StaticShapeDrafter.resolveWindows(config: newer, name: "draft_step.json", functionNames: functions)
        }
        #expect(text.contains("version 2"))
        let current = try config("{\"version\": 1, \"full_windows\": [512, 1024]}")
        _ = try StaticShapeDrafter.resolveWindows(config: current, name: "draft_step.json", functionNames: functions)
    }

    private func scanIndex(vocab: Int = 262_144, centroids: Int = 2048) -> DraftPack.Index {
        let json =
            "{\"vocab\": \(vocab), \"centroids\": \(centroids), \"per_centroid\": 128, \"layers\": 4, \"tensors\": {}}"
        return try! JSONDecoder().decode(DraftPack.Index.self, from: Data(json.utf8))
    }

    private func check(
        backbone: Int = 1536, vocab: Int? = 262_144, embeds: Int? = 3072, hidden: Int? = 1536,
        logits: Int? = 2048, x: Int? = 256, scanWidth: Int = 256, centroids: Int = 2048
    ) -> String {
        message {
            try StaticShapeDrafter.checkGeometry(
                scan: scanIndex(centroids: centroids), scanHiddenWidth: scanWidth, backboneHidden: backbone,
                vocabSize: vocab, inputsEmbedsWidth: embeds, hiddenWidth: hidden, centroidLogitsWidth: logits,
                stateWidth: x)
        }
    }

    @Test("A drafter that matches the target passes")
    func matching() {
        #expect(check() == "")
    }

    @Test("A drafter exported for a different backbone width is rejected")
    func backboneMismatch() {
        // An e2b drafter (backbone 1536) against an e4b target (2560).
        let text = check(backbone: 2560)
        #expect(text.contains("1536"))
        #expect(text.contains("2560"))
        #expect(check(backbone: 2560, embeds: 3072, hidden: 2560) != "")
    }

    @Test("A drafter or scan table for a different vocabulary is rejected")
    func vocabularyMismatch() {
        #expect(check(vocab: 151_936).contains("151936"))
        #expect(check(vocab: nil) == "")
    }

    @Test("A scan index whose centroid count disagrees with the program's centroid logits is rejected")
    func centroidMismatch() {
        #expect(check(logits: 2048, centroids: 2_048_000).contains("2048000"))
    }

    @Test("A scan table narrower or wider than the program's state is rejected")
    func scanWidthMismatch() {
        #expect(check(x: 256, scanWidth: 128).contains("128"))
    }

    @Test("A program with no readable dimension is rejected rather than assumed to match")
    func unreadableDimension() {
        #expect(check(embeds: nil) != "")
    }

    @Test("The legacy-export and coverage warnings are produced for the always-visible path")
    func loadWarnings() {
        let legacy = StaticShapeDrafter.loadWarnings(
            legacy: true, windows: [512], configName: "draft_step.json", maxContextLength: 8192)
        #expect(legacy.count == 2)
        #expect(legacy[0].contains("predates per-window draft steps"))
        #expect(legacy[1].contains("512 of the target's 8192"))
        let current = StaticShapeDrafter.loadWarnings(
            legacy: false, windows: [512, 8192], configName: "draft_step.json", maxContextLength: 8192)
        #expect(current.isEmpty)
        let short = StaticShapeDrafter.loadWarnings(
            legacy: false, windows: [512, 4096], configName: "draft_step.json", maxContextLength: 8192)
        #expect(short.count == 1)
        #expect(short[0].contains("--max-context-length 8192"))
    }
}

/// Loads an exported drafter: the directory named by `COREAI_TEST_DRAFTER_E2B`, an e2b export
/// (backbone width 1536). Skipped without it.
@Suite(
    "Draft head load against an exported program", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_DRAFTER_E2B"] != nil))
struct StaticShapeDrafterExportTests {
    private var component: DrafterComponent {
        DrafterComponent(
            directory: URL(fileURLWithPath: ProcessInfo.processInfo.environment["COREAI_TEST_DRAFTER_E2B"] ?? ""))
    }

    @Test("A drafter loads for the target it was exported for")
    func matchingTarget() async throws {
        _ = try await StaticShapeDrafter.load(
            component: component, backboneHiddenSize: 1536, maxContextLength: 16384, vocabSize: 262_144)
    }

    @Test("An e2b drafter is rejected for a wider backbone, with both widths in the message")
    func widerTarget() async {
        do {
            _ = try await StaticShapeDrafter.load(
                component: component, backboneHiddenSize: 2560, maxContextLength: 16384)
            Issue.record("expected a mismatch error")
        } catch {
            #expect("\(error)".contains("1536") && "\(error)".contains("2560"))
        }
    }

    @Test("A drafter that is not connected to a target refuses to draft")
    func unconnectedDrafter() async throws {
        let drafter = try await StaticShapeDrafter.load(
            component: component, backboneHiddenSize: 1536, maxContextLength: 16384, vocabSize: 262_144)
        var rng = SystemRandomNumberGenerator()
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await drafter.draftAll(
                confirmedPrefix: [1, 2, 3], count: 2, policy: .greedy, using: &rng)
        }
    }

    @Test("A different vocabulary is rejected")
    func otherVocabulary() async {
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await StaticShapeDrafter.load(
                component: component, backboneHiddenSize: 1536, maxContextLength: 16384, vocabSize: 151_936)
        }
    }
}
