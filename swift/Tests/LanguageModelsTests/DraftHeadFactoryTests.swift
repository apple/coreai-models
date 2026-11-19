// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels
@testable import CoreAIShared

@Suite("Draft head factory")
struct DraftHeadFactoryTests {
    private static func bundle(withDrafter: Bool, kind: String = "mtp") throws -> LanguageModelBundle {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "DraftHeadFactoryTests-\(UUID().uuidString)/gemma")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let drafter =
            withDrafter
            ? """
            , "drafter": "draft_step.aimodel" },
              "auxiliary_assets": { "drafter_config": "draft_step.json", "drafter_scan": "draft_scan.bin",
                "drafter_scan_index": "draft_scan.json" },
              "speculative": { "drafter_kind": "\(kind)" }
            """
            : " }"
        let metadata =
            """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "gemma",
              "assets": { "main": "model.aimodel"\(drafter),
              "language": { "tokenizer": "x/y", "vocab_size": 100, "max_context_length": 512 }
            }
            """
        try metadata.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        return try LanguageModelBundle(at: dir)
    }

    private func message(_ body: () throws -> Void) -> String {
        do {
            try body()
            return ""
        } catch {
            return "\(error)"
        }
    }

    @Test("A bundle with no draft head is rejected, with how to add one")
    func noDraftHead() throws {
        let bundle = try Self.bundle(withDrafter: false)
        let text = message { _ = try EngineFactory.requireDraftHead(bundle: bundle, targetEmitsHiddenStates: true) }
        #expect(text.contains("no draft head") && text.contains("export_drafter.py --bundle"))
    }

    @Test("A target whose decode graphs emit no hidden states is rejected, with how to re-export")
    func noHiddenStates() throws {
        let bundle = try Self.bundle(withDrafter: true)
        let text = message { _ = try EngineFactory.requireDraftHead(bundle: bundle, targetEmitsHiddenStates: false) }
        #expect(text.contains("hidden_states") && text.contains("--emit-hidden-states"))
    }

    @Test("A drafter of another kind is rejected before the target is asked about hidden states")
    func otherKind() throws {
        let bundle = try Self.bundle(withDrafter: true, kind: "block")
        let text = message { _ = try EngineFactory.requireDraftHead(bundle: bundle, targetEmitsHiddenStates: false) }
        #expect(text.contains("mtp"))
    }

    @Test("A bundle with a draft head and a target that emits hidden states yields the head's files")
    func accepted() throws {
        let bundle = try Self.bundle(withDrafter: true)
        let component = try EngineFactory.requireDraftHead(bundle: bundle, targetEmitsHiddenStates: true)
        #expect(component == DrafterComponent(directory: bundle.bundlePath))
    }

    @Test("The draft count defaults to one less than the verify tile, else the general default")
    func defaultDraftCount() {
        let options = EngineFactory.DraftHeadOptions()
        #expect(options.resolvedDraftTokens(verifyTileWidth: 8) == 7)
        #expect(options.resolvedDraftTokens(verifyTileWidth: 4) == 3)
        #expect(options.resolvedDraftTokens(verifyTileWidth: nil) == SpeculativeDefaults.numDraftTokens)
        #expect(options.resolvedDraftTokens(verifyTileWidth: 1) == SpeculativeDefaults.numDraftTokens)
        let explicit = EngineFactory.DraftHeadOptions(draftTokens: 5)
        #expect(explicit.resolvedDraftTokens(verifyTileWidth: 8) == 5)
    }
}
