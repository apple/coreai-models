// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation
import FoundationModels
import Testing

@Suite("CoreAILanguageModel public interface")
struct PublicInterfaceTests {
    /// Compile-time check that the documented public usage chain still
    /// resolves. Runtime throws on the first step (asset does not exist);
    /// subsequent lines exist purely for type-checking. If any public
    /// symbol or signature drifts, this target fails to build.
    ///
    /// Mirrors the snippet in README.md:
    /// ```swift
    /// let model = try await CoreAILanguageModel(resourcesAt: url)
    /// let session = LanguageModelSession(model: model)
    /// let response = try await session.respond(to: "…")
    /// print(response.content)
    /// ```
    @Test("Documented public usage compiles (runtime throws on missing asset)")
    func documentedUsageCompiles() async {
        let missing = URL(fileURLWithPath: "/nonexistent/model")
        await #expect(throws: (any Error).self) {
            let model = try await CoreAILanguageModel(resourcesAt: missing)
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: "test")
            _ = response.content
        }
    }

    /// Compile-time check that the resource-control surface
    @Test("Resource-control API compiles (runtime throws on missing asset)")
    func resourceControlAPICompiles() async {
        let missing = URL(fileURLWithPath: "/nonexistent/model")
        await #expect(throws: (any Error).self) {
            let model = try await CoreAILanguageModel(resourcesAt: missing, mode: .lazy)
            _ = model.estimatedSizeOnDiskBytes
            try await model.load()
            model.unload()
        }
    }
}
