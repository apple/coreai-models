// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// Tests for EngineOptions.
@Suite("EngineOptions")
struct EngineOptionsTests {
    @Test("EngineOptions initialization")
    func engineOptionsInit() {
        // Defaults
        let defaults = EngineOptions()
        #expect(defaults.variant == nil)
        #expect(defaults.kvCacheStrategy == .auto)

        // Custom variant
        let custom = EngineOptions(variant: "pipelined")
        #expect(custom.variant == "pipelined")

        // Custom KV cache strategy
        let withStrategy = EngineOptions(variant: "sequential", kvCacheStrategy: .growing)
        #expect(withStrategy.variant == "sequential")
        #expect(withStrategy.kvCacheStrategy == .growing)
    }

    @Test("Merging a bundle's tensor data keeps every other option, and an explicit entry wins")
    func mergingTensorDataKeepsEveryOption() {
        let explicit = URL(fileURLWithPath: "/explicit.bin")
        let options = EngineOptions(
            variant: "coreai-sequential", kvCacheStrategy: .growing, kvCacheSize: 4096, prefillChunkSize: 256,
            prefillChunkThreshold: 512, tensorData: ["shared": explicit])
        let merged = options.mergingTensorData([
            "shared": URL(fileURLWithPath: "/bundle.bin"),
            "extra": URL(fileURLWithPath: "/extra.bin"),
        ])
        #expect(merged.variant == "coreai-sequential")
        #expect(merged.kvCacheStrategy == .growing)
        #expect(merged.kvCacheSize == 4096)
        #expect(merged.prefillChunkSize == 256 && merged.prefillChunkThreshold == 512)
        #expect(merged.tensorData["shared"] == explicit)
        #expect(merged.tensorData["extra"] == URL(fileURLWithPath: "/extra.bin"))
    }
}
