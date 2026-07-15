// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("DiffusionPipeline")
struct DiffusionPipelineTests {
    @Test("Configuration defaults")
    func configurationDefaults() {
        let config = PipelineConfiguration(prompt: "a cat")
        #expect(config.prompt == "a cat")
        #expect(config.negativePrompt == "")
        #expect(config.stepCount == 50)
        #expect(config.guidanceScale == 7.5)
        #expect(config.schedulerType == .dpmSolverMultistep)
        #expect(!config.isImageToImage)
    }

    @Test("Scheduler type raw values match pipeline.json")
    func schedulerTypeRawValues() {
        #expect(SchedulerType(rawValue: "dpmpp") == .dpmSolverMultistep)
        #expect(SchedulerType(rawValue: "flow_match_euler") == .discreteFlow)
        #expect(SchedulerType(rawValue: "pndm") == .pndm)
    }

    @Test("Descriptors auto-load and throw on bad path")
    func descriptorsThrowBeforeLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: (any Error).self) { try await fn.inputDescriptors }
        await #expect(throws: (any Error).self) { try await fn.outputDescriptors }
    }

    @Test("Load with bad path throws")
    func loadBadPathThrows() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: (any Error).self) {
            try await fn.loadResources()
        }
    }
}

@Suite("TextEncoder tokenize padding")
struct TextEncoderTokenizeTests {
    @Test("Tokenize closure pads short input to maxLength")
    func padsShortInput() {
        let maxLength = 77
        let tokenize: @Sendable (String) -> [Int32] = { _ in
            var ids = [Int32](1...10)
            if ids.count < maxLength {
                ids += [Int32](repeating: 0, count: maxLength - ids.count)
            }
            return ids
        }
        let ids = tokenize("short prompt")
        #expect(ids.count == 77)
        #expect(ids.last == 0)
    }

    @Test("Tokenize closure truncates long input to maxLength")
    func truncatesLongInput() {
        let maxLength = 77
        let tokenize: @Sendable (String) -> [Int32] = { _ in
            var ids = [Int32](1...100)
            if ids.count > maxLength {
                ids = Array(ids.prefix(maxLength))
            }
            return ids
        }
        let ids = tokenize("very long prompt that produces many tokens")
        #expect(ids.count == 77)
        #expect(ids[76] == 77)
    }
}
