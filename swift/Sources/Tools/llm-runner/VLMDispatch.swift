// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAILanguageModels
import CoreAIShared
import Foundation
import FoundationModels

struct VLMDispatch {
    @available(macOS 27, iOS 27, *)
    static func run(
        bundlePath: String,
        imagePath: String,
        prompt: String,
        maxTokens: Int,
        sampling: SamplingConfiguration
    ) async throws {
        await PerformanceMetrics.shared.reset()
        await PerformanceMetrics.shared.startOverallTiming()

        let modelLoadSpan = InstrumentsProfiler.beginModelLoad(name: (bundlePath as NSString).lastPathComponent)
        let model = try await CoreAIVisionLanguageModel(resourcesAt: URL(fileURLWithPath: bundlePath))
        modelLoadSpan.end()

        let session = LanguageModelSession(model: model)
        let imageURL = URL(fileURLWithPath: (imagePath as NSString).expandingTildeInPath)
        let options = GenerationOptions(maximumResponseTokens: maxTokens)

        print("Generating...")
        let t0 = ContinuousClock.now
        var tokenCount = 0
        for try await partial in session.streamResponse(options: options) {
            Attachment(imageURL: imageURL)
            prompt
        } onPartialResponse: { partial in
            print(partial.content, terminator: "")
            fflush(stdout)
            tokenCount += 1
        }
        let elapsed = (ContinuousClock.now - t0).inSeconds
        print("\n\n--- \(tokenCount) tokens  \(String(format: "%.1f", Double(tokenCount) / elapsed)) tok/s ---")
        await PerformanceMetrics.shared.endOverallTiming()
        await PerformanceMetrics.shared.printSummary(verbose: CLILogger.isVerbose)
    }
}
