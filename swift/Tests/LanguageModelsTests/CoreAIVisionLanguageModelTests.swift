// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import CoreGraphics
import Foundation
import FoundationModels
import Testing

/// Loads a Gemma 4 vision-language bundle through the public `CoreAIVisionLanguageModel` API and asks
/// about an image, on the bundle named by `COREAI_TEST_GEMMA_VLM_BUNDLE` (an export made with
/// `models/gemma4/export.py --include-vision`). Skipped without it.
@Suite(
    "CoreAIVisionLanguageModel with a decoder-gather bundle", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA_VLM_BUNDLE"] != nil))
struct CoreAIVisionLanguageModelTests {
    /// A blue canvas with a red square on the left and a yellow circle on the right.
    private func shapesImage() -> CGImage {
        let width = 640
        let height = 480
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(red: 30 / 255, green: 60 / 255, blue: 200 / 255, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(red: 220 / 255, green: 30 / 255, blue: 30 / 255, alpha: 1)
        context.fill(CGRect(x: 100, y: 160, width: 200, height: 200))
        context.setFillColor(red: 250 / 255, green: 220 / 255, blue: 40 / 255, alpha: 1)
        context.fillEllipse(in: CGRect(x: 380, y: 160, width: 180, height: 180))
        return context.makeImage()!
    }

    @Test("A Gemma VLM bundle loads and answers about an image attachment")
    func respondsAboutAnImage() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA_VLM_BUNDLE"])
        let model = try await CoreAIVisionLanguageModel(resourcesAt: URL(fileURLWithPath: path))
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            options: GenerationOptions(maximumResponseTokens: 48)
        ) {
            Attachment(shapesImage())
            "What colors are the shapes in this image?"
        }
        let text = response.content.lowercased()
        #expect(["red", "yellow", "blue"].contains(where: text.contains), "response: \(text)")
    }
}
