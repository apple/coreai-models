// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

/// Input-count validation on the `[Float]` inference paths.
///
/// These paths copy `data.count` elements into an array sized by the descriptor, so
/// before `requireExactCount` was wired in, a short buffer left the tail zeroed and
/// produced silently wrong output, and an oversized one ran past the allocation.
/// Dynamic shapes make a mis-computed sequence length far likelier, so the throw
/// matters more than it used to.
///
/// The `fillStrided` capacity check is a `precondition` — a process trap, not a
/// thrown error — so it cannot be exercised in-process. It is the backstop for a
/// programming error; the recoverable case is what these tests cover.
@Suite("Input count validation")
struct InputValidationTests {
    // MARK: - Error surface (no asset needed)

    @Test("inputCountMismatch names the input, both counts, and the positional-binding hint")
    func inputCountMismatchDescription() {
        let err = CoreAIDiffusionError.inputCountMismatch(
            name: "hidden_states", shape: [1, 4096, 128], expected: 524_288, got: 4096)
        let msg = err.errorDescription ?? ""
        #expect(msg.contains("hidden_states"))
        #expect(msg.contains("524288"))
        #expect(msg.contains("4096"))
        #expect(msg.contains("[1, 4096, 128]"))
        // The hint matters: binding is positional, so a mismatch usually means the
        // caller passed inputs in the wrong order rather than sized one wrongly.
        #expect(msg.lowercased().contains("positional"))
    }

    // MARK: - Real-asset integration (skipped when exports aren't present)

    @Test(
        "run(floatInputs:) accepts an exactly-sized buffer",
        .enabled(if: Self.floatInputAssetURL() != nil))
    func acceptsExactCount() async throws {
        guard let url = Self.floatInputAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        guard let (shape, count) = try await Self.firstFloatInput(of: fn) else {
            Issue.record("asset had no static float input to size a buffer from")
            return
        }
        let data = [Float](repeating: 0, count: count)
        // Only asserting it does not throw inputCountMismatch; the model may still
        // reject the all-zero content for other reasons, which is not what's under test.
        do {
            _ = try await fn.run(floatInputs: [(data, shape)])
        } catch let error as CoreAIDiffusionError {
            if case .inputCountMismatch = error {
                Issue.record("exactly-sized buffer was rejected: \(error)")
            }
        } catch {
            // Non-CoreAIDiffusionError failures are unrelated to count validation.
        }
    }

    @Test(
        "run(floatInputs:) rejects a short buffer instead of zero-filling the tail",
        .enabled(if: Self.floatInputAssetURL() != nil))
    func rejectsShortBuffer() async throws {
        guard let url = Self.floatInputAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        guard let (shape, count) = try await Self.firstFloatInput(of: fn), count > 1 else {
            Issue.record("asset had no static float input to size a buffer from")
            return
        }
        let short = [Float](repeating: 0, count: count - 1)
        await #expect(throws: CoreAIDiffusionError.self) {
            _ = try await fn.run(floatInputs: [(short, shape)])
        }
    }

    @Test(
        "run(floatInputs:) rejects an oversized buffer before it can overrun the fill",
        .enabled(if: Self.floatInputAssetURL() != nil))
    func rejectsOversizedBuffer() async throws {
        guard let url = Self.floatInputAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        guard let (shape, count) = try await Self.firstFloatInput(of: fn) else {
            Issue.record("asset had no static float input to size a buffer from")
            return
        }
        // Reaching fillStrided with this would trap; requireExactCount must catch it first.
        let oversized = [Float](repeating: 0, count: count + 1)
        await #expect(throws: CoreAIDiffusionError.self) {
            _ = try await fn.run(floatInputs: [(oversized, shape)])
        }
    }

    @Test(
        "prepackInput rejects a mis-sized buffer",
        .enabled(if: Self.floatInputAssetURL() != nil))
    func prepackRejectsMisSizedBuffer() async throws {
        guard let url = Self.floatInputAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        guard let (shape, count) = try await Self.firstFloatInput(of: fn), count > 1 else {
            Issue.record("asset had no static float input to size a buffer from")
            return
        }
        await #expect(throws: CoreAIDiffusionError.self) {
            _ = try await fn.prepackInput(
                index: 0, data: [Float](repeating: 0, count: count - 1), shape: shape)
        }
    }

    // MARK: - Helpers

    /// The first fully-static float input's `(shape, elementCount)`.
    ///
    /// Returns nil when every input is int-typed or still has a dynamic axis, since
    /// there is no expected count to compare against in either case.
    private static func firstFloatInput(
        of fn: CoreAIDiffusionModelFunction
    ) async throws -> ([Int], Int)? {
        let descs = try await fn.inputDescriptors
        for name in descs.keys.sorted() {
            guard let desc = descs[name] else { continue }
            let isFloat =
                desc.scalarType == .float32 || desc.scalarType == .float16
                || desc.scalarType == .bfloat16
            guard isFloat, desc.shape.allSatisfy({ $0 > 0 }) else { continue }
            return (desc.shape, desc.shape.reduce(1, *))
        }
        return nil
    }

    private static func floatInputAssetURL() -> URL? {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // DiffusionPipelineTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // swift/
            .deletingLastPathComponent()  // repo root
        // VAE decoders take a float latent, so they exercise the [Float] path that
        // the text encoders (int32 token ids) do not.
        let candidates = [
            "exports/FLUX.2-klein-4B/VAEDecoder_half.aimodel",
            "exports/FLUX.2-klein-4B/VAEDecoder.aimodel",
            "exports/stable-diffusion-3.5-medium/VAEDecoder.aimodel",
        ]
        for candidate in candidates {
            let url = repoRoot.appendingPathComponent(candidate)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}
