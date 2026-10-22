// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A diffusion model bundle: a `ModelBundle` plus its decoded `DiffusionConfig`.
///
/// Shared across image (FLUX.2, Sana Sprint) and video (Wan) diffusion — the
/// same split the LLM/VLM bundles use.
///
/// ```swift
/// let bundle = try DiffusionBundle(at: url)
/// let steps = bundle.config.defaultSteps
/// let transformer = bundle.transformerFilename
/// ```
public struct DiffusionBundle: Sendable {
    public let modelBundle: ModelBundle
    public let config: DiffusionConfig

    public init(from path: String) throws {
        let expanded = (path as NSString).expandingTildeInPath
        try self.init(at: URL(fileURLWithPath: expanded, isDirectory: true))
    }

    public init(at url: URL) throws {
        try self.init(bundle: try ModelBundle(at: url))
    }

    public init(bundle: ModelBundle) throws {
        guard bundle.kind == .diffusion || bundle.kind == .videoDiffusion else {
            throw ModelBundle.BundleError.kindMismatch(expected: .diffusion, got: bundle.kind)
        }
        self.modelBundle = bundle
        let payload = try JSONDecoder().decode(DiffusionPayload.self, from: bundle.raw)
        guard let config = payload.diffusion else {
            throw ModelBundle.BundleError.missingField("diffusion")
        }
        self.config = config
    }

    // MARK: - Convenience accessors

    public var name: String { modelBundle.name }
    public var bundlePath: URL { modelBundle.bundlePath }

    /// `true` when this is a video-diffusion bundle (Wan).
    public var isVideo: Bool { modelBundle.kind == .videoDiffusion }

    /// Fine-grained image pipeline type; `nil` for video.
    public var type: DiffusionConfig.PipelineType? { config.type }

    // MARK: - Component assets

    /// The declared filename for a component from the bundle's `assets` map, by
    /// canonical key. Resolve it against `bundlePath` and check existence with
    /// `resolveExistingPipelineAsset(_:in:component:)`.
    public func assetFilename(for key: String) -> String? {
        modelBundle.assets[key]
    }

    public var textEncoderFilename: String? { assetFilename(for: DiffusionComponentKey.textEncoder) }
    public var transformerFilename: String? { assetFilename(for: DiffusionComponentKey.transformer) }
    public var vaeDecoderFilename: String? { assetFilename(for: DiffusionComponentKey.vaeDecoder) }
    public var vaeEncoderFilename: String? { assetFilename(for: DiffusionComponentKey.vaeEncoder) }
}

// MARK: - 0.2 payload shape

extension DiffusionBundle {
    /// Diffusion decodes its own payload rather than sharing the language module's:
    /// `CoreAIDiffusionPipeline` does not depend on `CoreAILanguageModels`; both wrap
    /// `ModelBundle` (from `CoreAIShared`) independently.
    fileprivate struct DiffusionPayload: Decodable {
        let diffusion: DiffusionConfig?
    }
}

extension ModelBundle {
    /// Lossy peek: a `DiffusionBundle` if this is a diffusion kind and its payload
    /// decodes, else `nil`. Use `DiffusionBundle(at:)` when you need the error.
    public var diffusion: DiffusionBundle? {
        try? DiffusionBundle(bundle: self)
    }
}
