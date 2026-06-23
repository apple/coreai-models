// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// In-memory representation of a model bundle directory (`metadata.json` + assets).
///
/// `ModelBundle` parses only the common fields shared across every bundle kind.
/// Kind-specific config blocks (`language`, `vlm`, `diffusion`, `segmenter`) are
/// decoded by per-kind types in their respective runner modules
/// (`LanguageBundle` in CoreAILanguageModels, etc.) using the preserved `raw`
/// JSON.
///
/// Two access patterns:
/// - **Inspection**: `let bundle = try ModelBundle(at: url)` then
///   `bundle.language?.tokenizer` (extension property in CoreAILanguageModels).
///   Lossy — returns `nil` for kind mismatch or malformed payload.
/// - **Strict load**: `let lang = try LanguageBundle(at: url)`. Throws on
///   kind mismatch / missing required fields. Use when the caller has
///   committed to a specific kind.
public struct ModelBundle: Sendable {
    public let metadataVersion: String
    public let kind: BundleKind
    public let name: String
    public let bundlePath: URL
    public let userData: [String: String]?

    /// Role-to-filename mapping from the `"assets"` field in metadata.json.
    public let assets: [String: String]

    /// Full metadata.json bytes, preserved so kind-specific decoders can read
    /// their own blocks without re-reading the file.
    public let raw: Data

    // MARK: - Component Keys

    public enum ComponentKey {
        public static let main = "main"
        public static let vision = "vision"
        public static let embedding = "embedding"
    }

    /// All component keys declared in this bundle's `assets` map, sorted.
    public var componentKeys: [String] {
        assets.keys.sorted()
    }

    /// Resolve a component's URL within the bundle by role key.
    public func modelURL(for key: String) -> URL? {
        guard let path = assets[key] else { return nil }
        return bundlePath.appending(path: path)
    }

    /// Required-component variant — throws `BundleError.missingField` if absent.
    public func requireModelURL(for key: String) throws -> URL {
        guard let url = modelURL(for: key) else {
            throw BundleError.missingField("assets.\(key)")
        }
        return url
    }

    /// Verify all declared assets exist on disk. Throws `BundleError.missingAsset`
    /// with guidance if a component is missing (e.g. after manual compilation).
    public func verify() throws {
        let fm = FileManager.default
        for (key, filename) in assets {
            let url = bundlePath.appending(path: filename)
            if !fm.fileExists(atPath: url.path) {
                throw BundleError.missingAsset(key: key, path: url)
            }
        }
    }

    // MARK: - Errors

    public enum BundleError: Error, CustomStringConvertible {
        case missingMetadata(URL)
        case malformedMetadata(URL, underlying: Error)
        case unsupportedVersion(String)
        case kindMismatch(expected: BundleKind, got: BundleKind)
        case missingField(String)
        case missingAsset(key: String, path: URL)
        case pointedAtModelAsset(URL)

        public var description: String {
            switch self {
            case .pointedAtModelAsset(let url):
                return "'\(url.lastPathComponent)' is a model asset, not a model bundle "
                    + "directory. A model bundle directory contains metadata, a tokenizer, "
                    + "and a model asset."
            case .missingMetadata(let url):
                return "metadata.json not found at \(url.path)"
            case .malformedMetadata(let url, let err):
                return "malformed metadata.json at \(url.path): \(err)"
            case .unsupportedVersion(let v):
                return "unsupported metadata_version '\(v)' (known: 0.2)"
            case .kindMismatch(let expected, let got):
                return "expected bundle kind \(expected), got \(got)"
            case .missingField(let name):
                return "metadata is missing required field '\(name)'"
            case .missingAsset(let key, let path):
                return """
                    Asset '\(key)' not found at \(path.path). \
                    If you compiled this model with `xcrun coreai-build compile`, \
                    update metadata.json "assets" to reference the compiled filename \
                    (e.g. modelName.architectureName.aimodelc). See models/README.md#compiled-models
                    """
            }
        }
    }

    // MARK: - Initialization

    public init(from path: String) throws {
        let expanded = (path as NSString).expandingTildeInPath
        try self.init(at: URL(fileURLWithPath: expanded, isDirectory: true))
    }

    public init(at url: URL) throws {
        // A model bundle is a *directory* (metadata.json + assets + tokenizer).
        // If the caller points us directly at a `.aimodel`/`.aimodelc` asset,
        // fail with actionable guidance. This must run before any filesystem
        // read: a compiled `.aimodelc` is itself a directory holding its own
        // unrelated metadata.json, which would otherwise parse as a bogus 0.1
        // bundle and surface a misleading "unsupported metadata_version" error.
        let ext = url.pathExtension.lowercased()
        if ext == "aimodel" || ext == "aimodelc" {
            throw BundleError.pointedAtModelAsset(url)
        }
        let metadataURL = url.appending(path: "metadata.json")
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            throw BundleError.missingMetadata(metadataURL)
        }
        let raw: Data
        do {
            raw = try Data(contentsOf: metadataURL)
        } catch {
            throw BundleError.malformedMetadata(metadataURL, underlying: error)
        }
        try self.init(raw: raw, bundlePath: url)
    }

    /// Designated init from raw bytes + bundle URL.
    public init(raw: Data, bundlePath: URL) throws {
        self.raw = raw
        self.bundlePath = bundlePath

        let envelope: VersionEnvelope
        do {
            envelope = try JSONDecoder().decode(VersionEnvelope.self, from: raw)
        } catch {
            throw BundleError.malformedMetadata(
                bundlePath.appending(path: "metadata.json"), underlying: error)
        }

        let version = envelope.metadataVersion ?? "0.1"
        guard version == "0.2" else {
            throw BundleError.unsupportedVersion(version)
        }

        let common: CommonFields
        do {
            common = try JSONDecoder().decode(CommonFields.self, from: raw)
        } catch {
            throw BundleError.malformedMetadata(
                bundlePath.appending(path: "metadata.json"), underlying: error)
        }
        self.metadataVersion = "0.2"
        self.kind = common.kind
        self.name = common.name
        self.userData = common.userData
        self.assets = common.assets ?? [:]
    }
}

// MARK: - Internal Codable shapes

extension ModelBundle {
    fileprivate struct VersionEnvelope: Decodable {
        let metadataVersion: String?

        enum CodingKeys: String, CodingKey {
            case metadataVersion = "metadata_version"
        }
    }

    fileprivate struct CommonFields: Decodable {
        let kind: BundleKind
        let name: String
        let userData: [String: String]?
        let assets: [String: String]?

        enum CodingKeys: String, CodingKey {
            case kind, name, assets
            case userData = "user_data"
        }
    }
}
