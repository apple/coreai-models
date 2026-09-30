// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Errors during pipeline loading.
public enum PipelineLoadError: Error, LocalizedError {
    case missingComponent(String)
    case missingConfig(String)
    case deprecatedFormat(String)
    case unsupportedConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let name):
            return "Required component '\(name)' not found in model directory"
        case .missingConfig(let detail):
            return "Invalid bundle configuration: \(detail)"
        case .deprecatedFormat(let message):
            return message
        case .unsupportedConfiguration(let detail):
            return "Unsupported configuration: \(detail)"
        }
    }
}

/// Resolves `path` against `directory` and verifies the asset exists on disk, throwing
/// `PipelineLoadError.missingComponent` with the attempted filename if not — e.g. when a
/// pipeline descriptor still names a source `.aimodel` that's since been compiled to
/// `.aimodelc` without updating the descriptor.
public func resolveExistingPipelineAsset(_ path: String, in directory: URL, component: String) throws -> URL {
    let url = directory.appendingPathComponent(path)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw PipelineLoadError.missingComponent("\(component) (expected \(url.lastPathComponent))")
    }
    return url
}
