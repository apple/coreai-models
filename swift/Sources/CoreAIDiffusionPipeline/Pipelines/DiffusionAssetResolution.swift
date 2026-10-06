// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Errors during pipeline loading.
public enum PipelineLoadError: Error, LocalizedError {
    case missingComponent(String)
    case missingConfig(String)
    case unsupportedConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let name):
            return "Required component '\(name)' not found in model directory"
        case .missingConfig(let detail):
            return "Invalid bundle configuration: \(detail)"
        case .unsupportedConfiguration(let detail):
            return "Unsupported configuration: \(detail)"
        }
    }
}

/// Resolves `path` against `directory` and verifies the asset exists on disk, throwing
/// `PipelineLoadError.missingComponent` with the attempted filename if not
public func resolveExistingPipelineAsset(_ path: String, in directory: URL, component: String) throws -> URL {
    let url = directory.appendingPathComponent(path)
    guard AIModelAsset.isValid(at: url) else {
        throw PipelineLoadError.missingComponent("\(component) (expected \(url.lastPathComponent))")
    }
    return url
}
