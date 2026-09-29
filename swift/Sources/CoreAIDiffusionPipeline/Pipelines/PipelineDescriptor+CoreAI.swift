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
