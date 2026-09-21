// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Everything that can go wrong in the video segmentation runtime.
public enum VideoSegmentationError: Error, CustomStringConvertible, LocalizedError {
    case invalidConfiguration(String)
    case missingFunction(name: String, available: [String])
    case missingOutput(function: String, name: String)
    case shapeMismatch(function: String, input: String, expected: [Int], actual: [Int])
    case unsupportedGeometry(String)
    case noPrompts
    case parityReferenceInvalid(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let detail):
            return detail
        case .missingFunction(let name, let available):
            return
                "Asset has no '\(name)' function. Available: \(available.sorted().joined(separator: ", ")). "
                + "Re-export with models/sam3_video/export.py."
        case .missingOutput(let function, let name):
            return "'\(function)' returned no output named '\(name)'."
        case .shapeMismatch(let function, let input, let expected, let actual):
            // Spelled out, since a shape mismatch SIGKILLs with no traceback of its own.
            return
                "\(function): input '\(input)' has shape \(actual), but the graph was traced "
                + "with \(expected)."
        case .unsupportedGeometry(let detail):
            return detail
        case .noPrompts:
            return "No text prompts were provided; video segmentation needs at least one."
        case .parityReferenceInvalid(let detail):
            return "Parity reference is unusable: \(detail)"
        }
    }

    public var errorDescription: String? { description }
}
