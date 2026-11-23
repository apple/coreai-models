// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Errors from reading or running an embedding bundle.
public enum EmbeddingError: Error, CustomStringConvertible, LocalizedError {
    /// The `embedding` block breaks the contract, or a tower needs an input nothing provides.
    case invalidMetadata(String)
    /// The call needs a tower the bundle does not have.
    case modalityNotInBundle(EmbeddingModality)
    /// A width not in `matryoshkaDimensions`.
    case unsupportedDimension(Int, supported: [Int])
    /// Embeddings from different spaces (or, for `cosine`, of different widths).
    case incompatibleEmbeddings(String)
    case imageConversionFailed
    /// Core AI failed to load or run a program, or a program returned something other than its
    /// declared outputs.
    case programFailed(String)

    public var description: String {
        switch self {
        case .invalidMetadata(let reason):
            return "invalid embedding metadata: \(reason)"
        case .modalityNotInBundle(let modality):
            return "the bundle has no \(modality.rawValue) tower"
        case .unsupportedDimension(let width, let supported):
            return "dimension \(width) is not one of \(supported)"
        case .incompatibleEmbeddings(let pair):
            return "embeddings from different spaces or widths: \(pair)"
        case .imageConversionFailed:
            return "cannot convert the image to RGB"
        case .programFailed(let reason):
            return reason
        }
    }

    public var errorDescription: String? { description }
}
