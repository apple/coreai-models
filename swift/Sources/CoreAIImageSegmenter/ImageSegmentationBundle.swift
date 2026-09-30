// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// A `kind: segmenter` model bundle: the main asset and the tokenizer folder.
public struct ImageSegmentationBundle: Sendable {
    public let modelBundle: ModelBundle
    public let modelURL: URL
    /// Only read by text-capable engines; point-only engines ignore it.
    public let tokenizerFolder: URL

    public init(from path: String) throws {
        try self.init(bundle: ModelBundle(from: path))
    }

    public init(bundle: ModelBundle) throws {
        guard bundle.kind == .segmenter else {
            throw ModelBundle.BundleError.kindMismatch(expected: .segmenter, got: bundle.kind)
        }
        try bundle.verifyAssetsExisting()
        self.modelBundle = bundle
        self.modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        self.tokenizerFolder = bundle.bundlePath.appending(path: "tokenizer")
    }
}
