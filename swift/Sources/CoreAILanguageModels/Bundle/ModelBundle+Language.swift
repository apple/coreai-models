// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

extension ModelBundle {
    /// Lossy peek for inspection: returns a `LanguageModelBundle` if and only if
    /// this bundle's `kind == .llm` and the LLM payload decodes cleanly.
    ///
    /// Returns `nil` for any other kind, missing fields, or malformed JSON.
    /// Strict callers should use `LanguageModelBundle(at:)` or
    /// `LanguageModelBundle(bundle:)` directly.
    public var language: LanguageModelBundle? {
        try? LanguageModelBundle(bundle: self)
    }
}
