// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Tokenizers

/// Single-owner handle wrapping a `ConstrainedGenerationSession`.
///
/// Not thread-safe — ownership transfers between the engine cache and the
/// active generation task via the checkout/checkin pattern. Only one task
/// holds the handle at a time; concurrent access is a programming error.
package final class ConstrainedSessionHandle: @unchecked Sendable {
    /// Schema string is immutable after init — cached to avoid repeated access.
    let schema: String

    /// Tokenizer used for jump-forward string encoding.
    private let tokenizer: any Tokenizer

    private var _session: ConstrainedGenerationSession

    init(session: consuming ConstrainedGenerationSession, tokenizer: any Tokenizer) {
        self.schema = session.schema
        self.tokenizer = tokenizer
        self._session = session
    }

    var isTerminated: Bool { _session.isTerminated }

    func acceptToken(_ token: Int32) -> Bool {
        _session.acceptToken(token)
    }

    func fillBitmask(into ptr: UnsafeMutablePointer<Int32>) -> ConstrainedGenerationSession.BitmaskResult {
        _session.fillBitmask(into: ptr)
    }

    func reset() { _session.reset() }

    @discardableResult
    func rollback(_ numTokens: Int = 1) -> Bool { _session.rollback(numTokens) }

    func findJumpForwardString() -> String? {
        _session.findJumpForwardString()
    }

    /// Decode a sequence of token IDs back into a string using the stored tokenizer.
    func decodeTokens(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map(Int.init))
    }

    /// Tokenize a string for jump-forward injection. Uses the stored tokenizer
    /// without adding special tokens.
    func tokenizeForJumpForward(_ string: String) -> [Int32] {
        tokenizer.encode(text: string).map(Int32.init)
    }
}
