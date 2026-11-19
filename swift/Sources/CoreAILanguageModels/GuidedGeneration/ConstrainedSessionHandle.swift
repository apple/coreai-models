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
    /// What the session was built for, read from the session itself. A cached handle is reused only
    /// for a request with an equal key.
    let key: ConstrainedSessionKey

    /// Tokenizer used for jump-forward string encoding.
    private let tokenizer: any Tokenizer

    private var _session: ConstrainedGenerationSession

    init(session: consuming ConstrainedGenerationSession, tokenizer: any Tokenizer) {
        self.key = ConstrainedSessionKey(
            schema: session.schema, maxWhitespaceCount: session.maxWhitespaceCount,
            stopTokenIds: session.stopTokenIds, vocabularySize: session.vocabularySize)
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

/// Everything a grammar session is built from. Sessions with equal keys accept the same tokens, so one
/// can be reset and reused for the other's request.
///
/// Stop tokens count: a request's single-token `stop` strings are among them, and they decide where
/// the grammar may end. They compare as a set.
package struct ConstrainedSessionKey: Hashable, Sendable {
    let schema: String
    let maxWhitespaceCount: Int?
    /// Sorted, without repeats; nil for xgrammar's defaults.
    let stopTokenIds: [Int32]?
    let vocabularySize: Int

    package init(schema: String, maxWhitespaceCount: Int?, stopTokenIds: [Int32]?, vocabularySize: Int) {
        self.schema = schema
        self.maxWhitespaceCount = maxWhitespaceCount
        self.stopTokenIds = stopTokenIds.map { Array(Set($0)).sorted() }
        self.vocabularySize = vocabularySize
    }
}
