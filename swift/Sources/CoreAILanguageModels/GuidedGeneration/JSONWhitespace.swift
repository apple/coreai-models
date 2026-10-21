// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Whitespace policy for JSON-schema-constrained generation.
///
/// Controls which whitespace the compiled grammar accepts between JSON tokens. With
/// ``flexible``, a greedy-decoding model can keep choosing whitespace after the last value and
/// never close the document; ``compact`` fixes the separators so the structure is fully determined
/// by the schema and generation ends when the schema is satisfied.
public enum JSONWhitespace: Sendable, Hashable {
    /// Any amount of whitespace (spaces, tabs, newlines) between JSON tokens. Permits
    /// pretty-printed output, and unbounded runs of whitespace.
    case flexible

    /// Fixed separators only: `", "` between elements and `": "` between keys and values,
    /// with no newlines or indentation. Use when the output is parsed rather than displayed
    /// verbatim; formatting can be reapplied after parsing. The default.
    case compact

    /// Whether the compiled grammar should accept arbitrary whitespace.
    var allowsAnyWhitespace: Bool {
        switch self {
        case .flexible: return true
        case .compact: return false
        }
    }
}
