// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A program the host can call (R3): functions of one asset with declared inputs and outputs.
struct Component: Sendable, Equatable {
    /// A key of the bundle's `assets`.
    var asset: String
    var inputs: [String: TensorSpec]
    var outputs: [String: TensorSpec]
    var computeUnits: [ComputeUnit]
    var functions: [FunctionEntry]
    /// How to build the `bias` input (EmbeddingGemma 2 iOS `encoder`).
    var bias: AdditiveBias?
}

/// A compute unit a component's programs run on.
enum ComputeUnit: String, Sendable, Decodable {
    case gpu
    case neuralEngine = "neural_engine"
}

/// `bias[0, k, 0, q]` is `unmasked` when key `k` is a real token, else `masked`.
struct AdditiveBias: Sendable, Equatable, Decodable {
    var masked: Float
    var unmasked: Float
}

/// One `functions` entry: a function name and the batch and length it takes.
struct FunctionEntry: Sendable, Equatable {
    var function: String
    var batch: ClosedRange<Int>
    /// `nil` for a component without an `L` axis.
    var length: ClosedRange<Int>?
}

/// A tensor declaration, `"dtype[shape]"`, e.g. `float32[B,3,224,224]`.
struct TensorSpec: Sendable, Equatable, CustomStringConvertible {
    enum DType: String, Sendable { case float32, float16, int32, bool }

    /// `B` the batch, `L` the text length, `N` a second batch axis, else a fixed size.
    enum Axis: Sendable, Equatable {
        case batch, length, secondBatch
        case fixed(Int)
    }

    var dtype: DType
    var shape: [Axis]

    /// Parses `spec`; throws `invalidMetadata` on an unknown dtype or axis.
    init(parsing spec: String) throws {
        guard let open = spec.firstIndex(of: "["), spec.hasSuffix("]"),
            let dtype = DType(rawValue: String(spec[..<open]))
        else { throw EmbeddingError.invalidMetadata("bad tensor spec '\(spec)'") }
        let body = spec[spec.index(after: open)..<spec.index(before: spec.endIndex)]
        self.dtype = dtype
        self.shape = try body.split(separator: ",").map { token in
            switch token.trimmingCharacters(in: .whitespaces) {
            case "B": return .batch
            case "L": return .length
            case "N": return .secondBatch
            case let size:
                guard let n = Int(size), n > 0 else {
                    throw EmbeddingError.invalidMetadata("bad axis '\(size)' in '\(spec)'")
                }
                return .fixed(n)
            }
        }
    }

    var description: String {
        let axes = shape.map { axis -> String in
            switch axis {
            case .batch: return "B"
            case .length: return "L"
            case .secondBatch: return "N"
            case .fixed(let n): return String(n)
            }
        }
        return "\(dtype.rawValue)[\(axes.joined(separator: ","))]"
    }
}

// MARK: - Function choice and batching (R4)

/// A call of one function: the rows it embeds and the padded batch and length.
struct Call: Sendable, Equatable {
    var function: String
    /// Indices into the rows passed to `plan`, at most `batch` of them.
    var rows: [Int]
    var batch: Int
    /// The padded token length; `nil` for a component without an `L` axis.
    var length: Int?
}

extension Component {
    /// Splits rows into calls.
    ///
    /// Each row takes the function length that is the shortest one holding it. Rows with the same
    /// length range go into calls of the largest batch not above the rows left; a batch below
    /// `batch.min` is padded. A static length pads to it; a dynamic one pads to the longest row
    /// of the call.
    ///
    /// - Parameters:
    ///   - lengths: each row's token count; ignored for a component without an `L` axis.
    ///   - name: the component name, for errors.
    func plan(lengths: [Int], name: String) throws -> [Call] {
        guard !lengths.isEmpty else { return [] }
        guard functions.contains(where: { $0.length != nil }) else {
            return Self.split(Array(lengths.indices), entries: functions, length: { _ in nil })
        }
        let ranges = Array(Set(functions.compactMap(\.length))).sorted { $0.upperBound < $1.upperBound }
        var groups: [ClosedRange<Int>: [Int]] = [:]
        for (row, length) in lengths.enumerated() {
            guard let range = ranges.first(where: { $0.upperBound >= length }) else {
                throw EmbeddingError.invalidMetadata("no function of \(name) takes \(length) tokens")
            }
            groups[range, default: []].append(row)
        }
        return ranges.flatMap { range -> [Call] in
            guard let rows = groups[range] else { return [] }
            let entries = functions.filter { $0.length == range }
            return Self.split(rows, entries: entries) { call in
                max(range.lowerBound, call.map { lengths[$0] }.max() ?? range.lowerBound)
            }
        }
    }

    private static func split(_ rows: [Int], entries: [FunctionEntry], length: ([Int]) -> Int?) -> [Call] {
        var calls: [Call] = []
        var next = 0
        while next < rows.count {
            let left = rows.count - next
            let entry =
                entries.filter { $0.batch.lowerBound <= left }.max { $0.batch.upperBound < $1.batch.upperBound }
                ?? entries.min { $0.batch.lowerBound < $1.batch.lowerBound }!
            let take = min(left, entry.batch.upperBound)
            let chunk = Array(rows[next..<(next + take)])
            calls.append(
                Call(
                    function: entry.function, rows: chunk, batch: max(take, entry.batch.lowerBound),
                    length: length(chunk)))
            next += take
        }
        return calls
    }
}

// MARK: - Decoding

extension Component: Decodable {
    enum CodingKeys: String, CodingKey {
        case asset, inputs, outputs, functions, bias
        case computeUnits = "compute_units"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        asset = try c.decode(String.self, forKey: .asset)
        inputs = try c.decode([String: String].self, forKey: .inputs).mapValues(TensorSpec.init(parsing:))
        outputs = try c.decode([String: String].self, forKey: .outputs).mapValues(TensorSpec.init(parsing:))
        computeUnits = try c.decode([ComputeUnit].self, forKey: .computeUnits)
        functions = try c.decode([FunctionEntry].self, forKey: .functions)
        bias = try c.decodeIfPresent(AdditiveBias.self, forKey: .bias)
        if computeUnits.isEmpty || functions.isEmpty {
            throw EmbeddingError.invalidMetadata("\(asset): compute_units and functions must not be empty")
        }
    }
}

extension FunctionEntry: Decodable {
    enum CodingKeys: String, CodingKey { case function, batch, length, min, max }

    /// A positive int, or `{min, max}` with `0 < min <= max`.
    static func range(_ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys) throws -> ClosedRange<Int> {
        let low: Int
        let high: Int
        if let n = try? c.decode(Int.self, forKey: key) {
            (low, high) = (n, n)
        } else {
            let r = try c.nestedContainer(keyedBy: CodingKeys.self, forKey: key)
            (low, high) = (try r.decode(Int.self, forKey: .min), try r.decode(Int.self, forKey: .max))
        }
        guard 0 < low, low <= high else {
            throw EmbeddingError.invalidMetadata("bad \(key.stringValue) range \(low)...\(high)")
        }
        return low...high
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        function = try c.decode(String.self, forKey: .function)
        batch = try Self.range(c, .batch)
        length = c.contains(.length) ? try Self.range(c, .length) : nil
    }
}
