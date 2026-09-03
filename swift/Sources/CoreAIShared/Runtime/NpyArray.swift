// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Minimal reader for NumPy `.npy` files.
///
/// Parity tooling compares Swift output against references dumped from the PyTorch side,
/// and `.npy` is what those scripts already write. Covers the dtypes that cross that
/// boundary — `float16`, `float32`, `int32`, `int64`, `uint8`, `bool` — in C order only,
/// which is what `numpy.save` produces unless asked otherwise.
public struct NpyArray: Sendable {
    public enum DType: Sendable {
        case float16
        case float32
        case int32
        case int64
        case uint8
        case bool
    }

    public let shape: [Int]
    public let dtype: DType
    public let data: Data

    /// Total element count implied by `shape`.
    public var count: Int { shape.reduce(1, *) }

    public enum NpyError: Error, CustomStringConvertible, LocalizedError {
        case notNpy(URL)
        case unsupportedDType(String, URL)
        case fortranOrder(URL)
        case dtypeMismatch(expected: String, actual: DType)
        case truncated(URL)

        public var description: String {
            switch self {
            case .notNpy(let url):
                return "Not a .npy file: \(url.path)"
            case .unsupportedDType(let header, let url):
                return "Unsupported .npy dtype in \(url.lastPathComponent) (header: \(header))"
            case .fortranOrder(let url):
                return
                    "\(url.lastPathComponent) is Fortran-ordered; re-save with numpy.ascontiguousarray"
            case .dtypeMismatch(let expected, let actual):
                return "Cannot read \(actual) as \(expected)"
            case .truncated(let url):
                return "\(url.lastPathComponent) is shorter than its header declares"
            }
        }

        public var errorDescription: String? { description }
    }

    public init(shape: [Int], dtype: DType, data: Data) {
        self.shape = shape
        self.dtype = dtype
        self.data = data
    }

    public static func load(_ url: URL) throws -> NpyArray {
        let raw = try Data(contentsOf: url)
        guard raw.count > 10, raw[raw.startIndex] == 0x93, raw[raw.startIndex + 1] == 0x4E else {
            throw NpyError.notNpy(url)
        }
        let version = raw[raw.startIndex + 6]
        let headerLength: Int
        let headerStart: Int
        if version == 1 {
            headerLength = Int(raw[raw.startIndex + 8]) | (Int(raw[raw.startIndex + 9]) << 8)
            headerStart = 10
        } else {
            headerLength =
                Int(raw[raw.startIndex + 8]) | (Int(raw[raw.startIndex + 9]) << 8)
                | (Int(raw[raw.startIndex + 10]) << 16) | (Int(raw[raw.startIndex + 11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLength
        guard raw.count >= dataStart else { throw NpyError.truncated(url) }
        let header =
            String(data: raw.subdata(in: headerStart..<dataStart), encoding: .ascii) ?? ""

        // Reject column-major files rather than reading them as C-order. numpy writes
        // `fortran_order: True` for any transposed view — a `permute` in the trace script
        // is enough — and interpreting that as row-major yields plausible-looking garbage:
        // near-zero cosine on data that is actually correct. Generators should pass the
        // array through `np.ascontiguousarray` first.
        if header.contains("'fortran_order': True") {
            throw NpyError.fortranOrder(url)
        }

        let dtype: DType
        // Order matters: "b1" would also match a substring search for "1", and "i8"
        // must be tested before "i4" is ruled out by an over-broad pattern.
        if header.contains("f2") {
            dtype = .float16
        } else if header.contains("f4") {
            dtype = .float32
        } else if header.contains("i4") {
            dtype = .int32
        } else if header.contains("i8") {
            dtype = .int64
        } else if header.contains("u1") {
            dtype = .uint8
        } else if header.contains("b1") {
            dtype = .bool
        } else {
            throw NpyError.unsupportedDType(header, url)
        }

        return NpyArray(
            shape: parseShape(from: header),
            dtype: dtype,
            data: raw.subdata(in: dataStart..<raw.count)
        )
    }

    private static func parseShape(from header: String) -> [Int] {
        guard let start = header.range(of: "("),
            let end = header.range(of: ")", range: start.upperBound..<header.endIndex)
        else { return [] }
        return header[start.upperBound..<end.lowerBound]
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    /// Every element widened to `Float`, in C order.
    public func asFloat() -> [Float] {
        let n = count
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { raw in
            switch dtype {
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                let source = raw.bindMemory(to: Float16.self)
                for i in 0..<n { out[i] = Float(source[i]) }
                #else
                fatalError("Float16 is not supported on this platform")
                #endif
            case .float32:
                let source = raw.bindMemory(to: Float.self)
                for i in 0..<n { out[i] = source[i] }
            case .int32:
                let source = raw.bindMemory(to: Int32.self)
                for i in 0..<n { out[i] = Float(source[i]) }
            case .int64:
                let source = raw.bindMemory(to: Int64.self)
                for i in 0..<n { out[i] = Float(source[i]) }
            case .uint8, .bool:
                let source = raw.bindMemory(to: UInt8.self)
                for i in 0..<n { out[i] = Float(source[i]) }
            }
        }
        return out
    }

    /// Every element as `Int32`, truncating a float dtype toward zero.
    ///
    /// Deliberately permissive about floats. Token-id references are usually saved as
    /// `int64`, but a generator that computed them through a float tensor produces `f4`,
    /// and refusing that would turn a readable dump into a failed parity run for no reason.
    public func asInt32() -> [Int32] {
        let n = count
        var out = [Int32](repeating: 0, count: n)
        data.withUnsafeBytes { raw in
            switch dtype {
            case .int32:
                let source = raw.bindMemory(to: Int32.self)
                for i in 0..<n { out[i] = source[i] }
            case .int64:
                let source = raw.bindMemory(to: Int64.self)
                for i in 0..<n { out[i] = Int32(truncatingIfNeeded: source[i]) }
            case .float32:
                let source = raw.bindMemory(to: Float.self)
                for i in 0..<n { out[i] = Int32(source[i]) }
            case .float16:
                #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                let source = raw.bindMemory(to: Float16.self)
                for i in 0..<n { out[i] = Int32(source[i]) }
                #else
                fatalError("Float16 is not supported on this platform")
                #endif
            case .uint8, .bool:
                let source = raw.bindMemory(to: UInt8.self)
                for i in 0..<n { out[i] = Int32(source[i]) }
            }
        }
        return out
    }

    /// Raw bytes as `UInt8`. Accepts `uint8` and `bool`, which share a representation —
    /// this is also how `numpy.packbits` output is read back.
    public func asUInt8() throws -> [UInt8] {
        guard dtype == .uint8 || dtype == .bool else {
            throw NpyError.dtypeMismatch(expected: "UInt8", actual: dtype)
        }
        let n = count
        var out = [UInt8](repeating: 0, count: n)
        data.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt8.self)
            for i in 0..<n { out[i] = source[i] }
        }
        return out
    }
}

extension NpyArray.DType: Equatable {}
