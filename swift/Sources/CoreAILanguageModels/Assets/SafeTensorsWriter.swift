// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Writes a `.safetensors` file, the layout ``SafeTensorsReader`` reads.
enum SafeTensorsWriter {
    /// One tensor to write: its raw bytes, laid out as `shape` elements of `dtype`.
    struct Tensor {
        let name: String
        /// Safetensors dtype string, e.g. `"U8"` or `"F16"`.
        let dtype: String
        let shape: [Int]
        let bytes: Data
    }

    /// The file holding `tensors`, in order, and `metadata`. The header is padded with spaces to a
    /// multiple of 8 bytes, as the format recommends, so every tensor starts aligned.
    static func file(_ tensors: [Tensor], metadata: [String: String] = [:]) throws -> Data {
        var header: [String: Any] = [:]
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var offset = 0
        for tensor in tensors {
            header[tensor.name] = [
                "dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, offset + tensor.bytes.count],
            ]
            offset += tensor.bytes.count
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
        var out = Data(capacity: 8 + json.count + offset)
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        for tensor in tensors { out.append(tensor.bytes) }
        return out
    }
}
