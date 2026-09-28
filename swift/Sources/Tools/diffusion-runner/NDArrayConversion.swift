// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Widen a `[Float]` buffer into an `NDArray` of the requested scalar type.
///
/// Not diffusion- or model-specific — a plain numeric adapter used by the parity path to feed
/// reference tensors (loaded as `Float`) to a model function at its native scalar type.
func floatsToNDArray(_ floats: [Float], asInt32: Bool, shape: [Int], scalarType: NDArray.ScalarType? = nil)
    -> NDArray
{
    if asInt32 {
        var array = NDArray(shape: shape, scalarType: .int32)
        let view = array.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<floats.count { ptr[i] = Int32(floats[i]) }
        }
        return array
    } else if scalarType == .float16 {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        var array = NDArray(shape: shape, scalarType: .float16)
        let view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<floats.count { ptr[i] = Float16(floats[i]) }
        }
        return array
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    } else {
        var array = NDArray(shape: shape, scalarType: scalarType ?? .float32)
        let view = array.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<floats.count { ptr[i] = floats[i] }
        }
        return array
    }
}
