// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Metal

// MARK: - Linear Attention State Binding

/// Immutable binding for a fixed-size linear attention state (conv or recurrent).
/// Unlike KV cache, these states never grow — they hold a single rolling state buffer
/// that is updated in-place on every encode step.
struct LinearAttnStateBinding {
    let name: String
    let buffer: MTLBuffer
    let scalarType: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
}