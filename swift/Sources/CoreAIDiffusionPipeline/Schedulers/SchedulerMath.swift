// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate

/// Evenly spaced floats between [start, end].
func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    let scale = (end - start) / Float(count - 1)
    return (0..<count).map { Float($0) * scale + start }
}
