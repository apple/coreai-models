// Additive causal attention mask for dynamic (GPU) engines.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Fills a pre-allocated additive causal attention mask in place.
///
/// Shape `[1, 1, query_len, key_len]`, `Float16`. `0` means attend, ``causalMaskSentinel``
/// (a large negative that becomes ~0 after softmax) means masked. Query row `q` has absolute
/// position `processedTokenCount + q` and attends to key columns `0...(processedTokenCount + q)`;
/// later keys are masked. `query_len` and `key_len` are read from the array's shape
/// (`key_len == processedTokenCount + query_len`).
///
/// For batch=1 this reproduces the implicit `is_causal` path exactly. Factored out, taking an
/// already-allocated ``NDArray``, so the fill math is unit-testable without a compiled model
/// (``NDArrayDescriptor`` has no public initializer, but `NDArray(shape:scalarType:)` does).
func fillAdditiveCausalMask(_ mask: inout NDArray, processedTokenCount: Int) {
    mask.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, shape, strides in
        // shape == [1, 1, query_len, key_len]
        let queryLen = shape[2]
        let keyLen = shape[3]
        for query in 0..<queryLen {
            let attendUpTo = processedTokenCount + query  // absolute position of this query
            for key in 0..<keyLen {
                let offset = query &* strides[2] &+ key &* strides[3]
                ptr[offset] = key <= attendUpTo ? 0 : causalMaskSentinel
            }
        }
    }
}

/// Builds the per-step additive causal mask ``NDArray`` for a dynamic-engine step.
///
/// Derives the shape from the step context — `query_len = tokens.count`,
/// `key_len = processedTokenCount + query_len` (the fetched KV-cache prefix length) — then fills
/// it via ``fillAdditiveCausalMask(_:processedTokenCount:)``. Allocates per step, matching
/// ``TokenInputHandler``'s own per-batch allocation on the dynamic engines.
func makeAdditiveCausalMask(context: InputContext, descriptor: NDArrayDescriptor) -> NDArray {
    let queryLen = context.tokens.count
    let keyLen = context.processedTokenCount + queryLen
    var mask = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, 1, queryLen, keyLen]))
    fillAdditiveCausalMask(&mask, processedTokenCount: context.processedTokenCount)
    return mask
}
