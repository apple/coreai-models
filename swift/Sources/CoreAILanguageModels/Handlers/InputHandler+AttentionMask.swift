// Additive causal attention mask for dynamic (GPU) engines.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Fills a `Float16` `[1, 1, queryLen, keyLen]` additive causal mask in place, reading both lengths
/// from the array's shape. Query `i` gets 0 for keys `0...processedTokenCount + i` and
/// ``causalMaskSentinel`` after that, which matches the implicit `is_causal` path.
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

/// Allocates and fills a new single-row causal mask for one dynamic-engine step, with
/// `queryLen = tokens.count` and `keyLen = processedTokenCount + queryLen`.
func makeAdditiveCausalMask(context: InputContext, descriptor: NDArrayDescriptor) -> NDArray {
    let queryLen = context.tokens.count
    let keyLen = context.processedTokenCount + queryLen
    var mask = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, 1, queryLen, keyLen]))
    fillAdditiveCausalMask(&mask, processedTokenCount: context.processedTokenCount)
    return mask
}
