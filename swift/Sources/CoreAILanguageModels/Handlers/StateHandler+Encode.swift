// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Metal

/// Encode an inference step with KV cache states, optional additional MTLBuffer
/// states, and logits output.
func encodeWithStates(
    function: InferenceFunction,
    inputs: [String: InferenceFunction.AsyncValue],
    keyState: inout InferenceFunction.AsyncMutableValue,
    keyCacheName: String,
    valState: inout InferenceFunction.AsyncMutableValue,
    valueCacheName: String,
    additionalStates: FixedMTLBufferState?,
    logitsBuffer: MTLBuffer,
    logitsName: String,
    logitsShape: [Int],
    logitsStrides: [Int],
    computeStream: ComputeStream
) throws {
    var asyncStates = InferenceFunction.AsyncMutableViews()
    asyncStates.insert(&keyState, for: keyCacheName)
    asyncStates.insert(&valState, for: valueCacheName)
    additionalStates?.bind(into: &asyncStates)

    var logitsOutput = unsafe InferenceFunction.AsyncMutableValue(
        unsafeBuffer: logitsBuffer, byteOffset: 0,
        scalarType: .float16, shape: logitsShape, strides: logitsStrides)
    var asyncOutputs = InferenceFunction.AsyncMutableViews()
    asyncOutputs.insert(&logitsOutput, for: logitsName)
    let _ = try function.encode(
        inputs: inputs, states: consume asyncStates,
        outputViews: consume asyncOutputs, to: computeStream)
}
