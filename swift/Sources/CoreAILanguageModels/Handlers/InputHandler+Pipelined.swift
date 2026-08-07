// Asynchronous token input handler for the Pipelined engine (MTLBuffer-based).
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Metal

/// Input handler for the pipelined engine. Owns the token and position MTLBuffers,
/// handles buffer rotation, and the prefill/decode source split.
///
/// `decodeOutputBuffers` is shared with the engine (GPU sampler writes next token
/// there; this handler reads the previous step's token during decode).
struct PipelinedTokenInputHandler {
    let inputIdsName: String
    let positionIdsName: String
    let inputIdsDescriptor: NDArrayDescriptor
    let positionIdsDescriptor: NDArrayDescriptor

    let inputTokensBuffer: MTLBuffer
    let cachePositionBuffers: [MTLBuffer]
    let decodeOutputBuffers: [MTLBuffer]
    let pipelineDepth: Int

    /// Build async input values for one encode step.
    ///
    /// - Prefill (`!tokens.isEmpty`): writes tokens to `inputTokensBuffer` at their
    ///   natural position (disjoint from prior chunks still in-flight on GPU).
    /// - Decode (`tokens.isEmpty`): reads from previous step's `decodeOutputBuffer`.
    func prepare(
        tokens: some Collection<Int32>,
        processedTokenCount: Int,
        step: Int
    ) throws -> [String: InferenceFunction.AsyncValue] {
        let queryLength = tokens.isEmpty ? 1 : tokens.count

        // Write tokens (prefill only — decode reads from GPU sampler output)
        if !tokens.isEmpty {
            let dst = inputTokensBuffer.contents()
                .assumingMemoryBound(to: Int32.self)
                .advanced(by: processedTokenCount)
            let tokenArray = Array(tokens)
            tokenArray.withUnsafeBufferPointer { src in
                memcpy(dst, src.baseAddress!, tokenArray.count * MemoryLayout<Int32>.size)
            }
        }

        // Token input
        let tokenShape = [1, queryLength]
        let tokenStrides = try resolvedStrides(descriptor: inputIdsDescriptor, shape: tokenShape)
        let tokenValue: InferenceFunction.AsyncValue
        if tokens.isEmpty {
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: decodeOutputBuffers[(step + pipelineDepth - 1) % pipelineDepth],
                byteOffset: 0, scalarType: .int32, shape: tokenShape, strides: tokenStrides)
        } else {
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: inputTokensBuffer,
                byteOffset: processedTokenCount * MemoryLayout<Int32>.size,
                scalarType: .int32, shape: tokenShape, strides: tokenStrides)
        }

        // Position input (rotating buffer)
        let posLength = processedTokenCount + queryLength
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: positionIdsDescriptor, shape: posShape)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePositionBuffers[step % pipelineDepth],
            byteOffset: 0, scalarType: .int32, shape: posShape, strides: posStrides)

        return [
            inputIdsName: tokenValue,
            positionIdsName: posValue,
        ]
    }

    /// Prepare inputs for warmup (always prefill mode, writes dummy tokens).
    func prepareWarmup(
        shape: Int,
        processedTokenCount: Int,
        step: Int
    ) throws -> [String: InferenceFunction.AsyncValue] {
        let ptr = inputTokensBuffer.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<shape { ptr[i] = 1 }

        let tokenShape = [1, shape]
        let tokenStrides = try resolvedStrides(descriptor: inputIdsDescriptor, shape: tokenShape)
        let tokenValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: inputTokensBuffer, byteOffset: 0,
            scalarType: .int32, shape: tokenShape, strides: tokenStrides)

        let posLength = processedTokenCount + shape
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: positionIdsDescriptor, shape: posShape)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePositionBuffers[step % pipelineDepth],
            byteOffset: 0, scalarType: .int32, shape: posShape, strides: posStrides)

        return [
            inputIdsName: tokenValue,
            positionIdsName: posValue,
        ]
    }
}
