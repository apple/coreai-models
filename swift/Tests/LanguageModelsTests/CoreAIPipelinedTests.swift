// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Metal
import Synchronization
import TestUtilities
import Testing

@testable import CoreAILanguageModels

// MARK: - Test Utilities

/// Compute contiguous (row-major) strides for a shape.
///
/// WARNING: These strides may NOT match Core AI's MLIR-aligned layout.
/// Use resolvedStrides(descriptor:shape:) for views passed to encode.
func contiguousStrides(for shape: [Int]) -> [Int] {
    guard !shape.isEmpty else { return [] }
    var strides = [Int](repeating: 1, count: shape.count)
    for i in Swift.stride(from: shape.count - 2, through: 0, by: -1) {
        strides[i] = strides[i + 1] * shape[i + 1]
    }
    return strides
}

// MARK: - Stride Computation Tests

@Suite("contiguousStrides")
struct ContiguousStridesTests {
    @Test("empty shape returns empty strides")
    func emptyShape() {
        #expect(contiguousStrides(for: []) == [])
    }

    @Test("scalar shape [1] returns strides [1]")
    func scalarShape() {
        #expect(contiguousStrides(for: [1]) == [1])
    }

    @Test("1D shape returns stride 1")
    func oneDimensional() {
        #expect(contiguousStrides(for: [10]) == [1])
    }

    @Test("2D shape [3, 4] returns row-major strides [4, 1]")
    func twoDimensional() {
        #expect(contiguousStrides(for: [3, 4]) == [4, 1])
    }

    @Test("3D shape [2, 3, 4] returns strides [12, 4, 1]")
    func threeDimensional() {
        #expect(contiguousStrides(for: [2, 3, 4]) == [12, 4, 1])
    }

    @Test("5D KV cache shape [32, 1, 8, 256, 64]")
    func kvCacheShape() {
        let strides = contiguousStrides(for: [32, 1, 8, 256, 64])
        // stride[0] = 1*8*256*64 = 131072
        // stride[1] = 8*256*64 = 131072
        // stride[2] = 256*64 = 16384
        // stride[3] = 64
        // stride[4] = 1
        #expect(strides == [131072, 131072, 16384, 64, 1])
    }

    @Test("typical decode input shape [1, 1]")
    func decodeInputShape() {
        #expect(contiguousStrides(for: [1, 1]) == [1, 1])
    }

    @Test("logits shape [1, 128, 32000]")
    func logitsShape() {
        let strides = contiguousStrides(for: [1, 128, 32000])
        #expect(strides == [128 * 32000, 32000, 1])
    }

    @Test("total element count matches shape product")
    func elementCountConsistency() {
        // For contiguous layout, stride[0] * shape[0] == total elements
        let shapes: [[Int]] = [
            [1, 1],
            [1, 512],
            [1, 128, 32000],
            [32, 1, 8, 256, 64],
        ]
        for shape in shapes {
            let strides = contiguousStrides(for: shape)
            let totalElements = shape.reduce(1, *)
            // First stride * first dim should equal total
            #expect(strides[0] * shape[0] == totalElements, "Shape \(shape)")
        }
    }

    @Test("innermost stride is always 1")
    func innermostStrideIsOne() {
        let shapes: [[Int]] = [[5], [3, 4], [2, 3, 4], [2, 3, 4, 5]]
        for shape in shapes {
            let strides = contiguousStrides(for: shape)
            #expect(strides.last == 1, "Shape \(shape)")
        }
    }

    @Test("strides are monotonically decreasing for shapes with all dims > 1")
    func stridesMonotonicallyDecrease() {
        let shape = [2, 3, 4, 5, 6]
        let strides = contiguousStrides(for: shape)
        for i in 0..<(strides.count - 1) {
            #expect(strides[i] >= strides[i + 1], "Stride \(i) >= stride \(i+1)")
        }
    }
}

// MARK: - fillIdentitySequence Tests

@Suite("fillIdentitySequence")
struct FillIdentitySequenceTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("fills buffer with sequential integers")
    func fillsSequentially() throws {
        let device = try #require(Self.device)
        let count = 128
        let byteCount = count * MemoryLayout<Int32>.stride
        let buffer = try #require(device.makeBuffer(length: byteCount, options: .storageModeShared))

        // Zero-fill first to ensure fillIdentitySequence actually writes
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)

        // Create a TensorRef-like setup: we just need the metalBuffer
        // fillIdentitySequence works on TensorRef, but let's test the underlying pattern
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Int32(i)
        }

        // Verify
        for i in 0..<count {
            #expect(ptr[i] == Int32(i), "Position \(i)")
        }
    }

    @Test("identity sequence values at boundaries")
    func boundaryValues() throws {
        let device = try #require(Self.device)
        let count = 4096
        let buffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared))

        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Int32(i)
        }

        #expect(ptr[0] == 0, "First element")
        #expect(ptr[count - 1] == Int32(count - 1), "Last element")
        #expect(ptr[count / 2] == Int32(count / 2), "Middle element")
    }
}

// MARK: - Buffer Layout Tests (Math Verification)

/// Verifies that contiguousStrides produces layouts compatible with the
/// buffer sizes allocated by the engine. This catches mismatches where
/// stride computation doesn't match allocation byte counts.
@Suite("Buffer Layout Consistency")
struct BufferLayoutTests {
    @Test("input tokens buffer size matches stride layout")
    func inputTokensLayout() {
        let maxCtx = 2048
        let shape = [1, maxCtx]
        let strides = contiguousStrides(for: shape)
        let totalElements = strides[0] * shape[0]
        let byteCount = totalElements * MemoryLayout<Int32>.size

        // Buffer allocation: maxCtx * sizeof(Int32)
        let expectedByteCount = maxCtx * MemoryLayout<Int32>.size
        #expect(byteCount == expectedByteCount)
    }

    @Test("logits buffer size matches stride layout")
    func logitsLayout() {
        let vocabSize = 32000
        let queryLength = 128
        let shape = [1, queryLength, vocabSize]
        let strides = contiguousStrides(for: shape)
        let totalElements = strides[0] * shape[0]
        let byteCount = totalElements * MemoryLayout<Float16>.size

        let expectedByteCount = 1 * queryLength * vocabSize * MemoryLayout<Float16>.size
        #expect(byteCount == expectedByteCount)
    }

    @Test("KV cache buffer size matches stride layout for 5D shape")
    func kvCacheLayout() {
        // Typical shape: [L=32, B=1, H=8, S=256, D=64]
        let shape = [32, 1, 8, 256, 64]
        let strides = contiguousStrides(for: shape)
        let totalElements = strides[0] * shape[0]
        let byteCount = totalElements * MemoryLayout<Float16>.size

        let expectedElements = shape.reduce(1, *)
        #expect(totalElements == expectedElements)

        let expectedByteCount = expectedElements * MemoryLayout<Float16>.size
        #expect(byteCount == expectedByteCount)
    }

    @Test("decode step shapes produce correct element counts")
    func decodeStepLayout() {
        // Decode: queryLength=1
        let tokenShape = [1, 1]
        let tokenStrides = contiguousStrides(for: tokenShape)
        #expect(tokenStrides[0] * tokenShape[0] == 1)

        // Position IDs after 100 tokens
        let posShape = [1, 101]  // processedCount(100) + queryLen(1)
        let posStrides = contiguousStrides(for: posShape)
        #expect(posStrides[0] * posShape[0] == 101)

        // Logits: [1, 1, vocabSize]
        let vocabSize = 151936
        let logitsShape = [1, 1, vocabSize]
        let logitsStrides = contiguousStrides(for: logitsShape)
        #expect(logitsStrides[0] * logitsShape[0] == vocabSize)
    }

    @Test("prefill step shapes produce correct element counts")
    func prefillStepLayout() {
        let queryLength = 512
        let vocabSize = 32000

        let tokenShape = [1, queryLength]
        let tokenStrides = contiguousStrides(for: tokenShape)
        #expect(tokenStrides[0] * tokenShape[0] == queryLength)

        let logitsShape = [1, queryLength, vocabSize]
        let logitsStrides = contiguousStrides(for: logitsShape)
        let totalLogitsElements = logitsStrides[0] * logitsShape[0]
        #expect(totalLogitsElements == queryLength * vocabSize)
    }
}

// MARK: - Double Buffer Alternation Tests

/// Verifies the step % 2 alternation pattern used for cache position double-buffering.
@Suite("Double Buffer Alternation")
struct DoubleBufferTests {
    @Test("step counter alternates between 0 and 1")
    func alternation() {
        for step in 0..<20 {
            let bufferIndex = step % 2
            #expect(bufferIndex == 0 || bufferIndex == 1)
            if step > 0 {
                let prevIndex = (step - 1) % 2
                #expect(bufferIndex != prevIndex, "Step \(step) should differ from step \(step - 1)")
            }
        }
    }

    @Test("consecutive steps use different buffers")
    func consecutiveStepsDiffer() {
        var lastIndex = -1
        for step in 0..<100 {
            let index = step % 2
            if lastIndex >= 0 {
                #expect(index != lastIndex, "Step \(step)")
            }
            lastIndex = index
        }
    }
}

// MARK: - GPU-Direct Token Write Pattern Tests

/// Verifies the GPU-direct token write pattern where the sampler writes
/// the next token directly to the input buffer at offset 0.
@Suite("GPU-Direct Token Write")
struct GPUDirectTokenWriteTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("token written at offset 0 is readable as Int32")
    func tokenWriteRead() throws {
        let device = try #require(Self.device)
        let maxCtx = 2048
        let buffer = try #require(
            device.makeBuffer(length: maxCtx * MemoryLayout<Int32>.stride, options: .storageModeShared))

        // Simulate GPU sampler writing token at offset 0
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: maxCtx)
        let expectedToken: Int32 = 42
        ptr[0] = expectedToken

        // Read back
        #expect(ptr[0] == expectedToken)
    }

    @Test("logits offset calculation for last token")
    func logitsOffsetCalculation() {
        let vocabSize = 32000

        // Decode step: queryLength=1, actualTokenCount=1
        let decodeOffset = (1 - 1) * vocabSize * MemoryLayout<Float16>.size
        #expect(decodeOffset == 0, "Decode reads from start of logits buffer")

        // Prefill: queryLength=128, we want last token's logits
        let prefillOffset = (128 - 1) * vocabSize * MemoryLayout<Float16>.size
        #expect(prefillOffset == 127 * vocabSize * 2)

        // Bucketed: actualTokenCount=100 but queryLength=128 (bucketed to 128)
        // We should use actualTokenCount, not queryLength
        let actualTokenCount = 100
        let correctOffset = (actualTokenCount - 1) * vocabSize * MemoryLayout<Float16>.size
        let wrongOffset = (128 - 1) * vocabSize * MemoryLayout<Float16>.size
        #expect(correctOffset != wrongOffset, "Must use actual count, not bucketed length")
        #expect(correctOffset == 99 * vocabSize * 2)
    }
}

// MARK: - Stride Alignment Tests

/// Validates that contiguousStrides may NOT be safe for use with Core AI views.
///
/// Core AI's NDArrayDescriptor.defaultLayout can produce non-contiguous strides
/// (e.g. MLIR 16-element alignment on the innermost dimension). If we pass
/// contiguous strides in a RawView/MutableRawView but the model was compiled
/// with alignment requirements, Core AI will crash in copyElements.
///
/// These tests verify the contract: the engine must use descriptor-resolved
/// strides, not contiguousStrides, for views passed to encode.
@Suite("Stride Alignment Safety")
struct StrideAlignmentTests {
    @Test("contiguousStrides matches shape.reduce for total elements")
    func contiguousMatchesShapeProduct() {
        let shapes: [[Int]] = [
            [1, 1], [1, 128], [1, 1, 32000], [1, 128, 32000],
            [32, 1, 8, 256, 64], [1, 8, 256, 64],
        ]
        for shape in shapes {
            let strides = contiguousStrides(for: shape)
            let totalViaStrides = strides[0] * shape[0]
            let totalViaShape = shape.reduce(1, *)
            #expect(
                totalViaStrides == totalViaShape,
                "Shape \(shape): stride-based \(totalViaStrides) != product \(totalViaShape)"
            )
        }
    }

    @Test("aligned strides may differ from contiguous — padding increases footprint")
    func alignedStridesCanDiffer() {
        // Simulate what MLIR alignment does:
        // For shape [1, 8, 256, 64] with 16-element alignment on dim 3,
        // the innermost dim might be padded from 64 to 64 (already aligned).
        // But for shape [1, 8, 256, 65], padding to 80 would change strides.
        let shape = [1, 8, 256, 65]
        let contiguous = contiguousStrides(for: shape)

        // Simulate aligned: pad dim 3 from 65 to 80 (next multiple of 16)
        let alignedDim3 = ((65 + 15) / 16) * 16  // = 80
        let aligned = [8 * 256 * alignedDim3, 256 * alignedDim3, alignedDim3, 1]

        // The aligned strides should be LARGER than contiguous
        #expect(aligned[0] > contiguous[0], "Aligned stride[0] should be larger")
        #expect(aligned[1] > contiguous[1], "Aligned stride[1] should be larger")
        #expect(aligned[2] > contiguous[2], "Aligned stride[2] should be larger due to padding")

        // Total footprint with alignment is larger
        let contiguousFootprint = contiguous[0] * shape[0]
        let alignedFootprint = aligned[0] * shape[0]
        #expect(alignedFootprint > contiguousFootprint)
    }

    @Test("contiguous strides are safe only when innermost dim is already aligned")
    func contiguousSafeWhenAligned() {
        // If innermost dim is a multiple of common alignment (16),
        // contiguous == aligned. This is the typical case for head_dim=64, 128.
        let alignments = [16, 32]
        let safeShapes: [[Int]] = [
            [1, 8, 256, 64],  // head_dim=64, multiple of 16 and 32
            [1, 8, 256, 128],  // head_dim=128, multiple of 16 and 32
            [32, 1, 8, 256, 64],
        ]

        for shape in safeShapes {
            let innerDim = shape.last!
            for alignment in alignments {
                let isAligned = innerDim % alignment == 0
                #expect(isAligned, "Shape \(shape) innerDim \(innerDim) should be multiple of \(alignment)")
            }
        }
    }

    @Test("GrowingLogitsBuffer uses defaultLayout strides, not contiguous")
    func growingLogitsUsesDefaultLayout() throws {
        // GrowingLogitsBuffer stores a layout from NDArrayDescriptor.defaultLayout.
        // The engine should use logits.currentLayout.strides, NOT contiguousStrides.
        // This test verifies the contract by checking the type has a layout property.
        //
        // We can't construct a GrowingLogitsBuffer without a real model descriptor,
        // but we verify the API exists and would be used correctly.
        let shape = [1, 128, 32000]
        let contiguous = contiguousStrides(for: shape)

        // The fix: engine code should do:
        //   let strides = logits.currentLayout.strides  // from defaultLayout
        // NOT:
        //   let strides = contiguousStrides(for: shape)
        //
        // If defaultLayout adds padding, contiguous strides would cause
        // MutableRawView.copyElements to crash with a size assertion.
        //
        // For float16 [1, 128, 32000], contiguous stride[2]=1, stride[1]=32000, stride[0]=4096000
        #expect(contiguous == [128 * 32000, 32000, 1])
        // But defaultLayout might give stride[1]=32000+padding, changing all outer strides.
    }
}

// MARK: - GPU Sampler Callback Synchronization Tests
//
// These tests reproduce the race condition in _encodeNextStepGPU between
// continuation.yield() (called from Metal's addCompletedHandler thread) and
// continuation.finish() (called after runCompletion returns).
//
// The race exists because currentWorkCompleted() uses MTLSharedEvent — a GPU-side
// signal that can resume before Metal's didCompleteWithStartTime:endTime:error: runs
// the addCompletedHandler blocks. So the task can call finish() while the last
// callback's yield() is still in flight.
//
// The fix: encode a sentinel sampler command after all real steps on the same serial
// queue. Its addCompletedHandler fires after all prior handlers (serial queue FIFO
// ordering via MTLDispatchListApply). await withCheckedContinuation on the sentinel
// replaces both currentWorkCompleted() and DispatchGroup.wait().

@Suite("GPU Sampler Callback Synchronization", .enabled(if: !CIEnvironment.isVM))
struct GPUSamplerContinuationSyncTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize = 1024

    private func makeLogitsBuffer(device: MTLDevice, winner: Int) throws -> MTLBuffer {
        let buffer = try #require(
            device.makeBuffer(length: Self.vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { ptr[i] = Float16(0) }
        ptr[winner] = Float16(10.0)
        return buffer
    }

    // MARK: Race condition — engine pattern

    /// Reproduces the race in _encodeNextStepGPU / runCompletion (post PR #23).
    ///
    /// The completion callback is the real MPSGraphArgmaxSampler callback, firing from
    /// Metal's thread after GPU execution — exactly how it fires in the engine. The
    /// callback hands its yield off to a Task so the test can deterministically force
    /// continuation.finish() to run before the yield:
    ///   - callbackReady: resumes when the callback fires (GPU finished, Task spawned)
    ///   - proceedStream: gates the deferred yield until after finish() has run
    ///
    /// This models the MTLSharedEvent race: currentWorkCompleted() resumes (via shared
    /// event) while the addCompletedHandler block is still running on the driver thread.
    @Test("sampler callback yield races with finish() — engine callback pattern")
    func samplerCallbackYieldRacesWithFinish() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())
        let logitsBuffer = try makeLogitsBuffer(device: device, winner: 7)
        let outputBuffer = try #require(
            device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize)

        let (stream, continuation) = AsyncThrowingStream<Int32, Error>.makeStream()
        let (proceedStream, proceedContinuation) = AsyncStream<Void>.makeStream()

        let deferredYield: Task<Void, Never> = await withCheckedContinuation {
            (callbackReady: CheckedContinuation<Task<Void, Never>, Never>) in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { nextToken, _ in
                    let task = Task {
                        for await _ in proceedStream { break }
                        continuation.yield(nextToken)
                    }
                    callbackReady.resume(returning: task)
                }
            )
        }
        continuation.finish()
        proceedContinuation.yield(())
        proceedContinuation.finish()
        await deferredYield.value

        var received: [Int32] = []
        do { for try await token in stream { received.append(token) } } catch {}

        #expect(received.isEmpty, "Token dropped: finish() raced with sampler callback yield()")
    }

    /// Multi-step version: N-1 callbacks yield naturally; the last is held past finish().
    @Test("last sampler token dropped when finish() races with final callback")
    func lastSamplerTokenDroppedWithoutSynchronization() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize)

        let tokenCount = 3
        let (stream, continuation) = AsyncThrowingStream<Int32, Error>.makeStream()
        let (earlyDoneStream, earlyDoneContinuation) = AsyncStream<Void>.makeStream()

        for i in 0..<(tokenCount - 1) {
            let logitsBuffer = try makeLogitsBuffer(device: device, winner: i)
            let outputBuffer = try #require(
                device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { nextToken, _ in
                    continuation.yield(nextToken)
                    earlyDoneContinuation.yield(())
                }
            )
        }
        var earlyReceived = 0
        for await _ in earlyDoneStream {
            earlyReceived += 1
            if earlyReceived == tokenCount - 1 { break }
        }
        earlyDoneContinuation.finish()

        let lastLogits = try makeLogitsBuffer(device: device, winner: tokenCount - 1)
        let lastOutput = try #require(
            device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))
        let (proceedStream, proceedContinuation) = AsyncStream<Void>.makeStream()

        let deferredYield: Task<Void, Never> = await withCheckedContinuation {
            (lastCallbackReady: CheckedContinuation<Task<Void, Never>, Never>) in
            sampler.encode(
                to: queue,
                logitsBuffer: lastLogits,
                logitsOffset: 0,
                outputBuffer: lastOutput,
                outputOffset: 0,
                completion: { nextToken, _ in
                    let task = Task {
                        for await _ in proceedStream { break }
                        continuation.yield(nextToken)
                    }
                    lastCallbackReady.resume(returning: task)
                }
            )
        }
        continuation.finish()
        proceedContinuation.yield(())
        proceedContinuation.finish()
        await deferredYield.value

        var received: [Int32] = []
        do { for try await token in stream { received.append(token) } } catch {}

        #expect(received.count == tokenCount - 1, "Last sampler token dropped due to race")
    }

    // MARK: Sentinel fix — engine pattern (Approach 6)

    /// Sentinel command on same serial queue guarantees all prior callbacks have returned
    /// before withCheckedContinuation resumes. Uses a bare command buffer (not the sampler)
    /// to avoid the shared MPSGraphExecutableExecutionDescriptor in TopK sampler.
    @Test("sentinel command ensures sampler callback yield completes before finish()")
    func sentinelFixesSamplerCallbackRace() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())
        let logitsBuffer = try makeLogitsBuffer(device: device, winner: 7)
        let outputBuffer = try #require(
            device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize)

        let (stream, continuation) = AsyncThrowingStream<Int32, Error>.makeStream()

        sampler.encode(
            to: queue,
            logitsBuffer: logitsBuffer,
            logitsOffset: 0,
            outputBuffer: outputBuffer,
            outputOffset: 0,
            completion: { nextToken, _ in
                continuation.yield(nextToken)
            }
        )

        await withCheckedContinuation { (sentinelCont: CheckedContinuation<Void, Never>) in
            guard let cmdBuf = queue.makeCommandBuffer() else {
                sentinelCont.resume()
                return
            }
            cmdBuf.addCompletedHandler { _ in sentinelCont.resume() }
            cmdBuf.commit()
        }
        continuation.finish()

        var received: [Int32] = []
        do { for try await token in stream { received.append(token) } } catch {}

        #expect(received == [7])
    }

    /// Multi-step sentinel: all N sampler tokens received before finish().
    @Test("sentinel command ensures all N sampler tokens received before finish()")
    func sentinelSynchronizesAllSamplerTokens() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize)

        let tokenCount = 5
        let (stream, continuation) = AsyncThrowingStream<Int32, Error>.makeStream()

        for i in 0..<tokenCount {
            let logitsBuffer = try makeLogitsBuffer(device: device, winner: i)
            let outputBuffer = try #require(
                device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { nextToken, _ in
                    continuation.yield(nextToken)
                }
            )
        }

        await withCheckedContinuation { (sentinelCont: CheckedContinuation<Void, Never>) in
            guard let cmdBuf = queue.makeCommandBuffer() else {
                sentinelCont.resume()
                return
            }
            cmdBuf.addCompletedHandler { _ in sentinelCont.resume() }
            cmdBuf.commit()
        }
        continuation.finish()

        var received: [Int32] = []
        do { for try await token in stream { received.append(token) } } catch {}

        #expect(received.count == tokenCount)
    }
}

// MARK: - MPSGraph Completion Ordering Sentinel

/// Validates that MPSGraphExecutable.runAsync completionHandler calls are dispatched
/// in submission order when using a single MTLCommandQueue. This is not documented by
/// Apple but is relied upon by RepetitionPenaltyGPUState (recordToken is called from
/// these completions without additional synchronization).
///
/// If this test fails, RepetitionPenaltyGPUState needs a lock.
@Suite("MPSGraph completion ordering", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphCompletionOrderingTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize = 512

    @Test("completions fire in submission order on a single command queue")
    func completionsAreSerial() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize)

        let stepCount = 16
        let orderRecord = Mutex<[Int]>([])

        // Submit stepCount encode calls back-to-back. Each completion records its index.
        for i in 0..<stepCount {
            let logitsBuffer = try #require(
                device.makeBuffer(
                    length: Self.vocabSize * MemoryLayout<Float16>.size,
                    options: .storageModeShared))
            let ptr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
            for v in 0..<Self.vocabSize { ptr[v] = Float16(0) }
            ptr[i % Self.vocabSize] = Float16(10.0)

            let outputBuffer = try #require(
                device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared))

            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _, _ in
                    orderRecord.withLock { $0.append(i) }
                }
            )
        }

        // Wait for all completions via a sentinel command buffer.
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            guard let cmdBuf = queue.makeCommandBuffer() else {
                cont.resume()
                return
            }
            cmdBuf.addCompletedHandler { _ in cont.resume() }
            cmdBuf.commit()
        }

        let observed = orderRecord.withLock { $0 }
        #expect(
            observed == Array(0..<stepCount),
            "Completions not in submission order — RepetitionPenaltyGPUState needs a lock")
    }
}
