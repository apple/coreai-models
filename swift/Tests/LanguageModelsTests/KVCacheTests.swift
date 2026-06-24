// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import MetalPerformanceShaders
import TestUtilities
import Testing

@testable import CoreAILanguageModels

// MARK: - KV Cache Factory Tests

/// Tests for KVCacheFactory utility methods - pure logic, no model required.
@Suite("KVCacheFactory Utilities")
struct KVCacheFactoryTests {
    @Test("detectSequenceDim returns 3 for 5D tensors")
    func detectSequenceDim5D() {
        // KV cache shape [L, B, H, S, D] (5D with layers) → seqDim = 3
        let shape5D = [32, 1, 8, 256, 64]  // 32 layers, batch 1, 8 heads, 256 seq, 64 dim
        #expect(KVCacheFactory.detectSequenceDim(shape: shape5D) == 3)
    }

    @Test("detectSequenceDim returns 2 for 4D tensors")
    func detectSequenceDim4D() {
        // KV cache shape [B, H, S, D] (4D per-layer) → seqDim = 2
        let shape4D = [1, 8, 256, 64]  // batch 1, 8 heads, 256 seq, 64 dim
        #expect(KVCacheFactory.detectSequenceDim(shape: shape4D) == 2)
    }

    @Test("describeKVCacheStructure formats 5D correctly")
    func describe5DShape() {
        let shape = [32, 1, 8, 2048, 64]
        let desc = KVCacheFactory.describeKVCacheStructure(shape: shape)
        #expect(desc.contains("32 layers"))
        #expect(desc.contains("1 batch"))
        #expect(desc.contains("8 heads"))
        #expect(desc.contains("2048 context"))
        #expect(desc.contains("64 head_dim"))
    }

    @Test("describeKVCacheStructure formats 4D correctly")
    func describe4DShape() {
        let shape = [1, 8, 512, 64]
        let desc = KVCacheFactory.describeKVCacheStructure(shape: shape)
        #expect(desc.contains("batch=1"))
        #expect(desc.contains("heads=8"))
        #expect(desc.contains("context=512"))
        #expect(desc.contains("head_dim=64"))
    }

    @Test("describeKVCacheStructure handles 2D and 3D")
    func describe2D3DShapes() {
        let shape2D = [16, 256]
        let desc2D = KVCacheFactory.describeKVCacheStructure(shape: shape2D)
        #expect(desc2D.contains("batch=16"))
        #expect(desc2D.contains("features=256"))

        let shape3D = [16, 128, 256]
        let desc3D = KVCacheFactory.describeKVCacheStructure(shape: shape3D)
        #expect(desc3D.contains("batch=16"))
        #expect(desc3D.contains("context=128"))
        #expect(desc3D.contains("features=256"))
    }
}

// MARK: - KV Cache Error Tests

@Suite("KVCacheError")
struct KVCacheErrorTests {
    @Test("capacityExceeded provides helpful message")
    func capacityExceededMessage() {
        let error = KVCacheError.capacityExceeded(needed: 4096, available: 2048)
        let desc = error.errorDescription ?? ""
        #expect(desc.contains("4096"))
        #expect(desc.contains("2048"))
        #expect(desc.contains("growing"))
    }

    @Test("allocationFailed shows byte count")
    func allocationFailedMessage() {
        let error = KVCacheError.allocationFailed(1_073_741_824)  // 1GB
        let desc = error.errorDescription ?? ""
        #expect(desc.contains("1073741824"))
    }

    @Test("unsupportedStrategy shows strategy name")
    func unsupportedStrategyMessage() {
        let error = KVCacheError.unsupportedStrategy("growing requires dynamic KV")
        let desc = error.errorDescription ?? ""
        #expect(desc.contains("growing requires dynamic KV"))
    }
}

// MARK: - KV Cache Stride Calculation Tests

/// Tests for stride calculation logic used in GrowingKVCache.
/// These are pure math tests - no Metal required.
@Suite("KV Cache Stride Calculations")
struct KVCacheStrideTests {
    @Test("Stride calculation for 5D KV cache shape")
    func strideCalculation() {
        // Shape: [L, B, H, S, D] = [4, 1, 8, 256, 64]
        let l = 4
        let b = 1
        let h = 8
        let s = 256
        let d = 64

        // Calculate strides (in elements, not bytes)
        let strides = [
            b * h * s * d,  // L stride: 1 × 8 × 256 × 64 = 131072
            h * s * d,  // B stride: 8 × 256 × 64 = 131072
            s * d,  // H stride: 256 × 64 = 16384
            d,  // S stride: 64
            1,  // D stride: 1 (contiguous)
        ]

        #expect(strides[0] == 131072, "L stride")
        #expect(strides[1] == 131072, "B stride")
        #expect(strides[2] == 16384, "H stride")
        #expect(strides[3] == 64, "S stride")
        #expect(strides[4] == 1, "D stride")

        // Total elements = L × B × H × S × D
        let totalElements = l * b * h * s * d
        #expect(totalElements == 524288, "Total elements")
    }

    @Test("Stride change when sequence dimension grows")
    func strideChangeOnGrowth() {
        // Original: [4, 1, 8, 256, 64]
        // New: [4, 1, 8, 512, 64]  (sequence doubled)
        let d = 64
        let oldS = 256
        let newS = 512

        let oldHeadStride = oldS * d  // 256 × 64 = 16384
        let newHeadStride = newS * d  // 512 × 64 = 32768

        // This is the key insight: when S grows, H stride changes
        // A simple memcpy would put data at wrong offsets
        #expect(oldHeadStride != newHeadStride, "Head stride must change when S grows")
        #expect(newHeadStride == 2 * oldHeadStride, "New head stride should be 2× old")
    }

    @Test("All strides change proportionally when S doubles")
    func allStridesChangeOnGrowth() {
        // Shape: [L, B, H, S, D] = [32, 1, 8, S, 64]
        let b = 1
        let h = 8
        let d = 64
        let oldS = 256
        let newS = 512

        // Old strides
        let oldStrides = [
            b * h * oldS * d,  // L stride
            h * oldS * d,  // B stride
            oldS * d,  // H stride
            d,  // S stride (unchanged)
            1,  // D stride (unchanged)
        ]

        // New strides
        let newStrides = [
            b * h * newS * d,  // L stride
            h * newS * d,  // B stride
            newS * d,  // H stride
            d,  // S stride
            1,  // D stride
        ]

        // L, B, H strides should double when S doubles
        #expect(newStrides[0] == 2 * oldStrides[0], "L stride doubles")
        #expect(newStrides[1] == 2 * oldStrides[1], "B stride doubles")
        #expect(newStrides[2] == 2 * oldStrides[2], "H stride doubles")
        // S and D strides unchanged
        #expect(newStrides[3] == oldStrides[3], "S stride unchanged")
        #expect(newStrides[4] == oldStrides[4], "D stride unchanged")
    }

    @Test("Buffer size calculation for growth")
    func bufferSizeGrowth() {
        let l = 32
        let b = 1
        let h = 8
        let d = 64
        let oldS = 256
        let newS = 512
        let bytesPerElement = 2  // Float16

        let oldBytes = l * b * h * oldS * d * bytesPerElement
        let newBytes = l * b * h * newS * d * bytesPerElement

        // 32 × 1 × 8 × 256 × 64 × 2 = 8,388,608 bytes = 8MB
        // 32 × 1 × 8 × 512 × 64 × 2 = 16,777,216 bytes = 16MB
        #expect(oldBytes == 8_388_608, "Old buffer: 8MB")
        #expect(newBytes == 16_777_216, "New buffer: 16MB")
        #expect(newBytes == 2 * oldBytes, "New buffer 2× old")
    }
}

// MARK: - Metal Blit Copy Tests

/// Tests for the actual KV cache growth mechanism using Metal blit encoder.
/// These verify the per-head strided copy approach that works correctly.
@Suite("Metal Blit Copy for KV Cache Growth")
struct MetalBlitCopyTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    /// Helper: Calculate linear index from multi-dimensional indices using given strides.
    private func linearIndex(_ indices: [Int], strides: [Int]) -> Int {
        zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }
    }

    @Test("Per-head blit copy preserves data during KV cache growth")
    func perHeadBlitPreservesData() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())

        // KV cache shape: [L, B, H, S, D] = [2, 1, 4, 8, 16] → [2, 1, 4, 16, 16]
        let l = 2
        let b = 1
        let h = 4
        let oldS = 8
        let newS = 16
        let d = 16
        let bytesPerElement = 2  // Float16

        let oldTotalElements = l * b * h * oldS * d
        let newTotalElements = l * b * h * newS * d

        // Create source and destination buffers
        let srcBuffer = try #require(
            device.makeBuffer(length: oldTotalElements * bytesPerElement, options: .storageModeShared))
        let dstBuffer = try #require(
            device.makeBuffer(length: newTotalElements * bytesPerElement, options: .storageModeShared))

        // Fill source with unique values encoding position
        let srcPtr = srcBuffer.contents().assumingMemoryBound(to: Float16.self)
        let oldStrides = [b * h * oldS * d, h * oldS * d, oldS * d, d, 1]

        // Use scaled values that stay within Float16's exact integer range (0-2048)
        // Max value: 1*1000 + 3*100 + 7*10 + 15 = 1385 (safe)
        for li in 0..<l {
            for hi in 0..<h {
                for si in 0..<oldS {
                    for di in 0..<d {
                        let idx = linearIndex([li, 0, hi, si, di], strides: oldStrides)
                        srcPtr[idx] = Float16(li * 1000 + hi * 100 + si * 10 + di)
                    }
                }
            }
        }

        // Clear destination with sentinel
        let dstPtr = dstBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<newTotalElements {
            dstPtr[i] = Float16(-1)
        }

        // Perform per-head blit copy (same approach as GrowingKVCache.ensureCapacity)
        let cmd = try #require(queue.makeCommandBuffer())
        let blit = try #require(cmd.makeBlitCommandEncoder())

        let newStrides = [b * h * newS * d, h * newS * d, newS * d, d, 1]

        for li in 0..<l {
            for hi in 0..<h {
                // Calculate offsets for this head
                let oldHeadOffset = linearIndex([li, 0, hi, 0, 0], strides: oldStrides) * bytesPerElement
                let newHeadOffset = linearIndex([li, 0, hi, 0, 0], strides: newStrides) * bytesPerElement

                // Size = S * D elements (one head's worth of KV cache)
                let copySize = oldS * d * bytesPerElement

                blit.copy(
                    from: srcBuffer,
                    sourceOffset: oldHeadOffset,
                    to: dstBuffer,
                    destinationOffset: newHeadOffset,
                    size: copySize
                )
            }
        }

        blit.endEncoding()
        cmd.commit()
        await cmd.completed()

        // Verify data preservation
        var errors = 0
        for li in 0..<l {
            for hi in 0..<h {
                for si in 0..<oldS {
                    for di in 0..<d {
                        let newIdx = linearIndex([li, 0, hi, si, di], strides: newStrides)
                        // match formula used when writing: l*1000 + h*100 + s*10 + d
                        let expectedValue = Float(li * 1000 + hi * 100 + si * 10 + di)
                        let actualValue = Float(dstPtr[newIdx])
                        if abs(actualValue - expectedValue) > 0.5 {
                            if errors < 10 {
                                print(
                                    "Mismatch at [\(li),0,\(hi),\(si),\(di)]: expected \(expectedValue), got \(actualValue)"
                                )
                            }
                            errors += 1
                        }
                    }
                }
            }
        }

        #expect(errors == 0, "Found \(errors) data preservation errors")

        // Verify padding region (s >= oldS) is untouched
        for li in 0..<l {
            for hi in 0..<h {
                for si in oldS..<newS {
                    let paddingIdx = linearIndex([li, 0, hi, si, 0], strides: newStrides)
                    let paddingValue = Float(dstPtr[paddingIdx])
                    if paddingValue != -1.0 && errors < 10 {
                        print("Padding at [\(li),0,\(hi),\(si),0] should be -1, got \(paddingValue)")
                    }
                    #expect(paddingValue == -1.0)
                }
            }
        }
    }

    @Test("Per-head blit copy handles realistic KV cache dimensions")
    func realisticKVCacheCopy() async throws {
        let device = try #require(Self.device)
        let queue = try #require(device.makeCommandQueue())

        // Realistic dimensions: [L=32, B=1, H=8, S=256, D=64] → S=512
        let l = 32
        let b = 1
        let h = 8
        let oldS = 256
        let newS = 512
        let d = 64
        let bytesPerElement = 2

        let oldTotalElements = l * b * h * oldS * d
        let newTotalElements = l * b * h * newS * d

        let srcBuffer = try #require(
            device.makeBuffer(length: oldTotalElements * bytesPerElement, options: .storageModeShared))
        let dstBuffer = try #require(
            device.makeBuffer(length: newTotalElements * bytesPerElement, options: .storageModeShared))

        // Just fill with sequential values for speed
        let srcPtr = srcBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<oldTotalElements {
            srcPtr[i] = Float16(Float(i % 1000))
        }

        let oldStrides = [b * h * oldS * d, h * oldS * d, oldS * d, d, 1]
        let newStrides = [b * h * newS * d, h * newS * d, newS * d, d, 1]

        // Measure copy time
        let start = SuspendingClock().now

        let cmd = try #require(queue.makeCommandBuffer())
        let blit = try #require(cmd.makeBlitCommandEncoder())

        for li in 0..<l {
            for hi in 0..<h {
                let oldHeadOffset = (li * oldStrides[0] + hi * oldStrides[2]) * bytesPerElement
                let newHeadOffset = (li * newStrides[0] + hi * newStrides[2]) * bytesPerElement
                let copySize = oldS * d * bytesPerElement

                blit.copy(
                    from: srcBuffer,
                    sourceOffset: oldHeadOffset,
                    to: dstBuffer,
                    destinationOffset: newHeadOffset,
                    size: copySize
                )
            }
        }

        blit.endEncoding()
        cmd.commit()
        await cmd.completed()

        let elapsed = SuspendingClock().now - start
        let elapsedMs = elapsed.inMilliseconds

        print("Per-head blit copy time: \(String(format: "%.2f", elapsedMs)) ms for \(l * h) heads")

        // Use higher threshold on VM due to virtualization overhead
        let threshold = CIEnvironment.isVM ? 2000.0 : 500.0
        #expect(
            elapsedMs < threshold,
            "Per-head blit copy should complete in < \(threshold)ms, got \(elapsedMs) ms (VM: \(CIEnvironment.isVM))")

        // Verify first and last head data
        let dstPtr = dstBuffer.contents().assumingMemoryBound(to: Float16.self)

        // Check first head [0,0,0,0,0]
        let firstIdx = 0
        #expect(Float(dstPtr[firstIdx]) == Float(srcPtr[0]))

        // Check last head's first element [L-1,0,H-1,0,0]
        let lastOldHeadOffset = ((l - 1) * oldStrides[0] + (h - 1) * oldStrides[2])
        let lastNewHeadOffset = ((l - 1) * newStrides[0] + (h - 1) * newStrides[2])
        #expect(Float(dstPtr[lastNewHeadOffset]) == Float(srcPtr[lastOldHeadOffset]))
    }
}

// MARK: - Extension to expose Duration inMilliseconds

extension Duration {
    var inMilliseconds: Double {
        let (seconds, attoseconds) = self.components
        return Double(seconds) * 1000.0 + Double(attoseconds) / 1_000_000_000_000_000.0
    }
}
