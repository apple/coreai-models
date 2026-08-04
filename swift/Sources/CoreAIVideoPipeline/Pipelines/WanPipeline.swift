// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers

/// Wan 2.1 text-to-video pipeline using Core AI backend.
///
/// Orchestrates: tokenize -> UMT5 encode -> 3D RoPE compute -> noise ->
/// denoise loop (flow-match Euler, static shift) -> denormalize -> 3D VAE decode -> frames.
///
/// Key design: 3D RoPE embeddings are pre-computed in Swift and passed as model inputs,
/// matching the diffusers WanRotaryPosEmbed exactly. The transformer operates on
/// channels-first 5D latents [1, 16, T', H', W'].
public struct WanPipeline: VideoPipeline {
    // MARK: - Components

    let transformer: CoreAIDiffusionModelFunction
    let textEncoder: CoreAIDiffusionModelFunction
    let decoder: CoreAIDiffusionModelFunction
    let tokenizer: any Tokenizer

    // MARK: - Architecture Constants

    let numAttentionHeads: Int
    let attentionHeadDim: Int
    let headDim: Int
    let textDim: Int
    let latentChannels: Int
    let patchSize: (Int, Int, Int)
    let ropeMaxSeqLen: Int

    public let defaultSteps: Int
    public let defaultGuidanceScale: Float
    let schedulerShift: Float

    public var lazyModelLoading: Bool

    private let configDefaultFrameCount: Int
    private let configDefaultFPS: Int

    public var defaultVideoSize: (width: Int, height: Int) { (832, 480) }
    public var defaultFrameCount: Int { configDefaultFrameCount }

    // MARK: - Compression Ratios

    private static let spatialCompression = 8
    private static let temporalCompression = 4

    private static let textSeqLen = 512

    // MARK: - VAE Normalization Constants (16 channels)

    private static let latentsMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    private static let latentsStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]

    // MARK: - Init

    public init(
        transformer: CoreAIDiffusionModelFunction,
        textEncoder: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        tokenizer: any Tokenizer,
        numAttentionHeads: Int = 12,
        attentionHeadDim: Int = 128,
        textDim: Int = 4096,
        latentChannels: Int = 16,
        patchSize: (Int, Int, Int) = (1, 2, 2),
        ropeMaxSeqLen: Int = 1024,
        defaultSteps: Int = 50,
        defaultGuidanceScale: Float = 5.0,
        schedulerShift: Float = 3.0,
        defaultFrameCount: Int = 81,
        defaultFPS: Int = 16,
        lazyModelLoading: Bool = true
    ) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.headDim = attentionHeadDim
        self.textDim = textDim
        self.latentChannels = latentChannels
        self.patchSize = patchSize
        self.ropeMaxSeqLen = ropeMaxSeqLen
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.schedulerShift = schedulerShift
        self.configDefaultFrameCount = defaultFrameCount
        self.configDefaultFPS = defaultFPS
        self.lazyModelLoading = lazyModelLoading
    }

    public init(from url: URL, lazyModelLoading: Bool = true) async throws {
        let metadataURL = url.appendingPathComponent("metadata.json")
        let metadataData = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: metadataData) as? [String: Any],
            let diffusion = json["diffusion"] as? [String: Any]
        else {
            throw WanError.invalidMetadata("metadata.json missing 'diffusion' block")
        }

        let numAttentionHeads = diffusion["num_attention_heads"] as? Int ?? 12
        let attentionHeadDim = diffusion["attention_head_dim"] as? Int ?? 128
        let textDim = diffusion["text_dim"] as? Int ?? 4096
        let latentChannels = diffusion["z_dim"] as? Int ?? 16
        let patchSizeArray = diffusion["patch_size"] as? [Int] ?? [1, 2, 2]
        let defaultSteps = diffusion["default_steps"] as? Int ?? 50
        let defaultGuidanceScale = (diffusion["default_guidance_scale"] as? NSNumber)?.floatValue ?? 5.0
        let shift = (diffusion["default_shift"] as? NSNumber)?.floatValue ?? 3.0
        let defaultFrameCount = diffusion["default_num_frames"] as? Int ?? 81
        let defaultFPS = diffusion["default_fps"] as? Int ?? 16

        let tokenizerURL = url.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)

        self.init(
            transformer: CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent("Transformer.aimodel")
            ),
            textEncoder: CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent("TextEncoder.aimodel")
            ),
            decoder: CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent("VAEDecoder.aimodel")
            ),
            tokenizer: tokenizer,
            numAttentionHeads: numAttentionHeads,
            attentionHeadDim: attentionHeadDim,
            textDim: textDim,
            latentChannels: latentChannels,
            patchSize: (patchSizeArray[0], patchSizeArray[1], patchSizeArray[2]),
            ropeMaxSeqLen: 1024,
            defaultSteps: defaultSteps,
            defaultGuidanceScale: defaultGuidanceScale,
            schedulerShift: shift,
            defaultFrameCount: defaultFrameCount,
            defaultFPS: defaultFPS,
            lazyModelLoading: lazyModelLoading
        )
    }

    // MARK: - VideoPipeline

    public func generateVideo(
        configuration: VideoConfiguration,
        progressHandler: @Sendable (VideoProgress) -> Bool
    ) async throws -> VideoGenerationResult {
        let width = configuration.width
        let height = configuration.height
        let numFrames = configuration.numFrames
        let steps = configuration.stepCount
        let guidanceScale = configuration.guidanceScale
        let seed = UInt64(configuration.seed)
        let fps = configuration.fps

        let latentFrames = (numFrames - 1) / Self.temporalCompression + 1
        let latentHeight = height / Self.spatialCompression
        let latentWidth = width / Self.spatialCompression
        let latentSize = latentChannels * latentFrames * latentHeight * latentWidth

        let _ = progressHandler(VideoProgress(step: 0, totalSteps: steps, phase: .encoding))

        // 1. Tokenize and encode prompt
        let promptEmbeddings = try await encodePrompt(configuration.prompt)
        let negativeEmbeddings = try await encodePrompt(configuration.negativePrompt)

        if lazyModelLoading {
            await textEncoder.unloadResources()
        }

        // 2. Pre-compute 3D RoPE
        let (ropeCos, ropeSin) = computeWanRoPE(
            numFrames: latentFrames, height: latentHeight, width: latentWidth
        )

        // 3. Generate or load noise
        let noise: [Float]
        if let noisePath = configuration.loadNoisePath {
            noise = try loadNumpy(from: noisePath, expectedCount: latentSize)
        } else {
            noise = generateNoise(count: latentSize, seed: seed)
        }

        // 4. Create scheduler
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps,
            trainStepCount: 1000,
            timeStepShift: schedulerShift,
            mu: nil,
            sigmaMax: 1.0
        )

        // 5. Denoising loop
        let latentShape = [1, latentChannels, latentFrames, latentHeight, latentWidth]
        var latents = noise
        let timeSteps = scheduler.timeSteps

        for (step, timestep) in timeSteps.enumerated() {
            let _ = progressHandler(VideoProgress(step: step, totalSteps: steps, phase: .denoising))

            let t: Float = Float(timestep)

            // Conditional prediction
            let noisePredCond = try await transformer.run(
                floatInputs: [
                    (latents, latentShape),
                    (promptEmbeddings, [1, Self.textSeqLen, textDim]),
                    ([t], [1]),
                    (ropeCos, ropeShape(latentFrames: latentFrames, height: latentHeight, width: latentWidth)),
                    (ropeSin, ropeShape(latentFrames: latentFrames, height: latentHeight, width: latentWidth)),
                ]
            )

            // Unconditional prediction
            let noisePredUncond = try await transformer.run(
                floatInputs: [
                    (latents, latentShape),
                    (negativeEmbeddings, [1, Self.textSeqLen, textDim]),
                    ([t], [1]),
                    (ropeCos, ropeShape(latentFrames: latentFrames, height: latentHeight, width: latentWidth)),
                    (ropeSin, ropeShape(latentFrames: latentFrames, height: latentHeight, width: latentWidth)),
                ]
            )

            // CFG: noise = uncond + scale * (cond - uncond)
            var noisePred = [Float](repeating: 0, count: latentSize)
            for i in 0 ..< latentSize {
                noisePred[i] = noisePredUncond[i] + guidanceScale * (noisePredCond[i] - noisePredUncond[i])
            }

            // Euler step
            latents = scheduler.step(output: noisePred, timeStep: timestep, sample: latents)

            if let dumpDir = configuration.dumpDirectory {
                try dumpIntermediate(latents, name: "step\(step)_output_latent", shape: latentShape, to: dumpDir)
            }
        }

        if lazyModelLoading {
            await transformer.unloadResources()
        }

        let _ = progressHandler(VideoProgress(step: steps, totalSteps: steps, phase: .decoding))

        // 6. Denormalize latents
        latents = denormalize(latents, channels: latentChannels, frames: latentFrames,
                              height: latentHeight, width: latentWidth)

        // 7. VAE decode
        let pixels = try await decoder.run(
            floatInputs: [(latents, latentShape)]
        )

        if lazyModelLoading {
            await decoder.unloadResources()
        }

        // 8. Extract frames
        let frames = extractFrames(
            pixels: pixels,
            numFrames: numFrames,
            height: height,
            width: width
        )

        return VideoGenerationResult(frames: frames, fps: fps, seed: seed)
    }

    // MARK: - Text Encoding

    private func encodePrompt(_ text: String) async throws -> [Float] {
        let encoded = tokenizer.encode(text: text)

        var inputIds = [Int32](repeating: 0, count: Self.textSeqLen)
        var attentionMask = [Int32](repeating: 0, count: Self.textSeqLen)
        let tokenCount = min(encoded.count, Self.textSeqLen)
        for i in 0 ..< tokenCount {
            inputIds[i] = Int32(encoded[i])
            attentionMask[i] = 1
        }

        let rawEmbeddings = try await textEncoder.run(
            intInputs: [
                (inputIds, [1, Self.textSeqLen]),
                (attentionMask, [1, Self.textSeqLen]),
            ]
        )

        // Zero-pad after actual sequence length (matching diffusers encode_prompt)
        let embDim = textDim
        var embeddings = rawEmbeddings
        for pos in tokenCount ..< Self.textSeqLen {
            let offset = pos * embDim
            for d in 0 ..< embDim {
                embeddings[offset + d] = 0
            }
        }

        return embeddings
    }

    // MARK: - 3D RoPE Pre-computation

    func computeWanRoPE(numFrames: Int, height: Int, width: Int) -> (cos: [Float], sin: [Float]) {
        let ppf = numFrames / patchSize.0
        let pph = height / patchSize.1
        let ppw = width / patchSize.2
        let seqLen = ppf * pph * ppw

        let hDim = 2 * (headDim / 6)  // 42
        let wDim = hDim                 // 42
        let tDim = headDim - hDim - wDim  // 44
        let theta: Float = 10000.0

        // Compute 1D rotary embeddings for each axis
        let (tCos, tSin) = compute1DRotaryEmbed(dim: tDim, maxPos: ropeMaxSeqLen, theta: theta)
        let (hCos, hSin) = compute1DRotaryEmbed(dim: hDim, maxPos: ropeMaxSeqLen, theta: theta)
        let (wCos, wSin) = compute1DRotaryEmbed(dim: wDim, maxPos: ropeMaxSeqLen, theta: theta)

        // Broadcast over 3D grid and concatenate
        var cos = [Float](repeating: 0, count: seqLen * headDim)
        var sin = [Float](repeating: 0, count: seqLen * headDim)

        for f in 0 ..< ppf {
            for h in 0 ..< pph {
                for w in 0 ..< ppw {
                    let seqIdx = f * pph * ppw + h * ppw + w
                    let baseOut = seqIdx * headDim
                    // t-axis: dims [0, tDim)
                    for d in 0 ..< tDim {
                        cos[baseOut + d] = tCos[f * tDim + d]
                        sin[baseOut + d] = tSin[f * tDim + d]
                    }
                    // h-axis: dims [tDim, tDim+hDim)
                    for d in 0 ..< hDim {
                        cos[baseOut + tDim + d] = hCos[h * hDim + d]
                        sin[baseOut + tDim + d] = hSin[h * hDim + d]
                    }
                    // w-axis: dims [tDim+hDim, headDim)
                    for d in 0 ..< wDim {
                        cos[baseOut + tDim + hDim + d] = wCos[w * wDim + d]
                        sin[baseOut + tDim + hDim + d] = wSin[w * wDim + d]
                    }
                }
            }
        }

        return (cos, sin)
    }

    /// Replicate diffusers get_1d_rotary_pos_embed with use_real=True, repeat_interleave_real=True.
    private func compute1DRotaryEmbed(dim: Int, maxPos: Int, theta: Float) -> (cos: [Float], sin: [Float]) {
        let halfDim = dim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for i in 0 ..< halfDim {
            freqs[i] = 1.0 / powf(theta, Float(2 * i) / Float(dim))
        }

        var cos = [Float](repeating: 0, count: maxPos * dim)
        var sin = [Float](repeating: 0, count: maxPos * dim)
        for pos in 0 ..< maxPos {
            for i in 0 ..< halfDim {
                let angle = Float(pos) * freqs[i]
                let c = cosf(angle)
                let s = sinf(angle)
                // repeat_interleave: [c0, c0, c1, c1, ...]
                cos[pos * dim + 2 * i] = c
                cos[pos * dim + 2 * i + 1] = c
                sin[pos * dim + 2 * i] = s
                sin[pos * dim + 2 * i + 1] = s
            }
        }
        return (cos, sin)
    }

    private func ropeShape(latentFrames: Int, height: Int, width: Int) -> [Int] {
        let ppf = latentFrames / patchSize.0
        let pph = height / patchSize.1
        let ppw = width / patchSize.2
        return [1, ppf * pph * ppw, 1, headDim]
    }

    // MARK: - Latent Denormalization

    private func denormalize(_ latents: [Float], channels: Int, frames: Int,
                             height: Int, width: Int) -> [Float]
    {
        var result = latents
        let spatialSize = frames * height * width
        for c in 0 ..< channels {
            let mean = Self.latentsMean[c]
            let std = Self.latentsStd[c]
            let offset = c * spatialSize
            for i in 0 ..< spatialSize {
                result[offset + i] = result[offset + i] * std + mean
            }
        }
        return result
    }

    // MARK: - Frame Extraction

    private func extractFrames(pixels: [Float], numFrames: Int, height: Int, width: Int) -> [CGImage] {
        // pixels layout: [1, 3, T, H, W] channels-first
        let frameSize = height * width
        var frames: [CGImage] = []

        for t in 0 ..< numFrames {
            var rgbaData = [UInt8](repeating: 255, count: height * width * 4)
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let srcIdx = y * width + x
                    let rSrc = 0 * numFrames * frameSize + t * frameSize + srcIdx
                    let gSrc = 1 * numFrames * frameSize + t * frameSize + srcIdx
                    let bSrc = 2 * numFrames * frameSize + t * frameSize + srcIdx

                    let r = UInt8(max(0, min(255, ((pixels[rSrc] + 1.0) / 2.0) * 255.0)))
                    let g = UInt8(max(0, min(255, ((pixels[gSrc] + 1.0) / 2.0) * 255.0)))
                    let b = UInt8(max(0, min(255, ((pixels[bSrc] + 1.0) / 2.0) * 255.0)))

                    let dstIdx = (y * width + x) * 4
                    rgbaData[dstIdx] = r
                    rgbaData[dstIdx + 1] = g
                    rgbaData[dstIdx + 2] = b
                }
            }

            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
            if let provider = CGDataProvider(data: Data(rgbaData) as CFData),
                let image = CGImage(
                    width: width, height: height,
                    bitsPerComponent: 8, bitsPerPixel: 32,
                    bytesPerRow: width * 4,
                    space: colorSpace,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                    provider: provider,
                    decode: nil, shouldInterpolate: false,
                    intent: .defaultIntent
                )
            {
                frames.append(image)
            }
        }
        return frames
    }

    // MARK: - Utilities

    private func generateNoise(count: Int, seed: UInt64) -> [Float] {
        var noise = [Float](repeating: 0, count: count)
        // Box-Muller transform with deterministic seed
        var state = seed
        for i in stride(from: 0, to: count - 1, by: 2) {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u1 = Float(state >> 33) / Float(1 << 31)
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float(state >> 33) / Float(1 << 31)
            let r = sqrtf(-2.0 * logf(max(u1, 1e-10)))
            let theta = 2.0 * Float.pi * u2
            noise[i] = r * cosf(theta)
            if i + 1 < count {
                noise[i + 1] = r * sinf(theta)
            }
        }
        return noise
    }

    private func loadNumpy(from path: String, expectedCount: Int) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { buffer -> [Float] in
            let headerSize = Self.parseNumpyHeaderSize(buffer)
            let floatPtr = buffer.baseAddress!.advanced(by: headerSize).assumingMemoryBound(to: Float.self)
            let count = (buffer.count - headerSize) / MemoryLayout<Float>.size
            return Array(UnsafeBufferPointer(start: floatPtr, count: min(count, expectedCount)))
        }
    }

    private static func parseNumpyHeaderSize(_ buffer: UnsafeRawBufferPointer) -> Int {
        guard buffer.count >= 10 else { return 128 }
        let headerLen = Int(buffer.load(fromByteOffset: 8, as: UInt16.self))
        return 10 + headerLen
    }

    private func dumpIntermediate(_ data: [Float], name: String, shape: [Int], to directory: String) throws {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).npy")
        try writeNumpy(data: data, shape: shape, to: url)
    }

    private func writeNumpy(data: [Float], shape: [Int], to url: URL) throws {
        let shapeStr = shape.map(String.init).joined(separator: ", ")
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shapeStr)), }"
        let paddedLen = ((header.count + 10 + 63) / 64) * 64
        let padding = paddedLen - header.count - 10

        var out = Data()
        out.append(contentsOf: [0x93] + "NUMPY".utf8 + [1, 0])
        var headerLen = UInt16(header.count + padding)
        out.append(Data(bytes: &headerLen, count: 2))
        out.append(header.data(using: .ascii)!)
        out.append(contentsOf: [UInt8](repeating: 0x20, count: padding - 1) + [0x0A])
        data.withUnsafeBufferPointer { buf in
            out.append(contentsOf: UnsafeRawBufferPointer(buf))
        }
        try out.write(to: url)
    }
}

// MARK: - Errors

public enum WanError: Error, LocalizedError {
    case invalidMetadata(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let msg):
            return "Invalid Wan pipeline metadata: \(msg)"
        }
    }
}
