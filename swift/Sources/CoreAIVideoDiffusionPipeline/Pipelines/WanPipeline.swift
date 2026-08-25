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
/// Orchestrates: tokenize -> UMT5 encode -> noise ->
/// denoise loop (flow-match Euler, static shift) -> denormalize -> 3D VAE decode -> frames.
///
/// The transformer computes 3D RoPE internally from the hidden_states shape.
/// Operates on channels-first 5D latents [1, 16, T', H', W'].
public struct WanPipeline: VideoPipeline {
    // MARK: - Components

    let transformer: CoreAIDiffusionModelFunction
    let textEncoder: CoreAIDiffusionModelFunction
    let decoder: CoreAIDiffusionModelFunction
    let tokenizer: any Tokenizer

    // MARK: - Architecture Constants

    let textDim: Int
    let latentChannels: Int

    public let defaultSteps: Int
    public let defaultGuidanceScale: Float
    let schedulerShift: Float

    public var lazyModelLoading: Bool

    private let configDefaultFrameCount: Int

    public var defaultVideoSize: (width: Int, height: Int) { (832, 480) }
    public var defaultFrameCount: Int { configDefaultFrameCount }

    // MARK: - Compression Ratios

    private static let spatialCompression = 8
    private static let temporalCompression = 4

    private static let textSeqLen = 226

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
        textDim: Int = 4096,
        latentChannels: Int = 16,
        defaultSteps: Int = 50,
        defaultGuidanceScale: Float = 5.0,
        schedulerShift: Float = 3.0,
        defaultFrameCount: Int = 81,
        lazyModelLoading: Bool = true
    ) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.textDim = textDim
        self.latentChannels = latentChannels
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.schedulerShift = schedulerShift
        self.configDefaultFrameCount = defaultFrameCount
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

        let textDim = diffusion["text_dim"] as? Int ?? 4096
        let latentChannels = diffusion["z_dim"] as? Int ?? 16
        let defaultSteps = diffusion["default_steps"] as? Int ?? 50
        let defaultGuidanceScale = (diffusion["default_guidance_scale"] as? NSNumber)?.floatValue ?? 5.0
        let shift = (diffusion["default_shift"] as? NSNumber)?.floatValue ?? 3.0
        let defaultFrameCount = diffusion["default_num_frames"] as? Int ?? 81

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
            textDim: textDim,
            latentChannels: latentChannels,
            defaultSteps: defaultSteps,
            defaultGuidanceScale: defaultGuidanceScale,
            schedulerShift: shift,
            defaultFrameCount: defaultFrameCount,
            lazyModelLoading: lazyModelLoading
        )
    }

    // MARK: - VideoPipeline

    public func generateVideo(
        configuration: VideoConfiguration,
        progressHandler: @Sendable (VideoProgress) -> Bool
    ) async throws -> VideoGenerationResult {
        // Validate configuration
        if configuration.numFrames <= 0 {
            throw WanError.invalidConfiguration("numFrames must be > 0")
        }
        if configuration.stepCount <= 0 {
            throw WanError.invalidConfiguration("stepCount must be > 0")
        }
        if configuration.width % Self.spatialCompression != 0 {
            throw WanError.invalidConfiguration(
                "width (\(configuration.width)) must be a multiple of \(Self.spatialCompression)")
        }
        if configuration.height % Self.spatialCompression != 0 {
            throw WanError.invalidConfiguration(
                "height (\(configuration.height)) must be a multiple of \(Self.spatialCompression)")
        }
        if (configuration.numFrames - 1) % Self.temporalCompression != 0 {
            throw WanError.invalidConfiguration(
                "numFrames (\(configuration.numFrames)) must satisfy (n-1) mod \(Self.temporalCompression) == 0 (e.g. 17, 33, 49, 65, 81)"
            )
        }
        let maxFrames = (Self.vaeTemporalFrames - 1) * Self.temporalCompression + 1
        if configuration.numFrames > maxFrames {
            throw WanError.invalidConfiguration(
                "numFrames (\(configuration.numFrames)) exceeds VAE capacity (\(maxFrames) frames)")
        }

        let width = configuration.width
        let height = configuration.height
        let numFrames = configuration.numFrames
        let steps = configuration.stepCount
        let guidanceScale = configuration.guidanceScale
        let seed = configuration.seed
        let fps = configuration.fps

        let latentFrames = (numFrames - 1) / Self.temporalCompression + 1
        let latentHeight = height / Self.spatialCompression
        let latentWidth = width / Self.spatialCompression
        let latentSize = latentChannels * latentFrames * latentHeight * latentWidth

        guard progressHandler(VideoProgress(step: 0, totalSteps: steps, phase: .encoding)) else {
            throw WanError.cancelled
        }

        // 1. Tokenize and encode prompt
        let promptEmbeddings = try await encodePrompt(configuration.prompt)
        let negativeEmbeddings = try await encodePrompt(configuration.negativePrompt)

        if lazyModelLoading {
            await textEncoder.unloadResources()
        }

        // 2. Generate or load noise
        let noise: [Float]
        if let noisePath = configuration.loadNoisePath {
            noise = try loadNumpy(from: noisePath, expectedCount: latentSize)
        } else {
            noise = generateNoise(count: latentSize, seed: seed)
        }

        // 3. Create scheduler
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps,
            trainStepCount: 1000,
            timeStepShift: schedulerShift,
            mu: nil,
            sigmaMax: 1.0
        )

        // 4. Denoising loop
        let latentShape = [1, latentChannels, latentFrames, latentHeight, latentWidth]
        var latents = noise
        let timeSteps = scheduler.timeSteps

        // Pre-pack constant inputs as NDArrays to avoid re-packing every step.
        let promptNDArray = try await transformer.prepackInput(
            index: 1, data: promptEmbeddings, shape: [1, Self.textSeqLen, textDim])
        let negativeNDArray = try await transformer.prepackInput(
            index: 1, data: negativeEmbeddings, shape: [1, Self.textSeqLen, textDim])

        // Pre-allocate reusable buffer for CFG computation (avoids per-step allocation)
        var cfgResult = [Float](repeating: 0, count: latentSize)

        for (step, timestep) in timeSteps.enumerated() {
            guard progressHandler(VideoProgress(step: step, totalSteps: steps, phase: .denoising)) else {
                throw WanError.cancelled
            }

            let t: Float = Float(timestep)

            // Dynamic CFG: skip unconditional pass for late steps
            let skipUncond: Bool
            if let cutoff = configuration.cfgCutoffFraction {
                skipUncond = Float(step) >= Float(steps) * (1.0 - cutoff)
            } else {
                skipUncond = false
            }

            let noisePred: [Float]
            if skipUncond {
                // Late steps: conditional only
                let cond = try await transformer.run(
                    inputs: [
                        .floats(latents, latentShape),
                        .cached(promptNDArray),
                        .floats([t], [1]),
                    ]
                )
                noisePred = cond
            } else {
                // Sequential CFG: two separate calls
                let noisePredCond = try await transformer.run(
                    inputs: [
                        .floats(latents, latentShape),
                        .cached(promptNDArray),
                        .floats([t], [1]),
                    ]
                )
                let noisePredUncond = try await transformer.run(
                    inputs: [
                        .floats(latents, latentShape),
                        .cached(negativeNDArray),
                        .floats([t], [1]),
                    ]
                )
                let n = vDSP_Length(latentSize)
                vDSP_vsub(noisePredUncond, 1, noisePredCond, 1, &cfgResult, 1, n)
                var scale = guidanceScale
                vDSP_vsma(cfgResult, 1, &scale, noisePredUncond, 1, &cfgResult, 1, n)
                noisePred = cfgResult
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

        guard progressHandler(VideoProgress(step: steps, totalSteps: steps, phase: .decoding)) else {
            throw WanError.cancelled
        }

        // 5. Denormalize latents
        denormalize(
            &latents, channels: latentChannels, frames: latentFrames,
            height: latentHeight, width: latentWidth)

        // 6. VAE decode (full sequence — the Wan 3D VAE uses causal convolutions
        // that require all frames at once for correct state propagation)
        let vaeT = Self.vaeTemporalFrames
        let pixels: [Float]
        if let tileSize = configuration.vaeTileSize {
            pixels = try await tiledVAEDecode(
                latents: latents,
                channels: latentChannels,
                frames: latentFrames,
                height: latentHeight,
                width: latentWidth,
                tileHeight: tileSize,
                tileWidth: tileSize,
                overlap: configuration.vaeTileOverlap
            )
        } else {
            // Pad temporal dimension to match VAE export shape if needed
            let vaeInput: [Float]
            let vaeShape: [Int]
            if latentFrames < vaeT {
                let padded = padLatentTile(
                    tile: latents, channels: latentChannels,
                    actualT: latentFrames, actualH: latentHeight, actualW: latentWidth,
                    targetT: vaeT, targetH: latentHeight, targetW: latentWidth)
                vaeInput = padded
                vaeShape = [1, latentChannels, vaeT, latentHeight, latentWidth]
            } else {
                vaeInput = latents
                vaeShape = latentShape
            }

            let rawPixels = try await decoder.run(floatInputs: [(vaeInput, vaeShape)])

            // Trim to actual output frames if we padded.
            // Layout is [C, T, H, W] — must extract contiguous frames from each channel.
            let outputFrames = (latentFrames - 1) * Self.temporalCompression + 1
            let fullDecodedFrames = (vaeT - 1) * Self.temporalCompression + 1
            if outputFrames < fullDecodedFrames {
                let frameSize = height * width
                var trimmed = [Float](repeating: 0, count: 3 * outputFrames * frameSize)
                for c in 0..<3 {
                    let srcOff = c * fullDecodedFrames * frameSize
                    let dstOff = c * outputFrames * frameSize
                    for i in 0..<(outputFrames * frameSize) {
                        trimmed[dstOff + i] = rawPixels[srcOff + i]
                    }
                }
                pixels = trimmed
            } else {
                pixels = rawPixels
            }
        }

        if lazyModelLoading {
            await decoder.unloadResources()
        }

        guard progressHandler(VideoProgress(step: steps, totalSteps: steps, phase: .assembling)) else {
            throw WanError.cancelled
        }

        // 7. Extract frames
        let expectedPixels = 3 * numFrames * height * width
        guard pixels.count >= expectedPixels else {
            throw WanError.invalidConfiguration(
                "VAE output size mismatch: got \(pixels.count) elements, expected \(expectedPixels)")
        }
        let frames = extractFrames(
            pixels: pixels,
            numFrames: numFrames,
            height: height,
            width: width
        )

        return VideoGenerationResult(frames: frames, fps: fps, seed: seed)
    }

    // MARK: - Text Encoding

    /// UMT5 EOS token ID. Used as fallback when the tokenizer returns empty for "".
    private static let umt5EosTokenId: Int = 1

    private func encodePrompt(_ text: String) async throws -> [Float] {
        var encoded = tokenizer.encode(text: text)

        // UMT5 requires at least an EOS token. The diffusers pipeline encodes ""
        // as [EOS] producing non-zero embeddings at position 0. Some swift-transformers
        // tokenizer configs omit the post-processor that appends EOS, so guard here.
        if encoded.isEmpty {
            encoded = [Self.umt5EosTokenId]
        }

        var inputIds = [Int32](repeating: 0, count: Self.textSeqLen)
        var attentionMask = [Int32](repeating: 0, count: Self.textSeqLen)
        let tokenCount = min(encoded.count, Self.textSeqLen)
        for i in 0..<tokenCount {
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
        for pos in tokenCount..<Self.textSeqLen {
            let offset = pos * embDim
            for d in 0..<embDim {
                embeddings[offset + d] = 0
            }
        }

        return embeddings
    }

    // MARK: - Latent Denormalization

    private func denormalize(
        _ latents: inout [Float], channels: Int, frames: Int,
        height: Int, width: Int
    ) {
        let spatialSize = frames * height * width
        let n = vDSP_Length(spatialSize)
        latents.withUnsafeMutableBufferPointer { buf in
            for c in 0..<channels {
                var std = Self.latentsStd[c]
                var mean = Self.latentsMean[c]
                let ptr = buf.baseAddress! + c * spatialSize
                vDSP_vsmsa(ptr, 1, &std, &mean, ptr, 1, n)
            }
        }
    }

    // MARK: - VAE Temporal Configuration

    /// Number of latent frames the VAE model was exported with.
    /// Wan 2.1 default: 5 latent frames -> 17 output pixel frames.
    static let vaeTemporalFrames = 21

    /// Temporal overlap between VAE chunks in latent frames.
    /// One frame of overlap prevents seam artifacts along the time axis.
    static let vaeTemporalOverlap = 1

    // MARK: - Tiled VAE Decode

    /// Decode latents using spatial and temporal tiling to reduce peak memory.
    ///
    /// Splits the latent tensor `[1, C, T, H, W]` into overlapping spatial tiles
    /// **and** temporal chunks that match the VAE export shape `[1, C, vaeT, tileH, tileW]`.
    /// Each chunk is decoded independently and blended with linear interpolation
    /// at overlap boundaries.
    ///
    /// The VAE was exported with a fixed temporal dimension (`vaeTemporalFrames`),
    /// so we must chunk along time as well as space.
    private func tiledVAEDecode(
        latents: [Float],
        channels: Int,
        frames: Int,
        height: Int,
        width: Int,
        tileHeight: Int,
        tileWidth: Int,
        overlap: Int
    ) async throws -> [Float] {
        let pixelScale = Self.spatialCompression  // 8x
        let outputChannels = 3
        let outputFrames = (frames - 1) * Self.temporalCompression + 1

        let fullPixelH = height * pixelScale
        let fullPixelW = width * pixelScale
        let pixelOverlapH = overlap * pixelScale
        let pixelOverlapW = overlap * pixelScale

        // Temporal chunking in latent space
        let vaeT = Self.vaeTemporalFrames
        let temporalOverlap = Self.vaeTemporalOverlap
        // Adjacent chunks share 1 latent frame, which maps to 1 pixel frame of overlap
        // (chunk N ends at pixel P, chunk N+1 starts at pixel P).
        let pixelTemporalOverlap = 1

        // Allocate output buffer: [C, T_out, H_pixel, W_pixel]
        let outputSize = outputChannels * outputFrames * fullPixelH * fullPixelW
        var output = [Float](repeating: 0, count: outputSize)

        // Compute tile positions
        let tileStartsH = computeTileStarts(totalSize: height, tileSize: tileHeight, overlap: overlap)
        let tileStartsW = computeTileStarts(totalSize: width, tileSize: tileWidth, overlap: overlap)
        let tileStartsT = computeTileStarts(totalSize: frames, tileSize: vaeT, overlap: temporalOverlap)

        // Process each spatio-temporal tile
        for (tChunkIdx, startT) in tileStartsT.enumerated() {
            let actualChunkT = min(vaeT, frames - startT)

            for (tileRow, startH) in tileStartsH.enumerated() {
                for (tileCol, startW) in tileStartsW.enumerated() {
                    let actualTileH = min(tileHeight, height - startH)
                    let actualTileW = min(tileWidth, width - startW)

                    // Extract spatio-temporal sub-tensor
                    var tileLatent = extractLatentSubTensor(
                        latents: latents,
                        channels: channels, frames: frames,
                        height: height, width: width,
                        startT: startT, chunkT: actualChunkT,
                        startH: startH, startW: startW,
                        tileH: actualTileH, tileW: actualTileW
                    )

                    // Pad to the VAE's fixed input shape
                    tileLatent = padLatentTile(
                        tile: tileLatent,
                        channels: channels,
                        actualT: actualChunkT, actualH: actualTileH, actualW: actualTileW,
                        targetT: vaeT, targetH: tileHeight, targetW: tileWidth
                    )

                    // Decode at the VAE's fixed shape
                    let tileShape = [1, channels, vaeT, tileHeight, tileWidth]
                    let tilePixels = try await decoder.run(
                        floatInputs: [(tileLatent, tileShape)]
                    )

                    // Actual decoded pixel dimensions (before padding region)
                    let pixTileT = (actualChunkT - 1) * Self.temporalCompression + 1
                    let pixTileH = actualTileH * pixelScale
                    let pixTileW = actualTileW * pixelScale
                    let pixStartT = (startT == 0) ? 0 : startT * Self.temporalCompression
                    let pixStartH = startH * pixelScale
                    let pixStartW = startW * pixelScale

                    // Full decoded tile strides (padded tile size)
                    let decodedStrideT = (vaeT - 1) * Self.temporalCompression + 1
                    let decodedStrideH = tileHeight * pixelScale
                    let decodedStrideW = tileWidth * pixelScale

                    blendTileIntoOutput(
                        output: &output,
                        tilePixels: tilePixels,
                        outputChannels: outputChannels,
                        outputFrames: outputFrames,
                        fullPixelH: fullPixelH,
                        fullPixelW: fullPixelW,
                        pixStartT: pixStartT,
                        pixStartH: pixStartH,
                        pixStartW: pixStartW,
                        pixTileT: pixTileT,
                        pixTileH: pixTileH,
                        pixTileW: pixTileW,
                        decodedStrideT: decodedStrideT,
                        decodedStrideH: decodedStrideH,
                        decodedStrideW: decodedStrideW,
                        pixelOverlapT: pixelTemporalOverlap,
                        pixelOverlapH: pixelOverlapH,
                        pixelOverlapW: pixelOverlapW,
                        isFirstTemporal: tChunkIdx == 0,
                        isFirstRow: tileRow == 0,
                        isFirstCol: tileCol == 0
                    )
                }
            }
        }

        return output
    }

    // MARK: - Frame Extraction

    private func extractFrames(pixels: [Float], numFrames: Int, height: Int, width: Int) -> [CGImage] {
        // pixels layout: [1, 3, T, H, W] channels-first
        let frameSize = height * width
        let n = vDSP_Length(frameSize)
        var frames: [CGImage] = []
        frames.reserveCapacity(numFrames)

        // Reusable per-channel buffer: (pixel + 1) * 127.5
        var channelBuf = [Float](repeating: 0, count: frameSize)
        var rgbaData = [UInt8](repeating: 255, count: frameSize * 4)

        for t in 0..<numFrames {
            let rOff = 0 * numFrames * frameSize + t * frameSize
            let gOff = 1 * numFrames * frameSize + t * frameSize
            let bOff = 2 * numFrames * frameSize + t * frameSize

            // Interleave R, G, B channels into RGBA using vectorized conversion
            for (cOff, rgbaChannel) in [(rOff, 0), (gOff, 1), (bOff, 2)] {
                // channelBuf = (pixels[cOff..] + 1.0) * 127.5
                var add: Float = 1.0
                var mul: Float = 127.5
                pixels.withUnsafeBufferPointer { buf in
                    vDSP_vsadd(buf.baseAddress! + cOff, 1, &add, &channelBuf, 1, n)
                }
                vDSP_vsmul(channelBuf, 1, &mul, &channelBuf, 1, n)

                // Clamp and convert to UInt8 at stride 4
                for i in 0..<frameSize {
                    rgbaData[i * 4 + rgbaChannel] = UInt8(max(0, min(255, channelBuf[i])))
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

    private func generateNoise(count: Int, seed: UInt32) -> [Float] {
        var rng = TorchRandomSource(seed: seed)
        return rng.normalArray([count])
    }

    private func loadNumpy(from path: String, expectedCount: Int) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try data.withUnsafeBytes { buffer -> [Float] in
            let headerSize = Self.parseNumpyHeaderSize(buffer)
            guard headerSize < buffer.count else {
                throw WanError.invalidConfiguration(
                    "Corrupt .npy file: header size \(headerSize) exceeds file size \(buffer.count)")
            }
            let headerBytes = buffer.bindMemory(to: UInt8.self)
            let headerStr = String(bytes: headerBytes[10..<headerSize], encoding: .ascii) ?? ""
            guard headerStr.contains("'<f4'") || headerStr.contains("'float32'") else {
                throw WanError.invalidConfiguration(
                    "Unsupported .npy dtype (expected float32/<f4): \(headerStr.prefix(60))")
            }
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
    case invalidConfiguration(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let msg):
            return "Invalid Wan pipeline metadata: \(msg)"
        case .invalidConfiguration(let msg):
            return "Invalid video configuration: \(msg)"
        case .cancelled:
            return "Video generation was cancelled"
        }
    }
}
