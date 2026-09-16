// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreAIVideoSegmenter
import CoreGraphics
import Foundation
import ImageIO

/// Per-frame detect-and-track output from a reference implementation, to compare this port
/// against. Any producer that writes the layout below will do.
///
/// Layout, one `.npy` per array so the files stay readable from either side:
///
/// ```
/// <dir>/manifest.json            video path, prompts, frame indices, video size
/// <dir>/frames/frame_0000.png    the exact frames PyTorch saw, lossless
/// <dir>/frame_0000_ids.npy       int32  [N]
/// <dir>/frame_0000_scores.npy    float32[N]
/// <dir>/frame_0000_boxes.npy     float32[N, 4]   xyxy, top-left origin
/// <dir>/frame_0000_masks.npy     uint8  [N, ceil(H*W/8)]  numpy.packbits, MSB first
/// ```
///
/// Masks are bit-packed: at video resolution a boolean array carries the same information
/// eight times over, and a 51-frame 1080p clip with five tracks is 65 MB packed against
/// 520 MB raw.
///
/// The PNGs matter. AVFoundation and PyAV decode the same file to different pixels: about
/// 0.65 code values apart on average with the colour space tagged, 3.7 when the container
/// leaves it unset. That gap belongs to the two decoders, so parity reads the dumped frames.
struct ParityReference {
    struct Object {
        let id: Int
        let score: Float
        let box: CGRect
        let mask: MaskBitset
    }

    private let directory: URL
    private let manifest: Manifest

    let videoURL: URL
    var prompts: [String] { manifest.prompts }
    var frameIndices: [Int] { manifest.frames }
    var frameCount: Int { manifest.frames.count }

    private struct Manifest: Decodable {
        let video: String
        let prompts: [String]
        let frames: [Int]
        let width: Int
        let height: Int
        let frameImages: String?

        enum CodingKeys: String, CodingKey {
            case video, prompts, frames, width, height
            case frameImages = "frame_images"
        }
    }

    init(directory: URL) throws {
        self.directory = directory
        let manifestURL = directory.appending(path: "manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw VideoSegmentationError.parityReferenceInvalid(
                "no manifest.json at \(manifestURL.path). See `ParityReference` for the "
                    + "layout a reference directory needs.")
        }
        do {
            self.manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw VideoSegmentationError.parityReferenceInvalid(
                "manifest.json is malformed: \(error)")
        }
        // The manifest records the clip it was produced from; resolve a relative path
        // against the reference directory so the pair can be moved together. Tested on the
        // recorded string, since `URL(fileURLWithPath:)` resolves against the working
        // directory and always reports an absolute path.
        self.videoURL =
            manifest.video.hasPrefix("/")
            ? URL(fileURLWithPath: manifest.video)
            : directory.appending(path: manifest.video)
        guard FileManager.default.fileExists(atPath: videoURL.path) else {
            throw VideoSegmentationError.parityReferenceInvalid(
                "manifest.json points at \(videoURL.path), which does not exist.")
        }
    }

    func objects(at frame: Int) throws -> [Object] {
        let prefix = String(format: "frame_%04d", frame)
        let ids = try load(prefix, "ids").asInt32()
        guard !ids.isEmpty else { return [] }

        let scores = try load(prefix, "scores").asFloat()
        let boxesArray = try load(prefix, "boxes")
        let boxes = boxesArray.asFloat()
        let masksArray = try load(prefix, "masks")
        let packed = try masksArray.asUInt8()

        guard scores.count == ids.count, boxes.count == ids.count * 4 else {
            throw VideoSegmentationError.parityReferenceInvalid(
                "\(prefix): \(ids.count) ids but \(scores.count) scores and "
                    + "\(boxes.count / 4) boxes.")
        }
        let bytesPerMask = (manifest.width * manifest.height + 7) / 8
        guard packed.count == ids.count * bytesPerMask else {
            throw VideoSegmentationError.parityReferenceInvalid(
                "\(prefix): expected \(ids.count * bytesPerMask) packed mask bytes for "
                    + "\(manifest.width)×\(manifest.height), got \(packed.count).")
        }

        return ids.indices.map { index in
            Object(
                id: Int(ids[index]),
                score: scores[index],
                // xyxy to a rect. `masks_to_boxes` reports inclusive extremes, so the
                // width here is `x1 - x0`, matching `MaskBitset.boundingBox`.
                box: CGRect(
                    x: CGFloat(boxes[index * 4]),
                    y: CGFloat(boxes[index * 4 + 1]),
                    width: CGFloat(boxes[index * 4 + 2] - boxes[index * 4]),
                    height: CGFloat(boxes[index * 4 + 3] - boxes[index * 4 + 1])),
                mask: MaskBitset(
                    packedBits: Array(
                        packed[(index * bytesPerMask)..<((index + 1) * bytesPerMask)]),
                    width: manifest.width, height: manifest.height))
        }
    }

    /// Mask logits at the model's own resolution, ordered like `objects(at:)`.
    ///
    /// Present only when the dump included them. They separate a tracker difference from an
    /// upsampling difference, which the final binary masks conflate.
    func lowResolutionMasks(at frame: Int) -> (values: [Float], side: Int)? {
        let url = directory.appending(path: String(format: "frame_%04d_lowres.npy", frame))
        guard let array = try? NpyArray.load(url), array.shape.count == 3 else { return nil }
        return (array.asFloat(), array.shape[1])
    }

    private func load(_ prefix: String, _ name: String) throws -> NpyArray {
        try NpyArray.load(directory.appending(path: "\(prefix)_\(name).npy"))
    }

    /// The exact frames PyTorch decoded, when the dump included them.
    ///
    /// `nil` sends the caller back to decoding ``videoURL``, which reintroduces the decoder
    /// difference; the CLI reports that when it happens.
    func decodedFrames() throws -> [CGImage]? {
        guard let folder = manifest.frameImages else { return nil }
        let directory = self.directory.appending(path: folder)
        var images: [CGImage] = []
        images.reserveCapacity(frameCount)
        for frame in manifest.frames {
            let url = directory.appending(path: String(format: "frame_%04d.png", frame))
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                throw VideoSegmentationError.parityReferenceInvalid(
                    "manifest.json declares frame images but \(url.lastPathComponent) is "
                        + "missing or unreadable.")
            }
            images.append(image)
        }
        return images
    }
}
