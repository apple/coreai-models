// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Tiled VAE decoding utilities for the Wan pipeline.
///
/// These functions handle the pure-math operations of splitting a 5D latent tensor
/// into overlapping spatial and temporal tiles, and blending decoded pixel tiles
/// back into a full-resolution output.

// MARK: - Tile Grid Computation

/// Compute tile start positions along one axis.
///
/// Tiles advance by `stride = tileSize - overlap` until coverage reaches `totalSize`.
/// The last tile is positioned so that `startPos + tileSize` covers the end.
/// When `tileSize >= totalSize`, a single tile at position 0 is returned.
func computeTileStarts(totalSize: Int, tileSize: Int, overlap: Int) -> [Int] {
    guard tileSize < totalSize else { return [0] }
    precondition(overlap < tileSize, "vaeTileOverlap (\(overlap)) must be less than vaeTileSize (\(tileSize))")
    let stride = tileSize - overlap
    var starts: [Int] = []
    var pos = 0
    while pos < totalSize {
        starts.append(pos)
        if pos + tileSize >= totalSize { break }
        pos += stride
    }
    return starts
}

// MARK: - Tile Extraction

/// Extract a sub-tensor from a 5D latent array with layout `[1, C, T, H, W]`.
///
/// Returns a flat array of shape `[C, chunkT, tileH, tileW]` copied from the
/// specified spatial and temporal region.
func extractLatentSubTensor(
    latents: [Float],
    channels: Int, frames: Int,
    height: Int, width: Int,
    startT: Int, chunkT: Int,
    startH: Int, startW: Int,
    tileH: Int, tileW: Int
) -> [Float] {
    let tileSize = channels * chunkT * tileH * tileW
    var tile = [Float](repeating: 0, count: tileSize)

    for c in 0..<channels {
        for t in 0..<chunkT {
            for h in 0..<tileH {
                let srcOff =
                    c * (frames * height * width)
                    + (startT + t) * (height * width)
                    + (startH + h) * width
                    + startW
                let dstOff =
                    c * (chunkT * tileH * tileW)
                    + t * (tileH * tileW)
                    + h * tileW
                for w in 0..<tileW {
                    tile[dstOff + w] = latents[srcOff + w]
                }
            }
        }
    }

    return tile
}

// MARK: - Tile Padding

/// Pad a tile from `[C, actualT, actualH, actualW]` to `[C, targetT, targetH, targetW]`.
///
/// Extra elements are zero-filled. If the tile is already at the target size, it is
/// returned without copying.
func padLatentTile(
    tile: [Float],
    channels: Int,
    actualT: Int, actualH: Int, actualW: Int,
    targetT: Int, targetH: Int, targetW: Int
) -> [Float] {
    if actualT == targetT && actualH == targetH && actualW == targetW {
        return tile
    }
    var padded = [Float](repeating: 0, count: channels * targetT * targetH * targetW)
    for c in 0..<channels {
        for t in 0..<actualT {
            for h in 0..<actualH {
                let srcOff = c * (actualT * actualH * actualW) + t * (actualH * actualW) + h * actualW
                let dstOff = c * (targetT * targetH * targetW) + t * (targetH * targetW) + h * targetW
                for w in 0..<actualW {
                    padded[dstOff + w] = tile[srcOff + w]
                }
            }
        }
    }
    return padded
}

// MARK: - Blend Weight

/// Compute the blend weight for a position within an overlap region.
///
/// Returns a linear ramp from 0 at the start of the overlap to 1 at the end.
/// If `isFirst` is true (no preceding tile along this axis), weight is always 1.
/// If `overlap` is zero, weight is always 1.
func blendWeight(position: Int, overlap: Int, isFirst: Bool) -> Float {
    guard !isFirst, overlap > 0, position < overlap else { return 1.0 }
    return Float(position) / Float(overlap)
}

// MARK: - Tile Blending

/// Blend a decoded pixel tile into the output buffer with linear interpolation
/// at overlap boundaries in all three dimensions (temporal, height, width).
///
/// Pixel layout: `[1, C, T, H, W]` channels-first.
///
/// - Parameters:
///   - output: Mutable output buffer, layout `[C, outputFrames, fullPixelH, fullPixelW]`.
///   - tilePixels: Decoded pixel data for this tile, layout `[C, decodedStrideT, decodedStrideH, decodedStrideW]`.
///   - pixTileT/H/W: Actual number of pixel frames/rows/cols to use from the decoded tile
///     (may be smaller than decoded strides for edge tiles with padding).
///   - decodedStrideT/H/W: Full decoded tile dimensions, used for indexing into `tilePixels`.
///   - pixelOverlapT/H/W: Overlap size in pixel space for blending along each axis.
///   - isFirstTemporal/Row/Col: Whether this is the first tile along each axis (no blending needed on the leading edge).
func blendTileIntoOutput(
    output: inout [Float],
    tilePixels: [Float],
    outputChannels: Int,
    outputFrames: Int,
    fullPixelH: Int,
    fullPixelW: Int,
    pixStartT: Int,
    pixStartH: Int,
    pixStartW: Int,
    pixTileT: Int,
    pixTileH: Int,
    pixTileW: Int,
    decodedStrideT: Int,
    decodedStrideH: Int,
    decodedStrideW: Int,
    pixelOverlapT: Int,
    pixelOverlapH: Int,
    pixelOverlapW: Int,
    isFirstTemporal: Bool,
    isFirstRow: Bool,
    isFirstCol: Bool
) {
    for c in 0..<outputChannels {
        for t in 0..<pixTileT {
            let globalT = pixStartT + t
            guard globalT < outputFrames else { continue }

            for h in 0..<pixTileH {
                let globalH = pixStartH + h
                guard globalH < fullPixelH else { continue }

                for w in 0..<pixTileW {
                    let globalW = pixStartW + w
                    guard globalW < fullPixelW else { continue }

                    let tileIdx =
                        c * (decodedStrideT * decodedStrideH * decodedStrideW)
                        + t * (decodedStrideH * decodedStrideW)
                        + h * decodedStrideW
                        + w
                    let outIdx =
                        c * (outputFrames * fullPixelH * fullPixelW)
                        + globalT * (fullPixelH * fullPixelW)
                        + globalH * fullPixelW
                        + globalW

                    let tileVal = tilePixels[tileIdx]

                    let weightT = blendWeight(position: t, overlap: pixelOverlapT, isFirst: isFirstTemporal)
                    let weightH = blendWeight(position: h, overlap: pixelOverlapH, isFirst: isFirstRow)
                    let weightW = blendWeight(position: w, overlap: pixelOverlapW, isFirst: isFirstCol)

                    let weight = weightT * weightH * weightW

                    if weight >= 1.0 {
                        output[outIdx] = tileVal
                    } else {
                        output[outIdx] = output[outIdx] * (1.0 - weight) + tileVal * weight
                    }
                }
            }
        }
    }
}
