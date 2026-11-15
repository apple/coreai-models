// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import CoreAIShared
import CoreGraphics
import CryptoKit
import Foundation

/// Content identity of an image: the SHA-256 digest of its encoded bytes.
struct ImageKey: Hashable, Sendable {
    let digest: SHA256.Digest

    init(data: Data) {
        digest = SHA256.hash(data: data)
    }
}

/// An image from an `image_url` content part.
struct RequestImage {
    let key: ImageKey
    let image: CGImage
}

/// Parsing of `image_url` content parts.
///
/// Only base64 data URLs (`data:image/<type>;base64,...`) are accepted. The request body limit bounds
/// their size; `maxImageSide` bounds the decoded pixel size.
enum ImageInput {
    /// Longest image side, in pixels, that a request may send.
    static let maxImageSide = 8192

    /// The images of `messages`, in prompt order. Throws `ServerError.badRequest` for an image that
    /// is not a base64 image data URL, that ImageIO cannot decode, or that exceeds `maxImageSide`.
    static func images(in messages: [ChatMessage]) throws -> [RequestImage] {
        try messages.flatMap { try $0.content.imageDataURLs.map(image(fromURL:)) }
    }

    private static func image(fromURL url: String) throws -> RequestImage {
        let data = try dataURLPayload(url)
        guard let image = try? CGImageUtils.load(data: data) else {
            throw ServerError.badRequest("image_url data could not be decoded as an image")
        }
        guard image.width <= maxImageSide, image.height <= maxImageSide else {
            throw ServerError.badRequest(
                "image_url image is \(image.width)x\(image.height) pixels; the server accepts up to "
                    + "\(maxImageSide) per side")
        }
        return RequestImage(key: ImageKey(data: data), image: image)
    }

    /// The decoded bytes of a `data:image/<type>;base64,<payload>` URL. Throws
    /// `ServerError.badRequest` for any other URL.
    private static func dataURLPayload(_ url: String) throws -> Data {
        guard url.prefix(5).lowercased() == "data:" else {
            let scheme = url.firstIndex(of: ":").map { url[..<$0].prefix(32) }
            throw ServerError.badRequest(
                "image_url must be a base64 data URL (data:image/<type>;base64,...); "
                    + (scheme.map { "\($0): URLs are not supported" } ?? "the value has no URL scheme"))
        }
        guard let comma = url.firstIndex(of: ","),
            case let header = url[url.index(url.startIndex, offsetBy: 5)..<comma].lowercased(),
            header.hasPrefix("image/"), header.hasSuffix(";base64"),
            let data = Data(base64Encoded: String(url[url.index(after: comma)...]), options: .ignoreUnknownCharacters),
            !data.isEmpty
        else {
            throw ServerError.badRequest("image_url data URL must be data:image/<type>;base64,<data>")
        }
        return data
    }
}
