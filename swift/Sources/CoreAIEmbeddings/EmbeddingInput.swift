// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// One input to embed: a text or an image.
public struct EmbeddingInput: Sendable {
    enum Content: Sendable {
        case text(String)
        case image(CGImage)
    }

    let content: Content

    var isText: Bool {
        if case .text = content { return true }
        return false
    }

    /// A text.
    public static func text(_ text: String) -> EmbeddingInput {
        EmbeddingInput(content: .text(text))
    }

    /// An image, read as stored.
    // TODO: apply EXIF orientation and take a region of interest, as Vision's request handlers do;
    // a sideways photo embeds sideways today.
    public static func image(_ image: CGImage) -> EmbeddingInput {
        EmbeddingInput(content: .image(image))
    }
}
