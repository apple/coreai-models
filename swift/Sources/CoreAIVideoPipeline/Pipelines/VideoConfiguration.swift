// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics

/// Configuration for a video generation request.
public struct VideoConfiguration: Sendable {
    public var prompt: String
    public var negativePrompt: String
    public var seed: UInt32
    public var stepCount: Int
    public var guidanceScale: Float
    public var numFrames: Int
    public var fps: Int
    public var width: Int
    public var height: Int
    public var conditioningImage: CGImage?

    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 42,
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        numFrames: Int = 49,
        fps: Int = 24,
        width: Int = 512,
        height: Int = 320,
        conditioningImage: CGImage? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.numFrames = numFrames
        self.fps = fps
        self.width = width
        self.height = height
        self.conditioningImage = conditioningImage
    }
}
