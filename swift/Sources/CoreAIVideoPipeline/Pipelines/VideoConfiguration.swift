// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Configuration for video generation.
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
    public var dumpDirectory: String?
    public var loadNoisePath: String?

    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = 42,
        stepCount: Int = 50,
        guidanceScale: Float = 5.0,
        numFrames: Int = 81,
        fps: Int = 16,
        width: Int = 832,
        height: Int = 480,
        dumpDirectory: String? = nil,
        loadNoisePath: String? = nil
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
        self.dumpDirectory = dumpDirectory
        self.loadNoisePath = loadNoisePath
    }
}
