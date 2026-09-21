// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Isolation domain for the video segmentation frame loop.
///
/// The loop is inherently sequential. Frame N's tracker reads the memory frame N-1 wrote,
/// and every stage shares one mutable `VideoInferenceSession`, which one actor keeps inside
/// a single isolation domain.
///
/// ``VideoSegmentationEngine`` sits on its own actor. It holds no per-frame state, so a
/// caller can unload the asset without serializing against a running frame loop.
@globalActor
public actor VideoSegmentationActor {
    public static let shared = VideoSegmentationActor()
}
