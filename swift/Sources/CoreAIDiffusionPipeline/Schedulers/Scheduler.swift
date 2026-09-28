// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Supported scheduler algorithms.
public enum SchedulerType: String, Sendable, CaseIterable {
    case discreteFlow = "flow_match_euler"
}

/// What the model predicts at each denoising step.
public enum PredictionType: String, Codable, Sendable {
    case epsilon
    case vPrediction = "v_prediction"
    case flow
    case flowMatching = "flow_matching"
}
