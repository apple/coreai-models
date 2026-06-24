// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Top-level model categories the runner ecosystem knows about.
///
/// The bundle's `kind` selects which kind-specific config block (and which
/// kind-specific Swift type — `LanguageBundle`, `DiffusionBundle`, etc.) is
/// expected on top of the common `ModelBundle`.
public enum BundleKind: String, Codable, Sendable, CaseIterable {
    case llm
    case vlm
    case diffusion
    case segmenter
    case speechRecognizer = "speech_recognizer"
    case speechRecognizerTDT = "speech_recognizer_tdt"
}
