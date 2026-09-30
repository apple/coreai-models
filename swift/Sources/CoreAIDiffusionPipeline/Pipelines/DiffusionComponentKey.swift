// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Canonical `assets` keys for diffusion bundle components.
///
/// Mirrors `ModelBundle.ComponentKey`. These are the keys the descriptor reads from a
/// bundle's `metadata.json` `assets` map. On-disk asset *filenames* (e.g. `Transformer.aimodel`)
/// are pipeline-specific and live next to their pipeline, not here.
public enum DiffusionComponentKey {
    public static let textEncoder = "text_encoder"
    public static let transformer = "transformer"
    public static let vaeDecoder = "vae_decoder"
    public static let vaeEncoder = "vae_encoder"
}
