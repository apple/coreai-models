// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The files of a draft head, as `models/gemma4/export_drafter.py` writes them.
///
/// A bundle names them in `metadata.json`: the program under `assets.drafter` and the sidecars under
/// `auxiliary_assets` (`drafter_config`, `drafter_scan`, `drafter_scan_index`); see
/// ``LanguageModelBundle/drafter``.
package struct DrafterComponent: Sendable, Equatable {
    /// The `draft_step` program (`.aimodel`).
    package var step: URL
    /// Its JSON sidecar: format version, full-attention windows and RoPE parameters.
    package var config: URL
    /// The masked embedder's candidate table.
    package var scan: URL
    /// The candidate table's JSON index.
    package var scanIndex: URL

    package init(step: URL, config: URL, scan: URL, scanIndex: URL) {
        self.step = step
        self.config = config
        self.scan = scan
        self.scanIndex = scanIndex
    }

    /// The files under the names the exporter gives them, in `directory`.
    package init(directory: URL) {
        self.init(
            step: directory.appendingPathComponent("draft_step.aimodel"),
            config: directory.appendingPathComponent("draft_step.json"),
            scan: directory.appendingPathComponent("draft_scan.bin"),
            scanIndex: directory.appendingPathComponent("draft_scan.json"))
    }
}
