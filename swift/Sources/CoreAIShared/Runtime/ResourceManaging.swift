// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Lifecycle management for lazily-loaded resources (models, buffers).
///
public protocol ResourceManaging: Sendable {
    /// Load model weights and allocate inference buffers.
    func loadResources() async throws
    /// Release all resources. Safe to call multiple times.
    func unloadResources() async
}

extension ResourceManaging {
    public func prewarmResources() async throws {
        try await loadResources()
        await unloadResources()
    }
}
