// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

#if os(macOS)

/// Covers the review comment about `contentsOfDirectory().first` being non-deterministic. Every
/// test builds a fake cache tree under a temporary root, so nothing depends on whatever happens to
/// be in the developer's real `~/.cache/huggingface`.
@Suite("HuggingFace cache snapshot resolution")
struct HuggingFaceCacheTests {
    private static let model = "openai/whisper-large-v3-turbo"
    private static let repoName = "models--openai--whisper-large-v3-turbo"

    /// Builds `<root>/.cache/huggingface/hub/<repo>/snapshots/<each>` plus an optional `refs/main`.
    private func makeCache(
        in root: URL, snapshots: [String], refsMain: String? = nil
    ) throws {
        let fm = FileManager.default
        let repo = root.appending(path: ".cache/huggingface/hub").appending(path: Self.repoName)
        for name in snapshots {
            try fm.createDirectory(
                at: repo.appending(path: "snapshots").appending(path: name),
                withIntermediateDirectories: true)
        }
        if let refsMain {
            let refs = repo.appending(path: "refs")
            try fm.createDirectory(at: refs, withIntermediateDirectories: true)
            try Data(refsMain.utf8).write(to: refs.appending(path: "main"))
        }
    }

    @Test("The model name maps to the cache directory name")
    func modelNameMapsToDirectory() throws {
        try withTempDirectory { root in
            try makeCache(in: root, snapshots: ["abc123"])
            let resolved = huggingFaceCacheSnapshot(forModelName: Self.model, root: root)
            #expect(resolved?.lastPathComponent == "abc123")
        }
    }

    @Test("refs/main takes precedence over the sorted fallback")
    func refsMainWins() throws {
        try withTempDirectory { root in
            // `mmm` sorts in the middle, so picking it can only be the result of reading refs/main.
            try makeCache(in: root, snapshots: ["aaa", "mmm", "zzz"], refsMain: "mmm")
            #expect(
                huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?.lastPathComponent
                    == "mmm")
        }
    }

    @Test("refs/main is trimmed of surrounding whitespace")
    func refsMainIsTrimmed() throws {
        try withTempDirectory { root in
            try makeCache(in: root, snapshots: ["aaa", "mmm"], refsMain: "  mmm\n")
            #expect(
                huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?.lastPathComponent
                    == "mmm")
        }
    }

    @Test("A refs/main pointing at nothing falls back to the sorted scan")
    func danglingRefsMainFallsBack() throws {
        try withTempDirectory { root in
            try makeCache(in: root, snapshots: ["aaa", "zzz"], refsMain: "deadbeef")
            #expect(
                huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?.lastPathComponent
                    == "zzz")
        }
    }

    @Test("An empty refs/main falls back to the sorted scan")
    func emptyRefsMainFallsBack() throws {
        try withTempDirectory { root in
            try makeCache(in: root, snapshots: ["aaa", "zzz"], refsMain: "\n")
            #expect(
                huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?.lastPathComponent
                    == "zzz")
        }
    }

    @Test("Without refs/main the choice is deterministic regardless of creation order")
    func fallbackIsDeterministic() throws {
        // This is what replaced `contentsOfDirectory().first`. Snapshot directories are named by
        // commit hash, so no ordering means "newest" — sorting only guarantees reproducibility.
        for order in [["mmm", "aaa", "zzz"], ["zzz", "mmm", "aaa"], ["aaa", "zzz", "mmm"]] {
            try withTempDirectory { root in
                try makeCache(in: root, snapshots: order)
                #expect(
                    huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?
                        .lastPathComponent == "zzz", "creation order \(order)")
            }
        }
    }

    @Test("Dotfiles are filtered out")
    func dotfilesFiltered() throws {
        try withTempDirectory { root in
            // `.zzz` sorts after `abc`, so an unfiltered scan would pick it.
            try makeCache(in: root, snapshots: ["abc", ".zzz"])
            #expect(
                huggingFaceCacheSnapshot(forModelName: Self.model, root: root)?.lastPathComponent
                    == "abc")
        }
    }

    @Test("An uncached repository returns nil")
    func uncachedReturnsNil() throws {
        try withTempDirectory { root in
            #expect(huggingFaceCacheSnapshot(forModelName: Self.model, root: root) == nil)
        }
    }

    @Test("A missing or empty snapshots directory returns nil")
    func missingSnapshotsReturnsNil() throws {
        try withTempDirectory { root in
            let repo = root.appending(path: ".cache/huggingface/hub").appending(path: Self.repoName)
            try FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
            #expect(huggingFaceCacheSnapshot(forModelName: Self.model, root: root) == nil)

            try FileManager.default.createDirectory(
                at: repo.appending(path: "snapshots"), withIntermediateDirectories: true)
            #expect(huggingFaceCacheSnapshot(forModelName: Self.model, root: root) == nil)
        }
    }

    @Test("The default root is the user home directory")
    func defaultRootIsHome() {
        // The only assertion that touches the real cache, and it is safe because the name cannot
        // exist. Confirms the defaulted parameter still points where production expects.
        #expect(huggingFaceCacheSnapshot(forModelName: "definitely/not-cached-\(UUID())") == nil)
    }
}

#endif
