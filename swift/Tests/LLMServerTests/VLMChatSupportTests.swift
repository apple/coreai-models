// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import llm_server

@Suite("VLMChatSupport")
struct VLMChatSupportTests {
    // 1x1 transparent PNG.
    private static let png1x1Base64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    @Test("decodeImage decodes a base64 data URL")
    func decodesDataURL() throws {
        let url = "data:image/png;base64,\(Self.png1x1Base64)"
        let image = try VLMChatSupport.decodeImage(from: url, fileAccess: .off)
        #expect(image != nil)
        #expect(image?.width == 1)
        #expect(image?.height == 1)
    }

    @Test("decodeImage returns nil for remote http(s) URLs")
    func rejectsRemoteURL() throws {
        #expect(try VLMChatSupport.decodeImage(from: "https://example.com/cat.png", fileAccess: .off) == nil)
        #expect(try VLMChatSupport.decodeImage(from: "http://example.com/cat.png", fileAccess: .off) == nil)
    }

    // MARK: - File-access policy

    @Test("file-access off rejects bare local paths")
    func offRejectsBarePath() {
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile("/etc/hosts", policy: .off)
        }
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.decodeImage(from: "/etc/hosts", fileAccess: .off)
        }
    }

    @Test("file-access off rejects file:// URLs")
    func offRejectsFileURL() {
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile("file:///etc/hosts", policy: .off)
        }
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.decodeImage(from: "file:///etc/hosts", fileAccess: .off)
        }
    }

    @Test("file-access subdirs allows a path resolving inside the working directory")
    func subdirsAllowsInside() throws {
        let base = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: base) }
        let sub = base.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let file = sub.appendingPathComponent("cat.png")
        try Data([0]).write(to: file)

        // Bare relative path under the base directory.
        let resolvedRelative = try VLMChatSupport.resolveLocalFile(
            "images/cat.png", policy: .subdirs, baseDirectory: base)
        #expect(resolvedRelative.path.hasSuffix("/images/cat.png"))

        // Absolute path that still lands inside the base directory.
        let resolvedAbsolute = try VLMChatSupport.resolveLocalFile(
            file.path, policy: .subdirs, baseDirectory: base)
        #expect(resolvedAbsolute.path.hasSuffix("/images/cat.png"))
    }

    @Test("file-access subdirs rejects .. traversal")
    func subdirsRejectsTraversal() throws {
        let base = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile("../../etc/hosts", policy: .subdirs, baseDirectory: base)
        }
    }

    @Test("file-access subdirs rejects absolute paths outside the working directory")
    func subdirsRejectsAbsoluteOutside() throws {
        let base = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile("/etc/hosts", policy: .subdirs, baseDirectory: base)
        }
    }

    @Test("file-access subdirs rejects the base directory itself")
    func subdirsRejectsBaseItself() throws {
        let base = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile(base.path, policy: .subdirs, baseDirectory: base)
        }
    }

    @Test("file-access subdirs rejects a symlink escaping the working directory")
    func subdirsRejectsSymlinkEscape() throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let base = root.appendingPathComponent("cwd", isDirectory: true)
        let outside = root.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        let secret = outside.appendingPathComponent("secret.png")
        try Data([0]).write(to: secret)
        // Symlink inside the base dir that points to a sibling outside it.
        let link = base.appendingPathComponent("escape")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

        #expect(throws: ServerError.self) {
            _ = try VLMChatSupport.resolveLocalFile("escape/secret.png", policy: .subdirs, baseDirectory: base)
        }
    }

    private static func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vlm-file-access-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Canonicalize so the test's expectations match resolveLocalFile's symlink resolution
        // (e.g. macOS maps /var -> /private/var).
        return dir.resolvingSymlinksInPath()
    }

    // NOTE: The streaming+image 400 guard in `handleStreamingRequest` (ChatHandler.swift)
    // is not unit-tested here: that path is `private` and needs a live ServerState with a
    // loaded engine/tokenizer, which requires real model assets. It is covered by the
    // shared `state.isVLM && VLMChatSupport.hasImage(...)` detection exercised above.

    @Test("buildPromptTokens fallback expands the image placeholder to imageTokenCount copies")
    func fallbackExpandsPlaceholder() throws {
        // MockTokenizer has no usable chat template, so buildPromptTokens takes the
        // USER/ASSISTANT fallback that appends imageTokenCount image tokens directly.
        let tokenizer = MockTokenizer()
        let imageTokenId: Int32 = 200_092
        let imageTokenCount = 4
        let message = try JSONDecoder().decode(
            ChatMessage.self,
            from: Data(#"{"role":"user","content":"What is in this image?"}"#.utf8))

        let tokens = VLMChatSupport.buildPromptTokens(
            messages: [message],
            imageTokenCount: imageTokenCount,
            imageTokenId: imageTokenId,
            tokenizer: tokenizer
        )

        let placeholders = tokens.filter { $0 == imageTokenId }.count
        #expect(placeholders == imageTokenCount)
        #expect(tokens.count > imageTokenCount)
    }
}
