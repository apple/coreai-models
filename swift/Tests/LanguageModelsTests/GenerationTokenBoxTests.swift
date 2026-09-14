// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Covers the active-token bookkeeping both sequential engines share: install/isBusy
/// transitions, cancellation, and the clearIfActive guard that keeps a newer generation
/// from being cleared by a stale iterator.
@Suite("GenerationTokenBox")
struct GenerationTokenBoxTests {
    @Test("A fresh box is idle")
    func freshBoxIsIdle() {
        #expect(!GenerationTokenBox().isBusy)
    }

    @Test("install makes the box busy")
    func installMakesBusy() {
        let box = GenerationTokenBox()
        box.install(GenerationToken())
        #expect(box.isBusy)
    }

    @Test("cancelActive clears the token and cancels it")
    func cancelActiveClearsAndCancels() {
        let box = GenerationTokenBox()
        let token = GenerationToken()
        box.install(token)

        box.cancelActive()

        #expect(!box.isBusy)
        #expect(token.isCancelled)
    }

    @Test("cancelActive is safe when idle")
    func cancelActiveWhenIdle() {
        let box = GenerationTokenBox()
        box.cancelActive()
        #expect(!box.isBusy)
    }

    @Test("clearIfActive clears the matching token")
    func clearIfActiveMatching() {
        let box = GenerationTokenBox()
        let token = GenerationToken()
        box.install(token)

        box.clearIfActive(token)

        #expect(!box.isBusy)
    }

    @Test("clearIfActive leaves a newer token untouched")
    func clearIfActiveNonMatching() {
        let box = GenerationTokenBox()
        let old = GenerationToken()
        let new = GenerationToken()
        box.install(old)
        box.install(new)

        // The stale iterator for `old` finishes and tries to release the engine.
        box.clearIfActive(old)

        // `new` is still the active generation.
        #expect(box.isBusy)

        box.clearIfActive(new)
        #expect(!box.isBusy)
    }

    @Test("clearIfActive is safe when idle")
    func clearIfActiveWhenIdle() {
        let box = GenerationTokenBox()
        box.clearIfActive(GenerationToken())
        #expect(!box.isBusy)
    }

    @Test("install does not cancel the prior token")
    func installDoesNotCancelPrior() {
        let box = GenerationTokenBox()
        let old = GenerationToken()
        let new = GenerationToken()
        box.install(old)
        box.install(new)

        // install() only supersedes; it never cancels. Callers that want the previous
        // generation stopped call cancelActive() first (as generate() does).
        #expect(!old.isCancelled)
        #expect(box.isBusy)
    }

    @Test("cancelActive then install models a fresh turn")
    func cancelThenInstall() {
        let box = GenerationTokenBox()
        let old = GenerationToken()
        box.install(old)

        // generate() supersedes any in-flight generation.
        box.cancelActive()
        #expect(old.isCancelled)

        let new = GenerationToken()
        box.install(new)
        #expect(box.isBusy)
        #expect(!new.isCancelled)
    }
}
