---
name: auditing-helper-reuse
description: Use when a model-export change adding 500+ lines of source under python/src/coreai_models (new layers, encoders or an exporter) is about to become a final PR and may re-implement helpers the tree already has (RoPE / rotate_half, norms, MLPs, attention blocks, SDPA and masks, safetensors loaders, torch.export / conversion boilerplate and decompositions, metadata constants, CLI or test helpers), or when a reviewer asks to swap local code for an existing helper.
---

# Auditing helper reuse before a model-export PR

## Overview

Replace re-implemented helpers with existing ones only when the replacement is an **identity
transform**: exported graphs, weights and outputs stay exactly the same.

**A reuse that changes numbers or lowering is a model change, not a cleanup**, however small the
tolerance. That includes deleting a branch the target model never takes.

`scripts/graph_identity.py compare` is the arbiter: reading code finds candidates, `compare`
decides. Count added non-test source toward the 500 lines.

## Preconditions and invariant

- **Before:** the new code has tests, and every one that can run here passes. Failures elsewhere
  are acceptable only if they fail the same way on the base branch in the same environment;
  otherwise stop and fix them. A tier that cannot run (missing package, no network) is recorded as
  not run: it does not block the audit, but the PR stays in draft until it has run.
- **Invariant:** every test has the same outcome as in the baseline, and `compare` reports 0
  differences.

The **baseline** is the unedited PR head in a detached worktree; snapshot its `python/src` and run
its tests there. Run the suite one test file per process (a Core AI runtime crash kills a whole
run). Rerun a differing test on its own, three times, in both trees: flaky in both is the same
outcome, flaky in one is a difference.

## Process

1. **Snapshot the baseline first.** Write a spec (see the script's docstring) that builds every new
   module from tiny seeded configs. Give every branch an edit touches its own case: a GQA config
   even if the real model has as many KV heads as query heads, and both values of a module flag,
   set in the module that reads it after the change. Cover a caller that only builds from real
   weights with its smallest submodule that uses the helper. Put config-dependent facts in
   `extras()`: CLI `--help`, written metadata (no timestamps), errors from construction-time guards,
   and `mlir_hash` of converted programs. Then:
   - **control:** snapshot again; `compare` must report 0 (the spec is deterministic);
   - **negative control:** in a scratch copy, flip one constant in each area you plan to edit;
     `compare` must report differences for each, or the spec is blind there and must be fixed.
2. **Inventory every helper family**, starting from `references/helper-map.md` (check it against
   the tree; it goes stale). The biggest wins are often whole blocks copied from a peer model,
   export boilerplate and private imports, not the RoPE math.
3. **Check each candidate by reading both implementations**: layout, dtype, eps, constants, op
   order. A helper that fits only some call sites is still a reuse at those sites. Passing your
   config to a shared class is fine if the class reads only fields it already has.
4. **Apply only identities.** Reject near-misses, anything that needs the shared helper or your
   config changed to fit, and anything `compare` cannot cover. Write each rejection down with its
   reason; it answers "why not reuse X?". A reviewer's swap request goes through the same check and
   the Boundaries; if `compare` differs, reply with the first difference it reports.
5. **Prove it:** `compare` exits 0 against the baseline; the suite matches the baseline (reused test
   helpers keep the same test count); the new code's optional tiers run via the `uv run --with ...`
   overlay their docstrings name, or are listed as not run in the PR's Tests section; `ruff check`
   and `ruff format --check` pass. Have an independent reviewer try to refute each applied reuse;
   they also catch what `compare` cannot see, such as a new import pulling in a heavy package.
6. **Commit.** Fold each reuse into the commit that introduced the copy (`fixup! <subject>`, then
   `GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash <base>`). A promotion goes first: commit the
   move and the switch of callers already in the base on top of `<base>`, then
   `git rebase --onto <that commit> <base> <pr-branch>`; the new code's switch is one more fixup.
   Confirm `git diff <tested-sha>` is empty and every commit imports on its own. Rewriting history
   needs a force-push, so do it only on your own branch before reviews start; after that, add the
   reuses as new commits on top, promotion first.

## Boundaries

- `models/` and `primitives/` never import exporter packages (`export/`, `llm/`, `vlm/`,
  `diffusion/`, `embedding/`, ...), and `primitives/` never imports `models/`.
- A peer model's public classes are fair game (`mistral.TransformerBlock`, `DiffusionGemmaDenseMLP`).
- Never import a private `_helper` from another module, or `primitives/ios` from a macOS model.
  Promote the helper unchanged and public (layers and math to `primitives/`, export helpers such
  as decompositions to a leaf module like `_constants.py`), switch **every** caller and prove each
  one, and count `sys.modules` before and after importing a caller to check the new home's cost.
  If a caller cannot be proven, or the helper is a one-liner, keep your own copy instead.
  Extending a promoted helper is a later change with its own proof.

## Red flags — stop

- "Outputs match within 1e-6" or "cosine ≥ 0.9999"
- "Needs an on-device check to confirm" (reject the reuse instead)
- "That branch is dead for this model"
- "I read both, they're identical, skip the snapshot"
- "Only the raw graph differs"

The script's docstring documents the spec format; `references/helper-map.md` lists where helpers
live and look-alikes that are not identical.
