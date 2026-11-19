# Contributing to Core AI Models

Thank you for your interest in Core AI Models. We welcome contributions
within a defined scope. Please read this document before you open an issue
or a pull request.

## What we accept

- **Bug fixes** in the Python export scripts and Swift runtime utilities
- **New models**: export recipes, with or without new runtime support
- **Documentation and test improvements**

## What we do not accept

- Major new features, or changes to the public API surface of the Swift
  package or the Python primitives
- Models whose license does not permit use and redistribution

If you want to propose something outside this scope, open an issue first.

## Before you open a pull request

For anything beyond a trivial fix, open an issue that describes the bug
or the enhancement. Wait for a maintainer to respond before you write code.
We close large pull requests that have no justifying issue.

## Adding a new model

We prefer models under BSD, MIT, Apache 2.0, and Creative Commons (CC0,
CC BY 3.0, CC BY 4.0) licenses. We review models under a custom license more
closely and may decline them.

Use the [model request](https://github.com/apple/coreai-models/issues/new?template=model_request.yml)
form to propose a model before you build the export recipe. In your pull
request, include sample runtime output and performance benchmark results.

## Set up your environment

Install `uv`, then see the [README](README.md#requirements) for requirements.
Install `pre-commit`, then install the hooks:

```bash
uv tool install pre-commit  # or: brew install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Check your change

Continuous integration runs these checks. Run them locally first.

```bash
uv run --frozen --only-group lint ruff check python/src/ python/tests/ models/
uv run --frozen --only-group lint ruff format --check python/src/ python/tests/ models/
swift format lint --strict --recursive swift/Sources/ swift/Tests/
uv run pytest python/tests/test_model_units -x -q
```

New functionality and bug fixes need tests.

Write commit messages that say what changed and why.

## Sign your commits

All commits must be signed and verified. We can't build or merge pull requests
that have unsigned or unverified commits. See GitHub's
[commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
documentation.

## Our standards

We evaluate contributions on technical quality, adherence to existing
conventions, your understanding of the change, and clarity of communication.

You may use AI-assisted tools. You remain fully accountable for everything
you submit. Review it, understand it, and be ready to explain it. We treat
low-quality submissions that show no human review as spam.

## Report issues

Search [existing issues](https://github.com/apple/coreai-models/issues)
first. Then use an
[issue template](https://github.com/apple/coreai-models/issues/new/choose):

- **Bug report** for something that does not work as expected
- **Model request** for a model you want in the catalog
- **Workflow or runtime feedback** for the export workflow, Swift utilities,
  or docs

## Code of conduct

This project follows the
[Apple Open Source Code of Conduct](https://github.com/apple/.github/blob/main/CODE_OF_CONDUCT.md).
