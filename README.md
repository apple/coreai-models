# Core AI Models

Model export recipes, Python primitives, and Swift runtime utilities for building on-device AI with [Core AI](https://developer.apple.com/documentation/coreai).

The main components include:
- **Model export** — Recipes to export popular open source models from Hugging Face and other sources to Core AI format.
- **Reusable primitives** — Python building blocks for authoring custom Core AI models in PyTorch.
- **Runtime utilities** — Swift package built on top of Core AI framework to run models on macOS and iOS.
- **Skills** — Plugins to help coding agents leverage Core AI effectively.

| Directory | What's inside                                                                                |
| --------- | -------------------------------------------------------------------------------------------- |
| `models/` | Model catalog with README and export recipes.                                                |
| `python/` | Python primitives for authoring and utilities for exporting models. |
| `swift/`  | Swift package (`coreai-models`): runtime utilities to integrate Core AI models in your app.  |
| `skills/` | Pluggable skills that enable coding agents to leverage Core AI more effectively.             |

## Requirements

If you haven't installed `uv`, install it by

```bash
brew install uv
```
or
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once installed successfully, refer to the README.md for each model or family of models, in `models` folder for their exporting recipe.

## Requirements (running and app integration)

- **macOS and iOS 27.0+**

- **Xcode 27.0+**

Core AI models are exported as standalone `.aimodel` files for integration into apps via the Core AI framework.

Some models require additional resources. Language models require a tokenizer, for instance, and diffusion models run multiple models in sequence as part of a single pipeline. For these cases, export recipes in this repo produce a resource folder containing one or more `.aimodel` files alongside any required resources. The Swift package in this repo provides runtime utilities for integrating these into an app.

Command line interface (CLI) tools are also included for running exported models directly on a Mac (requires Xcode 27.0+). See each model's README for available tools and example invocations.

## Explore supported models

Find supported models by

```bash
git clone https://github.com/apple/coreai-models.git && cd coreai-models
uv run coreai.model.registry --list-models
```

Run `uv run coreai.model.registry --help` for details.

## Agent Skills

This repo includes a plugin with skills to enable coding agents to use Core AI like an expert.

### Available skills

| Skill | Description |
| --- | --- |
| `working‑with‑coreai` | End-to-end workflow for deploying PyTorch models on Apple silicon, covering export with `coreai-torch` and running with the Core AI runtime. |
| `model‑authoring` | Empirical rules for authoring PyTorch models for on-device execution on Apple platforms, covering BC1S layout, op compatibility, KV cache patterns, precision rules, MoE, and common issues. |
| `model‑compression‑exploration` | Systematically explore weight compression configurations (quantization and palettization) for a PyTorch model using `coreai-opt`. |

### Install

Installation differs depending on your coding agent of choice.

#### Claude Code

Register the marketplace:

```
/plugin marketplace add git@github.com:apple/coreai-models.git
```

Alternatively, register the marketplace from a local git checkout:

```
/plugin marketplace add /path/to/coreai-models
```

Install the plugin:

```
/plugin install coreai-skills@coreai-models
```

#### Codex CLI

Register the marketplace:

```
codex plugin marketplace add https://github.com/apple/coreai-models
```

Alternatively, register the marketplace from a local git checkout:

```
codex plugin marketplace add /path/to/coreai-models
```

Launch Codex in your workspace:

```
codex
```

Install the plugin through the interactive browser: once the Codex session is
active in your terminal, open the plugin manager by typing `/plugins`, locate the
`coreai-models` marketplace tab (use your arrow keys or the built-in search),
select `coreai-skills`, and choose Install.

#### Gemini CLI

Install the extension from a local directory:

```
gemini extensions install /path/to/coreai-models/skills
```

Once installed, the skills activate automatically based on your task context,
or you can invoke them explicitly.

## Contributing

### We are not accepting code contributions at this time

Core AI Models is focused on maintaining a curated, well-tested gallery of
models and a reliable Swift package. We are not accepting pull requests at launch while we learn how the community uses this project.

If you open a pull request, it will be closed. This is not a reflection of
the quality of your contribution but it is a deliberate scope decision for this release.

### What we do welcome

We actively want your feedback! GitHub Issues are open for:

- **Bug reports** — if something in the Python scripts or Swift utilities does
  not work as expected
- **Model requests** — if you have ideas for models you would like to see, or
  improvements to the workflow or Swift utilities

Use the [issue templates](../../issues/new/choose) to get started.

## Support

- [GitHub Issues](../../issues) — Feedback, bug reports, and feature requests

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
