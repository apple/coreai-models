# Parakeet TDT

Automatic speech recognition (ASR) model from NVIDIA. Pairs a FastConformer encoder with a Token-and-Duration Transducer (TDT) decoder that predicts `(token, duration)` pairs each step, letting greedy decoding skip blank-only frames for ~2-4x faster inference vs. standard RNN-T.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run export.py
```

Saves a bundle directory at `<repo-root>/exports/<model>_<dtype>_<static|dynamic>/` containing three `.aimodel` assets (`encoder`, `decoder_step`, `joint`), the processor (feature extractor + tokenizer), and a `metadata.json` describing the bundle. Pass `--output-dir <path>` to override the destination.

```sh
uv run export.py --help
```

**Options:**

| Flag               | Description                                    | Default                       |
| ------------------ | ---------------------------------------------- | ----------------------------- |
| `--model`          | Model variant                                  | `nvidia/parakeet-tdt-0.6b-v3` |
| `--output-dir`     | Output directory for the bundle                | `<repo-root>/exports/`        |
| `--dtype`          | `float16`, `float32`                           | `float32`                     |
| `--dynamic`        | Encoder accepts variable audio length          | static (5s default)           |
| `--audio-seconds`  | Length of dummy audio for static encoder trace | `5.0`                         |
| `--overwrite`      | Overwrite existing bundle                      | —                             |

**Supported models:**

| Model                         | Parameters |
| ----------------------------- | ---------- |
| nvidia/parakeet-tdt-0.6b-v3   | 0.6B       |

## Why three graphs?

Parakeet TDT's runtime decoding is autoregressive with duration-aware time advancement: each step samples a `(token, duration)` pair from the joint network, then advances the encoder frame pointer by `duration` (and only runs the LSTM prediction net when the token is not blank). That control flow lives in `ParakeetTDTGenerationMixin.generate`, not in `forward`, so `torch.export` cannot capture it as a single graph. The bundle exposes the three building blocks the runtime needs:

| Graph          | Inputs                                                              | Outputs                                            |
| -------------- | ------------------------------------------------------------------- | -------------------------------------------------- |
| `encoder`      | `input_features (B, T_audio, n_mels)`                               | `encoder_hidden_states (B, T_enc, decoder_hidden)` |
| `decoder_step` | `input_ids (B, 1)`, `hidden_state`, `cell_state`                    | `decoder_output`, `new_hidden_state`, `new_cell_state` |
| `joint`        | `decoder_hidden_states (B, 1, H)`, `encoder_hidden_states (B, 1, H)` | `logits (B, 1, vocab + len(durations))`            |

The encoder graph already includes `encoder_projector`, so the joint network's two addends share the same hidden size.

## Streaming

This recipe exports the full-utterance encoder; cache-aware / chunked-attention streaming is not yet implemented in `transformers` for Parakeet. The `decoder_step` and `joint` graphs are already streaming-shaped (single-step, explicit LSTM state in/out), so once a chunked encoder lands upstream the same bundle layout extends to streaming with only an encoder swap.

[^1]: [TDT paper](https://arxiv.org/abs/2304.06795) · [HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
