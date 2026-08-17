# Muse Glimmer

Meta's Muse Glimmer 30B for on-device agentic tasks via Core AI. Apache 2.0 license.

## Supported Models

| Model            | Parameters | Context | macOS | iOS |
|------------------|-----------|---------|-------|-----|
| Muse-Glimmer-30B | ~29.6B    | 131072  | Yes   | No  |

**Architecture notes:**

- Dense causal transformer with perception encoder (ViT-G/14, ~1.8B, separate).
- **Local/Global attention**: [S,S,S,G] repeating (39 sliding + 13 full). Sliding window = 2048.
- **RoPE on local layers only** (θ=500K). Global layers skip RoPE.
- **Gated attention**: learned gate projection (`sigmoid(gate_proj(x))`) on attention output.
- **Extreme GQA**: 32 Q / 2 KV heads (16:1 ratio).
- **CenteredRMSNorm** (`1 + weight`) for all layer norms; plain RMSNorm for final norm.
- **Weight-less RMSNorm** on embeddings (no learned scale).
- **QK norm**: shared RMSNorm applied to Q and K per-head before attention.
- **qk_scale_factor**: multiplies Q directly after QK norm (3.87). SDPA uses default `1/sqrt(d)`.
- **output_multiplier**: scales final hidden state before lm_head (0.196).
- **Logit softcapping**: `tanh(logits/20) * 20`.
- SwiGLU MLP, no attention bias.

## Evaluation Results

Perplexity on [WikiText-2](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) (word_perplexity, lower is better):

| Model | Compression | Precision | word_ppl |
|-------|-------------|-----------|----------|
| 30B   | none        | FP16      | 7.71     |
| 30B   | INT4        | FP16      | ~8.4     |

## Export

```bash
uv run coreai.llm.export muse-glimmer-30b
```

## Run

```bash
swift run -c release llm-runner --model path/to/exported_model --prompt "Hello"
```

## Benchmark

```bash
swift run -c release llm-benchmark --model path/to/exported_model -p 512 -g 1024 -n 5
```

## Notes

- At 30B parameters, the model requires ~60 GB memory unquantized (FP16).
  With INT4 quantization, the language model fits in ~17 GB.
- DFlash speculative decoding drafter is not yet implemented.
- The perception encoder (ViT-G/14) is separate and not included in the
  text-only export.
