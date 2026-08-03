# Gemma 4 VLM — Core AI benchmarks (Apple silicon, GPU)

Prefill / decode throughput for the exported Gemma 4 vision-language bundles
(`embed` + `main` + `vision`), measured on-device via `llm-runner`.

## Setup

- **Device:** Apple M5 Max, 128 GB unified memory (macOS).
- **Weights:** 4-bit (uint4, asymmetric, group size 32), int8 externalized PLE, bf16 compute / bf16 KV cache.
- **Sampling:** greedy.
- **Input:** a 768×768 image (256 vision soft-tokens) plus a text prompt of the listed length.
- **State:** warm — the Core AI specialization cache is populated from a prior run (cold model-load times listed separately).

### Methodology note

These are **single-pass** `llm-runner` timings (one prefill + one decode run), not
averaged trials from the `llm-benchmark` harness. The benchmark harness is
text-only today and cannot drive the VLM `main` graph (which takes
`inputs_embeds`, not `input_ids`), so VLM numbers come from the runner's own
`PerformanceMetrics`. Treat them as representative, not as averaged benchmark
results.

**Prefill throughput is strongly prompt-length dependent.** At very short prompts
the measurement is dominated by the one-time vision-encoder pass and first-token
overhead, so prefill tok/s looks low; it rises steeply once the prompt is long
enough to amortize that fixed cost.

## Gemma 4 E2B VLM

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) |
| ---: | ---: | ---: |
| 270    | 62.9  | — (cold) |
| 991    | 308.7 | 63.8 |
| 4,467  | 4,234 | ~56 |
| 14,267 | 5,289 | ~26 |
| 22,141 | 4,652 | ~39 |

- Cold model load ≈ 6.7 s; warm load ≈ 1.1 s.

## Gemma 4 26B-A4B VLM (MoE)

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) |
| ---: | ---: | ---: |
| 272    | 44.9  | — (cold) |
| 4,467  | 1,006 | ~33 |
| 14,267 | 2,030 | ~37 |
| 22,141 | 2,188 | ~36 |

- Cold model load ≈ 18 s (18 GB bundle).

## Notes

- Decode was measured with short generation lengths (8–32 tokens) and is
  approximate; the short-generation samples are noisy.
- The 26B model is a Mixture-of-Experts (A4B ≈ ~4B active parameters), which is
  why its decode is only modestly slower than E2B despite the much larger weight
  footprint, while its prefill (compute-bound) is roughly 2–2.5× slower.
