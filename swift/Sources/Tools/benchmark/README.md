# llm-benchmark

Measure LLM inference throughput on Apple Silicon using Core AI engines.
Reports prompt (prefill) and generation (decode) tokens/sec across multiple
trials, modeled after [mlx-lm](https://github.com/ml-explore/mlx-lm)'s
benchmark format.

## Usage

```bash
swift run -c release llm-benchmark --model path/to/exported_model_bundle
```

## Options

| Flag                        | Default  | Description                            |
|-----------------------------|----------|----------------------------------------|
| `--model <path>`            | required | Path to an exported model bundle       |
| `-p, --prompt-tokens <n>`   | 512      | Synthetic prompt length (random tokens)|
| `-g, --generation-tokens <n>` | 1024   | Number of tokens to generate           |
| `-n, --num-trials <n>`      | 5        | Number of timed trials (after warmup)  |
| `--seed <n>`                | 0        | Random seed for the synthetic prompt   |
| `--output-json <path>`      | —        | Write structured results to JSON file  |

## Example

```bash
# Quick benchmark: 256 prompt, 512 generation, 3 trials
swift run -c release llm-benchmark \
    --model exports/qwen3_0_6b_dynamic \
    -p 256 -g 512 -n 3

# Save results for CI comparison
swift run -c release llm-benchmark \
    --model exports/qwen3_0_6b_dynamic \
    --output-json results/qwen3_0_6b.json
```

## Output

```
⏳ Preparing AI asset...
⚙️  Warming up engine...
🔄 Benchmarking with 512 prompt tokens, 1024 generation tokens

🧪 Trial 1
⚡ Prompt:     1042.315 tokens/sec
🏃 Generation: 58.721 tokens/sec

🧪 Trial 2
⚡ Prompt:     1087.442 tokens/sec
🏃 Generation: 59.103 tokens/sec

📊 Benchmark Summary:
==================================================
Prompt:     1064.879 tokens/sec
Generation: 58.912 tokens/sec
==================================================
```

## JSON output format

```json
{
  "model": "qwen3_0_6b_dynamic",
  "prompt_tokens": 512,
  "generation_tokens": 1024,
  "num_trials": 5,
  "trials": [
    {"prompt_tps": 1042.3, "gen_tps": 58.7},
    {"prompt_tps": 1087.4, "gen_tps": 59.1}
  ],
  "averages": {
    "prompt_tps": 1064.9,
    "generation_tps": 58.9
  }
}
```

## Notes

- Always build in **Release** mode (`-c release`) for accurate numbers.
- The first trial after warmup may still be slightly slower due to kernel
  compilation caches; subsequent trials are more stable.
- Prompt throughput measures time-to-first-token (prefill latency).
- Generation throughput measures decode tokens per second after the first.
- The synthetic prompt uses random token IDs — results reflect raw engine
  throughput, not model quality.
