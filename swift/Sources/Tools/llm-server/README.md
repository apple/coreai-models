# llm-server

`llm-server` is a prototype LLM server engine, to be used for evaluations and to serve as an example for
integrating Core AI into high-performance agentic workflows.

It serves a single exported LLM bundle over the chat-completions and completions HTTP APIs that eval
harnesses and agent frameworks use.

## How to build and use

Export a model (see [models/README.md](../../../../models/README.md)), then build and start the server:

```bash
uv run coreai.llm.export qwen3-0.6b
swift run -c release llm-server --model exports/qwen3_0_6b_4bit_dynamic --port 8080
```

Then send it a request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "reasoning_effort": "none"}'
```

### Serving several requests at once

By default the server decodes one request at a time. To decode several together, export the model with a
batch dimension:

- `--attention-mask batched` gives the graph an attention-mask input, so each row of the batch can be at its
  own position in its own sequence.
- `--dynamic-batch-size 8` sets the most rows one forward pass can take.

Then start the server with `--concurrent-requests auto`, which uses the bundle's maximum (8 here), or a
smaller number:

```bash
uv run coreai.llm.export qwen3-0.6b --attention-mask batched --dynamic-batch-size 8
swift run -c release llm-server --model exports/qwen3_0_6b_4bit_dynamic --concurrent-requests auto
```

New requests join the batch as soon as a row frees up. `GET /ready` shows how many are active and queued.

## How to integrate into your own (or a public) eval harness

- **Over HTTP.** Point the harness at `http://127.0.0.1:8080/v1`. Generation tasks use
  `/v1/chat/completions`; log-likelihood tasks use `/v1/completions` with `echo` and `logprobs`. For
  example, with lm-evaluation-harness:

  ```bash
  lm_eval --model local-completions --tasks hellaswag \
    --model_args model=qwen3-0.6b,base_url=http://127.0.0.1:8080/v1/completions,tokenizer=Qwen/Qwen3-0.6B,num_concurrent=8
  ```

  Match the harness's concurrency to `--concurrent-requests`; requests beyond it wait in a queue of
  `--max-queue-depth`, and the server returns HTTP 429 when that queue is full.
- **Without HTTP.** `--replay requests.jsonl` runs a file of requests through the same generation code,
  writes one JSON result per line to `--replay-output`, and exits. The request format is
  `CoreAILMCommon/ReplayTypes.swift`. Add `--replay-concurrent` to run them together.
- **Reproducibility.** Responses carry a `system_fingerprint` that changes with the model, engine, batch
  size, context length and sampling defaults. Pass `seed` for repeatable sampling.

## Common options

| Option                                               | Description                                                               | Default  |
| ---------------------------------------------------- | ------------------------------------------------------------------------- | -------- |
| `--model`                                            | Model bundle directory                                                    | required |
| `--port`                                             | HTTP port                                                                 | `8080`   |
| `--concurrent-requests`                              | Requests decoded together: a number, or `auto` for the bundle's max batch | `1`      |
| `--max-queue-depth`                                  | Requests waiting beyond the concurrent ones before HTTP 429               | `16`     |
| `--variant`                                          | Engine: `auto`, `coreai-pipelined`, `coreai-sequential`                   | `auto`   |
| `--max-tokens`                                       | Default generation limit per request                                      | `512`    |
| `--temperature`, `--top-k`, `--top-p`, `--min-p`     | Default sampling (temperature `0` is greedy)                              | `0.7`    |
| `--default-reasoning-effort`, `--no-thinking`        | Default reasoning effort for models that think                            | model's  |
| `--replay`, `--replay-output`, `--replay-concurrent` | Run a JSONL request file instead of serving                               | off      |
| `--clear-coreai-cache`                               | Re-specialize the model on load                                           | off      |

For the full list, run `swift run llm-server --help`.

## Endpoints

| Endpoint                    | Description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/chat/completions` | Chat generation, with streaming, tool calls and `reasoning_effort`. Requests with the same `X-Session-ID` header reuse the cached prompt prefix.                    |
| `POST /v1/completions`      | Prompt scoring (`echo`, `logprobs`) for log-likelihood evals. Needs an engine that returns logits: `coreai-sequential`, which batched bundles select automatically. |
| `POST /v1`                  | Routes to completions when the body has a `prompt`, otherwise to chat completions.                                                                                  |
| `GET /v1/models`            | The served model.                                                                                                                                                   |
| `GET /ready`                | Readiness, prompt cache use, process and GPU memory, and batch occupancy.                                                                                           |
| `GET /v1/stats`             | Request and token counters since start.                                                                                                                             |
| `GET /health`               | Liveness.                                                                                                                                                           |
