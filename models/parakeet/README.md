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
| `--audio-seconds`  | Length of dummy audio for static encoder trace (ignored with `--dynamic` or `--streaming`) | `5.0`                         |
| `--overwrite`      | Overwrite existing bundle                      | —                             |
| `--streaming`      | Fixed streaming window (see [Streaming](#streaming)) | —                       |

**Supported models:**

| Model                         | Parameters |
| ----------------------------- | ---------- |
| nvidia/parakeet-tdt-0.6b-v3   | 0.6B       |

## Running

### In your iOS and macOS applications

```swift
import CoreAISpeech

// Load an exported bundle directory (metadata.json + encoder/decoder_step/joint .aimodel assets + processor/).
let model = try await SpeechRecognitionModel(resourcesAt: URL(fileURLWithPath: "coreai-models/exports/parakeet-tdt-0.6b-v3_float32_static"))

// Transcribe an audio file — decoded and resampled to the model's sample rate automatically:
let (text, stats) = try await model.transcribe(audioURL: URL(fileURLWithPath: "audio.wav"))
print(text)

// Or transcribe raw mono PCM you already hold at model.sampleRate:
let (text2, _) = try await model.transcribe(pcm: pcmSamples)
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release speech-recognizer --model path/to/exported_bundle_dir --audio-path path/to/audio.wav
```

Accepts any audio the system can decode (`wav`, `flac`, `m4a`, …). Add `--warmup` to run a full transcription pass (encode + decode) on silence before timing, or `--verbose` for debug output. Omit the audio file to run a silence latency benchmark.

## Why three graphs?

Parakeet TDT's runtime decoding is autoregressive with duration-aware time advancement: each step samples a `(token, duration)` pair from the joint network, then advances the encoder frame pointer by `duration` (and only runs the LSTM prediction net when the token is not blank). That control flow lives in `ParakeetTDTGenerationMixin.generate`, not in `forward`, so `torch.export` cannot capture it as a single graph. The bundle exposes the three building blocks the runtime needs:

| Graph          | Inputs                                                              | Outputs                                            |
| -------------- | ------------------------------------------------------------------- | -------------------------------------------------- |
| `encoder`      | `input_features (B, T_audio, n_mels)`                               | `encoder_hidden_states (B, T_enc, decoder_hidden)` |
| `decoder_step` | `input_ids (B, 1)`, `hidden_state`, `cell_state`                    | `decoder_output`, `new_hidden_state`, `new_cell_state` |
| `joint`        | `decoder_hidden_states (B, 1, H)`, `encoder_hidden_states (B, 1, H)` | `logits (B, 1, vocab + len(durations))`            |

The encoder graph already includes `encoder_projector`, so the joint network's two addends share the same hidden size.

## Streaming

Parakeet TDT v3 is an *offline* FastConformer: attention is bidirectional over the whole utterance (`att_context_style: regular`), and `transformers` has no cache-aware Parakeet encoder. So streaming is done by **buffered inference** — re-run the whole encoder over a bounded `[left | chunk | right]` window each hop, consume only the chunk's encoder frames, and carry the transducer state across hops.

This is NVIDIA's own algorithm, from NeMo [`examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py). Its own example applies it to a non-cache-aware checkpoint, so this is a supported upstream mode rather than a workaround. `decoder_step` and `joint` were already the right shape — single-step with explicit LSTM state — so only the encoder's traced window changes.

### Exporting a streaming bundle

```sh
uv run export.py --streaming --dtype float16
```

| Flag | Description | Default |
| --- | --- | --- |
| `--streaming` | Fixed streaming window; records geometry in `metadata.json`. Ignores `--audio-seconds`. Mutually exclusive with `--dynamic`. | — |
| `--chunk-frames` | Encoder frames consumed per hop. Sets the emission cadence. Smaller costs more — see below. | `12` (0.96 s) |
| `--right-context-frames` | Lookahead. Latency is `(chunk + right) × 80 ms`. Must be ≥ `max(durations)`. | `12` (0.96 s) |
| `--left-context-frames` | Past context. Free — costs no latency. | `126` (10.08 s) |

Produces `parakeet-tdt-0.6b-v3_float16_streaming150/`, plus a `streaming` block in `metadata.json` that the runtime reads so callers don't have to restate the geometry.

The three frame counts are read only when `--streaming` is set, and `--audio-seconds` only when it isn't. The export warns rather than failing when it sees a flag the chosen shape mode doesn't use, so a stray `--chunk-frames` can't quietly produce a window you didn't ask for.

**Only a `--streaming` bundle can stream.** The window is fixed when the encoder is traced, so
it is chosen here and cannot be changed later; `startStream` rejects a `--static` or `--dynamic`
bundle and names the re-export command. Two geometries worth starting from:

| geometry | export flags | latency |
| --- | --- | --- |
| balanced (the default) | `--streaming` | 1.92 s |
| accuracy (NeMo's 10-2-2) | `--streaming --chunk-frames 25 --right-context-frames 25 --left-context-frames 125` | 4.00 s |

The `150` in the name is the window's **usable encoder frame count** — `left + chunk + right` = `126 + 12 + 12`, or 12.0 s at 80 ms per frame. It comes from the three flags above, so a different geometry produces a different suffix; it is not a model size or a latency figure.

**The chunk size is a throughput knob, not a quality one.** Every hop re-encodes the whole window to consume one chunk, so the encoder work per second of audio scales with `window / chunk`: halving the chunk doubles it.

### Frame arithmetic

One encoder frame is `hop_length × subsampling_factor` = 1280 samples / 16 kHz = **80 ms**. Everything follows from making the PCM window a whole number of encoder frames:

```
window_samples = W × 1280
mel frames     = 8W + 1      (frameCount is 1 + N/hop, torch.stft center=True)
encoder frames = W + 1       (ceil(mel/8): three stride-2 kernel-3 pad-1 convs)
usable frames  = W           (the last encoder frame covers the zero-padded remainder)
```

The export runs one forward pass and fails if the traced encoder disagrees with this arithmetic, because a one-frame slice error is 80 ms of audio and would drop or duplicate words at every chunk boundary.

### Ramp-up

At the start of a session the window holds less audio than it was traced for. That matters because the encoder is full-attention and non-causal: a frame's representation depends on how much audio surrounds it, so a window that grows hop by hop decodes the opening under a different regime than steady state, and the transducer ends up consuming a sequence stitched from mismatched representations.

So while audio is still arriving, the window is zero-filled to its traced size — the padding stands in for audio not yet received, and every hop presents the same extent. Frames are still only consumed where real audio backs them. The final flush is deliberately *not* padded: there the zeros would mean "no more speech", and masking them honestly is what cues the sentence-final token.

Cost is a one-time increase in front-end work per session, since the mel is computed across the full window during ramp-up rather than just the real prefix.

A `--dynamic` bundle cannot stream either: its time axis is symbolic, so there is no traced
window at all, and the `float32` dynamic encoder is separately unreliable on the GPU path at many
shapes.

The recorded block carries only what the runtime reads — left, chunk, right, the window's mel
frame count, and the sample rate, hop and subsampling factor it was traced at. Left context is
*derived* at load (`window − chunk − right`) and the recorded copy is cross-checked against it, so
a block that has been edited into disagreeing with itself fails to load rather than running a
geometry the file misdescribes.

### Running

```swift
import CoreAISpeech

let model = try await SpeechRecognitionModel(
    resourcesAt: URL(fileURLWithPath: "exports/parakeet-tdt-0.6b-v3_float16_streaming150"))

// This package does not capture audio. A host app owns AVAudioEngine, converts to
// mono float32 at model.sampleRate, and pushes buffers in.
let updates = try await model.startStream()

Task {
    do {
        for try await update in updates {
            switch update {
            case .partial(let segment):   render(segment.text)          // never retracts
            case .finalized(let segment): commit(segment.text, segment.startTime...segment.endTime)
            }
        }
    } catch {
        print("Streaming failed.")
    }
}

try await model.append(pcm: buffer)     // call from a Task, not an audio render callback
try await model.finishStream()
```

`startStream` takes no geometry: the bundle's window is the geometry, and `activeStreamingConfig` reports it — chunk, right, the derived left, and the theoretical latency. What the caller does control is `EndpointingConfig`: where a transcript is cut for display, and how long a gap has to be before the predictor is reset. Neither changes a tensor shape.

```bash
swift run -c release speech-recognizer --model <bundle> --audio-path audio.wav --stream
```

The CLI has no geometry flags, for the same reason — it prints the window the bundle records. `--endpoint-frames` sets the silence threshold, `--realtime` paces file input at 1× so reported latency is realistic, and `--deferred-decode` chunks the encoder but decodes once at the end; all three require `--stream`. `--reset-after-silence-frames` overrides the predictor reset and deliberately does *not*, since it applies to offline transcription too.

Partials are rewritten in place on a terminal, and dropped entirely when stdout is redirected so that piped output stays diffable.

### Segments and endpointing

A segment closes when the decoder has gone `silenceFrames` (default 10, so 0.8 s) without emitting, or — past the `maxSegmentFrames` cap (default 375, 30 s) — at the first pause after it. Silence is counted in *frames of audio the decoder skipped*, duration-weighted: a blank carrying duration 4 contributes 4 frames, so the threshold means 0.8 s of audio rather than "one hop produced nothing".

Closing a segment is a display boundary: the transducer state carries straight across it. Zeroing the predictor at *every* endpoint instead cost roughly 3.8 s of dropped audio each time, while it re-established context from a cold start.

A long gap is the exception. `resetAfterSilenceFrames` (default 40, so 3.2 s) restores the predictor's start condition — a zeroed LSTM plus the blank as the previous label, which is all-zeros because the blank's embedding row is itself zero. Without it, a predictor that has emitted a sentence-final token and then consumed tens of seconds of blanks resists re-entering an emitting state: the duration head jumps across the re-onset and the resuming utterance loses its opening words.

That loss reproduces identically offline and under HF's own `generate()`, so it is the checkpoint's behaviour rather than the chunking's, and the reset is a streaming-side correction to it. It is applied inside the decode loop rather than at a hop boundary, so `--deferred-decode` runs the same rule and still agrees with a live stream. Offline transcription defaults to `0` to stay reference-exact — `--parity-test` compares tokens and transcript against PyTorch traces — so pass `--reset-after-silence-frames` to opt an offline run in.

### Gating the mic

Endpointing and `resetAfterSilenceFrames` advance only when hops run, so both assume the host pushes audio continuously — including through pauses.

An app that gates the mic with a voice-activity detector must not simply withhold audio: that freezes the session rather than pausing it, so the open segment never finalizes and the predictor reset never fires. Call `finishStream()` at the pause and `startStream()` again when speech resumes, after a hangover long enough that an inter-word gap does not split one utterance. Flush a few hundred milliseconds of pre-roll on the transition, since a detector fires after onset. `hop` and `segmentIndex` are per-session, so timestamps restart and the app supplies its own offset.

### `--deferred-decode` is the correctness test

It chunks the encoder but concatenates the outputs and decodes in one pass. This is NeMo's `simulated` flag under a name that says what changes — upstream describes it as "encoder is evaluated on chunks, output is concatenated and decoded at one step … expected to provide the same results".

That expectation is what makes it a test. Because it shares the encoder path but not the incremental decode, `--deferred-decode` and plain `--stream` must produce byte-identical transcripts. Any difference is a bug in the state carry, the duration-overshoot carry, or the frame partition — and comparing two strings finds those without needing a quality judgement. Note it defers *all* decoding, so it emits no partials and never exercises endpointing; it also holds every consumed frame in memory (~32 KB per second of audio), so it is for short files rather than long sessions.

### Streaming removes the length limit

The offline path silently pads or truncates PCM to the traced window, so `transcribe` on a static bundle covers only as much audio as that window holds — longer input is dropped without an error. A streaming bundle has no such bound: it slides the same fixed window across input of any length, so a session transcribes an arbitrarily long stream.

[^1]: [TDT paper](https://arxiv.org/abs/2304.06795) · [Parakeet TDT v3 paper](https://arxiv.org/abs/2509.14128) · [HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
