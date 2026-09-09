# SAM 3 Video Segmentation

SAM 3 detects objects matching a text prompt on every frame of a video and links them
into masklets with stable object IDs.[^1]

For single-image segmentation, see [`models/sam3`](../sam3/README.md).

## How this export is structured

`transformers.Sam3VideoModel.forward` doesn't take tensors. It takes a mutable
`Sam3VideoInferenceSession` (dicts, sets, and an object registry that renumbers itself
partway through a video) and returns `dict[int, Tensor]`. `torch.export` can't trace
that. The session is host state, not graph state.

Therefore, the `.aimodel` holds seven tensor kernels. The session, the memory ring buffer,
and all the tracking heuristics (NMS, detection-to-track association, hotstart, keep-alive,
reconditioning, occlusion suppression) will be handled by the Swift runtime.

| Function | Inputs | Outputs | Runs |
|---|---|---|---|
| `image_encode` | `pixel_values` | `last_hidden_state` | once per frame |
| `text_encode` | `input_ids`, `attention_mask` | `text_features` | once per prompt, per video |
| `detect` | `last_hidden_state`, `text_features`, `attention_mask` | `pred_masks`, `pred_boxes`, `pred_logits`, `presence_logits`, `semantic_seg` | P times per frame |
| `tracker_encode` | `last_hidden_state` | `vision_feat_{0,1,2}`, `vision_pos_{0,1,2}` | once per frame |
| `tracker_step` | vision feats, fixed memory bank, key mask | `pred_masks`, `high_res_masks`, `object_pointer`, `object_score_logits` | M times per frame |
| `memory_encode` | `vision_feat_2`, `mask_logits`, `object_score_logits`, `binarize_mask` | `maskmem_features`, `maskmem_pos_enc` | M times per frame |
| `tracker_mask_init` | vision feats, `mask_input` | `object_pointer` | when a new object appears |

P is the number of text prompts, M the number of tracked objects. There is one ViT pass
per frame, shared by the detector and the tracker. See [Multiple prompts](#multiple-prompts).

### The memory bank

HF builds the tracker's memory with a variable-length `torch.cat`, so the length changes
every frame and for every object. This export uses a fixed layout plus an additive key
mask:

```
K = spatial_slots * H*W  +  ptr_slots * (hidden_dim / mem_dim)
  = 10 * 5184            +  24 * 4                              = 51,936
```

`spatial_slots=10` isn't a tuning knob. HF can populate at most `max_cond_frame_num` (4)
conditioning memories plus `num_maskmem - 1` (6) recent ones, and the export refuses to
run if you size it below that. `ptr_slots=24` is a budget, since HF's conditioning-frame
object-pointer branch has no cap. On overflow the runtime logs a warning and drops the
temporally furthest pointer.

Padding costs nothing numerically: cross-attention doesn't care about key order, RoPE
repeats the same per-slot pattern across all spatial slots, and masked slots contribute
zero. `python/tests/test_model_units/test_export/test_video_export.py` checks that the
padded bank matches a variable-length one.

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

### Gated access

SAM3 is a gated model on [Hugging Face](https://huggingface.co/facebook/sam3). Accept
the terms of the [license](https://huggingface.co/facebook/sam3), generate an HF token,
and add it to your machine before exporting.

```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Export

```sh
uv run models/sam3_video/export.py
```

Saves to `<repo-root>/exports/sam3_video_<dtype>/`, a bundle directory holding the
`.aimodel`, a `tokenizer/` folder, and a `metadata.json` (`kind: video_segmenter`). The
`runtime` block in that metadata records the slot geometry the host has to pack to, and
the `tracking` block records the checkpoint's heuristic thresholds.

**Options:**

| Flag | Description | Default |
|---|---|---|
| `--dtype` | `float16` or `float32` | `float16` |
| `--image-size` | Input resolution. Must match the checkpoint. | `1008` |
| `--spatial-slots` | Spatial memory slots per object | `10` |
| `--ptr-slots` | Object-pointer slots per object | `24` |
| `--output-dir` | Bundle destination | `<repo-root>/exports/` |
| `--output-name` | Custom bundle directory name | derived |
| `--overwrite` | Replace an existing bundle | off |
| `--include-debug-info` | Embed conversion debug info | off |
| `--dry-run` | Print the resolved config and exit | off |

### Multiple prompts

`add_text_prompt` takes a list, and duplicate strings collapse to one prompt id:

```python
processor.add_text_prompt(inference_session=session, text=["person", "dog"])
```

Objects stay attributed to whichever prompt found them, and `postprocess_outputs`
returns that grouping:

```python
results = processor.postprocess_outputs(session, model_outputs)
results["prompt_to_obj_ids"]   # {"person": [0, 1], "dog": [2]}
```

Non-overlap constraints are applied within a prompt group, not across groups, so two
prompts can legitimately return overlapping masks for the same pixels.

Each extra prompt costs one more `detect` pass per frame. `image_encode`,
`tracker_encode`, and the tracker are shared, and `text_encode` runs once per prompt for
the whole video.

| Function | Scales with prompts |
|---|---|
| `image_encode` | no |
| `detect` | yes |
| `tracker_encode` | no |
| `text_encode` | once per prompt, per video |

## The Swift runtime (PENDING: this is design only)

`swift/Sources/CoreAIVideoSegmenter` mimics the HF logic in Swift, and the
`video-segmenter` tool drives it end to end:

```sh
swift run video-segmenter --model exports/sam3_video_float16 \
    --video clip.mp4 --prompt person --prompt dog --output out.mp4
```

That writes an mp4 with each object's mask, box, and `#id prompt score` caption composited
over the original frames. Colors are keyed on the object id, so a track keeps its color
across the whole video.

The Swift API equivalent:

```swift
let segmenter = try await VideoSegmenter(resourcesAt: "exports/sam3_video_float16")

// One call, annotated video out.
try await segmenter.renderAnnotatedVideo(
    from: sourceURL, prompts: ["person"], to: destinationURL)

// Or consume per-frame results.
for try await frame in segmenter.segment(videoAt: sourceURL, prompts: ["person"]) {
    print(frame.frameIndex, frame.objects.map(\.id))
}
```

Frames arrive `hotstart_delay` (15) behind the decoder, because a track removed on frame
20 must never have been shown on frame 8. That means 15 decoded frames are held as well,
about 124 MB at 1080p. `--hotstart-delay 0` turns off both the delay and the removal rules
that need it.

## Supported models

| Model | Parameters |
|---|---|
| facebook/sam3 | 848M, detector plus tracker |

[^1]: [Paper](https://arxiv.org/abs/2511.16719) · [HuggingFace](https://huggingface.co/facebook/sam3)
