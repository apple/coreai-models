# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for the EmbeddingGemma 2 image path (``embedding_gemma2_vision.py``).

* Tiny-config parity against HF ``Gemma4VisionModel`` + ``Gemma4MultimodalEmbedder``
  (present in the repo's transformers; no checkpoint needed).
* Host preprocessing helpers against HF ``Gemma4ImageProcessor``.
* Scatter / sequence assembly against HF ``masked_scatter`` semantics.
* ``torch.export`` at the static default budget (280 soft tokens / 2520 patches).
* Real-weight parity (soft tokens and end-to-end image embedding) against HF
  ``EmbeddingGemma2Model``. Needs transformers >= 5.19 and the local snapshot; skipped
  otherwise. Run with an overlay, e.g.::

      uv run --no-sync --with "transformers>=5.19" --with "torchvision==0.24.*" \\
          python -m pytest -q -p no:rerunfailures -p no:xdist -o addopts="" <this file>
"""

import math

import pytest
import torch
import torch.nn.functional as F

from coreai_models.models.macos.embedding_gemma2_vision import (
    EG2_BOI_TOKEN_ID,
    EG2_BOS_TOKEN_ID,
    EG2_EOI_TOKEN_ID,
    EG2_EOS_TOKEN_ID,
    EG2_IMAGE_TOKEN_ID,
    EmbeddingGemma2ImageEncoder,
    EmbeddingGemma2VisionConfig,
    aspect_ratio_preserving_size,
    build_image_input_ids,
    build_image_inputs_embeds,
    pad_to_budget,
    patch_position_ids,
    patchify,
    preprocess_resized_image,
    scatter_soft_tokens,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    TINY_VISION_CONFIG as TINY,
)
from tests.test_model_units.test_models._embedding_gemma2_common import (
    _cos,
    _st_mean_pool_normalize,
)

gemma4_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
gemma4_config = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")


def _hf_tiny_vision(cfg: EmbeddingGemma2VisionConfig):
    # transformers >= 5.19 names the vision RoPE "axial"; older 5.x builds call the same
    # (mathematically identical) per-axis RoPE "default".
    rot = gemma4_modeling.Gemma4VisionRotaryEmbedding
    rope_type = "axial" if hasattr(rot, "compute_axial_rope_parameters") else "default"
    hf_cfg = gemma4_config.Gemma4VisionConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        global_head_dim=cfg.head_dim,
        patch_size=cfg.patch_size,
        pooling_kernel_size=cfg.pooling_kernel_size,
        position_embedding_size=cfg.position_embedding_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_parameters={"rope_type": rope_type, "rope_theta": cfg.rope_theta},
        hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=False,
        standardize=False,
    )
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = gemma4_modeling.Gemma4VisionModel(hf_cfg).eval()
    # HF init leaves norms at 1 / zeros; randomize every float tensor so the test bites.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    return hf_cfg, model


def _load_from_hf_vision(ours: EmbeddingGemma2ImageEncoder, hf_vision, hf_embedder):
    state = {}
    for k, v in hf_vision.state_dict().items():
        local = EmbeddingGemma2ImageEncoder.map_hf_key("vision_tower." + k)
        if local is not None and local in dict(ours.named_parameters()):
            state[local] = v
    for k, v in hf_embedder.state_dict().items():
        state["embed_vision." + k] = v
    ours.load_state_dict(state)


def _random_inputs(cfg: EmbeddingGemma2VisionConfig, grids: list[tuple[int, int]], seed: int = 0):
    """Random patches for images whose block grid is ``(bh, bw)`` blocks, padded to budget."""
    g = torch.Generator().manual_seed(seed)
    k = cfg.pooling_kernel_size
    pv, pos, counts = [], [], []
    for bh, bw in grids:
        ph, pw = bh * k, bw * k
        img = torch.rand(3, ph * cfg.patch_size, pw * cfg.patch_size, generator=g)
        patches = patchify(img, cfg.patch_size)
        positions = patch_position_ids(ph, pw)
        patches, positions = pad_to_budget(patches, positions, cfg.max_patches)
        pv.append(patches)
        pos.append(positions)
        counts.append(bh * bw)
    return torch.stack(pv), torch.stack(pos), counts


# --- Tiny-config parity vs HF ---


def test_tiny_soft_tokens_match_hf(monkeypatch) -> None:
    # The directory conftest forces USE_HF_IMPL=true; run the Core AI lowering path of the SDPA
    # primitive (bool-mask -> -1e4 bias), which is what gets exported.
    monkeypatch.setenv("USE_HF_IMPL", "false")
    hf_cfg, hf = _hf_tiny_vision(TINY)
    text_cfg = gemma4_config.Gemma4TextConfig(hidden_size=TINY.text_hidden_size)
    torch.manual_seed(1)
    embedder = gemma4_modeling.Gemma4MultimodalEmbedder(hf_cfg, text_cfg).eval()
    ours = EmbeddingGemma2ImageEncoder(TINY).eval()
    _load_from_hf_vision(ours, hf, embedder)
    pixel_values, position_ids, counts = _random_inputs(TINY, [(1, 5), (4, 3)])
    with torch.no_grad():
        ref = embedder(
            hf(pixel_values=pixel_values, pixel_position_ids=position_ids).last_hidden_state
        )
        got = ours(pixel_values, position_ids)
    assert got.shape == (2, TINY.max_soft_tokens, TINY.text_hidden_size)
    # HF strips padding slots and concatenates images; ours keeps the static budget.
    got_valid = torch.cat([got[i, :n] for i, n in enumerate(counts)])
    torch.testing.assert_close(got_valid, ref, rtol=1e-5, atol=1e-5)
    for i, n in enumerate(counts):
        assert torch.count_nonzero(got[i, n:]) == 0


def test_rejects_wrong_patch_count() -> None:
    ours = EmbeddingGemma2ImageEncoder(TINY)
    with pytest.raises(ValueError, match="expected"):
        ours(torch.zeros(1, TINY.max_patches - 9, TINY.patch_dim), torch.zeros(1, 99, 2).long())


# --- Host preprocessing vs Gemma4ImageProcessor ---


def _hf_aspect_fn():
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        get_aspect_ratio_preserving_size,
    )

    return get_aspect_ratio_preserving_size


@pytest.mark.parametrize("budget", [70, 140, 280, 560, 1120])
def test_aspect_ratio_size_matches_hf(budget: int) -> None:
    hf_fn = _hf_aspect_fn()
    sizes = [(300, 451), (451, 300), (512, 512), (48, 48), (20, 20), (1080, 1920), (1920, 1080)]
    sizes += [(50, 2000), (2000, 50), (31, 1000), (3000, 40), (17, 33), (4032, 3024)]
    for h, w in sizes:
        try:
            ref = hf_fn(
                height=h, width=w, patch_size=16, max_patches=budget * 9, pooling_kernel_size=3
            )
        except ValueError:
            with pytest.raises(ValueError):
                aspect_ratio_preserving_size(h, w, 16, budget, 3)
            continue
        assert aspect_ratio_preserving_size(h, w, 16, budget, 3) == ref, (h, w)


def test_preprocess_matches_hf_processor_without_resize() -> None:
    """At the target size HF skips the resize, so patching/positions/padding must be exact."""
    np = pytest.importorskip("numpy")
    try:
        from transformers.models.gemma4.image_processing_pil_gemma4 import (
            Gemma4ImageProcessorPil as Proc,
        )
    except ImportError:
        pytest.skip("Gemma4ImageProcessorPil unavailable")
    pil = pytest.importorskip("PIL.Image")
    # Fixed points of the resize policy (the policy is not idempotent in general, e.g.
    # 480x240 -> 1104x528 -> 1152x528), so HF leaves these untouched.
    for h, w in [(624, 960), (960, 624), (768, 768), (576, 1056)]:
        assert aspect_ratio_preserving_size(h, w) == (h, w)
        arr = (np.random.default_rng(h + w).integers(0, 256, size=(h, w, 3))).astype("uint8")
        out = Proc(max_soft_tokens=280)(images=[pil.fromarray(arr)], return_tensors="pt")
        pv, pos, n = preprocess_resized_image(torch.from_numpy(arr).permute(2, 0, 1))
        assert n == int(out["num_soft_tokens_per_image"][0])
        torch.testing.assert_close(pv, out["pixel_values"].float(), rtol=0, atol=1e-7)
        assert torch.equal(pos, out["image_position_ids"].long())


# --- Sequence assembly ---


def test_build_image_input_ids_layout() -> None:
    ids = build_image_input_ids(3)
    assert ids == [
        EG2_BOS_TOKEN_ID,
        EG2_BOI_TOKEN_ID,
        EG2_IMAGE_TOKEN_ID,
        EG2_IMAGE_TOKEN_ID,
        EG2_IMAGE_TOKEN_ID,
        EG2_EOI_TOKEN_ID,
        EG2_EOS_TOKEN_ID,
    ]


def test_scatter_matches_masked_scatter() -> None:
    torch.manual_seed(0)
    embeds = torch.randn(2, 9, 8)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[0, 2:6] = True
    mask[1, 1:4] = True
    soft = torch.randn(2, 6, 8)  # extra trailing rows must be ignored
    got = scatter_soft_tokens(embeds, mask, soft)
    for b, n in ((0, 4), (1, 3)):
        ref = embeds[b : b + 1].masked_scatter(mask[b : b + 1, :, None], soft[b, :n])
        torch.testing.assert_close(got[b : b + 1], ref, rtol=0, atol=0)


def test_build_image_inputs_embeds() -> None:
    table = torch.randn(300000, 4)

    def embed(ids):
        return F.embedding(ids, table) * 2.0

    soft = torch.randn(2, 5, 4)
    embeds, mask, ids = build_image_inputs_embeds(embed, soft, [5, 3])
    assert ids.shape == (2, 9) and mask[1].tolist() == [1] * 7 + [0, 0]
    torch.testing.assert_close(embeds[0, 2:7], soft[0])
    torch.testing.assert_close(embeds[1, 2:5], soft[1, :3])
    torch.testing.assert_close(embeds[1, 0], table[EG2_BOS_TOKEN_ID] * 2.0)
    torch.testing.assert_close(embeds[1, 5], table[EG2_EOI_TOKEN_ID] * 2.0)


# --- torch.export at the static default budget ---


def test_torch_export_default_budget_static_shape(monkeypatch) -> None:
    monkeypatch.setenv("USE_HF_IMPL", "false")  # the lowering path that gets exported
    cfg = EmbeddingGemma2VisionConfig(num_hidden_layers=2)  # real widths, 2 layers for speed
    torch.manual_seed(0)
    model = EmbeddingGemma2ImageEncoder(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    k = cfg.pooling_kernel_size
    img = torch.randint(0, 256, (3, 15 * k * 16, 18 * k * 16), dtype=torch.uint8)
    pixel_values, position_ids, n = preprocess_resized_image(img)
    assert pixel_values.shape == (1, 2520, 768) and n == 270
    with torch.no_grad():
        ep = torch.export.export(model, (pixel_values, position_ids))
        eager = model(pixel_values, position_ids)
        exported = ep.module()(pixel_values, position_ids)
    assert exported.shape == (1, 280, 512)
    torch.testing.assert_close(exported, eager, rtol=1e-5, atol=1e-5)
    # No dynamic dims and no data-dependent ops (nonzero / masked_select / item) in the graph.
    targets = {str(node.target) for node in ep.graph.nodes if node.op == "call_function"}
    data_dependent = ("aten.nonzero", "aten.masked_select", "aten.item", "aten._local_scalar_dense")
    assert not any(op in t for t in targets for op in data_dependent), sorted(targets)


# --- Real weights (transformers >= 5.19 + local snapshot) ---


def _synthetic_images():
    """Deterministic natural-ish RGB images (smooth gradients + shapes + noise)."""
    np = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    out = []
    for h, w in [(300, 451), (180, 640), (777, 333)]:
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        img = np.stack(
            [
                128 + 100 * np.sin(xx / (17 + h % 7)),
                128 + 100 * np.cos(yy / 23),
                255 * (xx + yy) / (h + w),
            ],
            -1,
        )
        cy, cx, r = h / 2, w / 3, min(h, w) / 4
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = [230, 40, 40]
        img += rng.normal(0, 8, img.shape)
        out.append(pil.fromarray(np.clip(img, 0, 255).astype("uint8")))
    return out


@pytest.fixture(scope="module")
def eg2_hf(embeddinggemma2_snapshot):
    pytest.importorskip(
        "transformers.models.embedding_gemma2.modeling_embedding_gemma2",
        reason="needs transformers>=5.19",
    )
    from transformers import AutoProcessor
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2Model,
    )

    snapshot = str(embeddinggemma2_snapshot)
    processor = AutoProcessor.from_pretrained(snapshot)
    model = EmbeddingGemma2Model.from_pretrained(
        snapshot, dtype=torch.float32, audio_config=None
    ).eval()
    # Build ours on the SDPA lowering path that gets exported (not the HF-eager shim).
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("USE_HF_IMPL", "false")
        ours = EmbeddingGemma2ImageEncoder.from_pretrained(snapshot, dtype=torch.float32)
    return processor, model, ours


@pytest.fixture(scope="module")
def eg2_runs(eg2_hf):
    processor, model, ours = eg2_hf
    runs = []
    for img in _synthetic_images():
        inputs = processor(images=[img], return_tensors="pt")
        # The processor drops num_soft_tokens_per_image; count the placeholders instead.
        n = int((inputs["input_ids"][0] == EG2_IMAGE_TOKEN_ID).sum())
        with torch.no_grad():
            ref_soft = model.get_image_features(
                inputs["pixel_values"], inputs["image_position_ids"], return_dict=True
            ).pooler_output[0]
            got_soft = ours(inputs["pixel_values"], inputs["image_position_ids"])
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_position_ids=inputs["image_position_ids"],
            )
        runs.append((inputs, n, ref_soft, got_soft, out.last_hidden_state))
    return runs


def test_real_soft_tokens_match_hf(eg2_runs) -> None:
    for inputs, n, ref_soft, got_soft, _ in eg2_runs:
        assert ref_soft.shape == (n, 512)
        assert got_soft.shape == (1, 280, 512)
        cos = _cos(got_soft[0, :n], ref_soft)
        assert float(cos.min()) >= 0.99999, float(cos.min())
        assert torch.count_nonzero(got_soft[0, n:]) == 0
        assert inputs["input_ids"][0].tolist() == build_image_input_ids(n)


def test_real_end_to_end_image_embedding_matches_hf(eg2_hf, eg2_runs) -> None:
    _, model, _ = eg2_hf
    lm = model.language_model
    for inputs, n, _, got_soft, ref_hidden in eg2_runs:
        with torch.no_grad():
            embeds, mask, ids = build_image_inputs_embeds(lm.embed_tokens, got_soft, n)
            hidden = lm(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        assert torch.equal(ids, inputs["input_ids"])
        got = _st_mean_pool_normalize(hidden, mask)
        ref = _st_mean_pool_normalize(ref_hidden, inputs["attention_mask"])
        for dim in (768, 512, 256, 128):
            c = F.cosine_similarity(
                F.normalize(got[:, :dim].double(), dim=-1),
                F.normalize(ref[:, :dim].double(), dim=-1),
            )
            assert float(c.min()) >= 0.99999, (dim, float(c.min()))
        assert math.isclose(float(got.norm()), 1.0, rel_tol=1e-5)
