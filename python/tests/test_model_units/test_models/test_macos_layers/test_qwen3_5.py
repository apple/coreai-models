# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Qwen3.5 model parity with HuggingFace."""

import asyncio
import copy

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention as HFAttention,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer as HFDecoderLayer,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet as HFGatedDeltaNet,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM,
)

from coreai_models._constants import (
    CONV_STATES_NAME,
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    RECURRENT_STATES_NAME,
    VALUE_CACHE_NAME,
)
from coreai_models.export import pipeline as export_pipeline
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.qwen3_5 import (
    Attention,
    GatedDeltaNet,
    Qwen3_5ForCausalLM,
    TransformerBlock,
)
from coreai_models.primitives.macos.mlp import MLP
from tests._runner_infra._deps import _hf_hub_reachable
from tests._runner_infra.testing_utils import ForCausalLMTestBase, run_compare_coreai


def _make_qwen3_5_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    num_hidden_layers: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    head_dim: int = 16,
) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=head_dim,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    config.layer_types = ["full_attention", "linear_attention"]
    return config


def _make_qwen3_5_moe_config() -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=100,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config.layer_types = ["full_attention", "linear_attention"]
    return config


def _make_component_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        hidden_size=4,
        head_dim=16,
        intermediate_size=6,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def _load_via_mutate(
    our: torch.nn.Module,
    hf_sd: dict[str, torch.Tensor],
    config: Qwen3_5TextConfig,
    layer_idx: int,
    submodule: str,
) -> None:
    """Fuse HF per-projection weights the production way, then load onto ``our``.

    Our modules take fused projections (``qkv_gate_proj``, ``in_proj_all``,
    ``conv_weight``) with no HF counterpart, so rather than duplicate the packing
    arithmetic here -- where it could drift from the loader and hide a real bug -- run
    the HF weights through ``Qwen3_5ForCausalLM._mutate_state_dict`` and load the result.

    ``_mutate_state_dict`` is layer-keyed and reads head counts off the model, so host
    it on a meta-device model whose ``layer_idx`` is of the type under test.
    """
    layer_type = "full_attention" if submodule == "self_attn" else "linear_attention"
    host_config = copy.deepcopy(config)
    host_config.num_hidden_layers = layer_idx + 1
    host_config.layer_types = [layer_type] * host_config.num_hidden_layers
    host = Qwen3_5ForCausalLM(host_config, model_device="meta")

    prefix = f"model.layers.{layer_idx}.{submodule}."
    sd = {f"{prefix}{k}": v.clone() for k, v in hf_sd.items()}
    host._mutate_state_dict(sd)
    our.load_state_dict(
        {k.removeprefix(prefix): v for k, v in sd.items()}, strict=True, assign=True
    )


def _setup_attention_weights(our: Attention, hf: HFAttention, config: Qwen3_5TextConfig) -> None:
    """Give HF its per-projection weights and ours the fused equivalent."""
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim
    hidden = config.hidden_size
    has_bias = config.attention_bias

    hf_sd: dict[str, torch.Tensor] = {}
    for proj, shape in [
        ("q_proj", (n_heads * head_dim * 2, hidden)),
        ("k_proj", (n_kv * head_dim, hidden)),
        ("v_proj", (n_kv * head_dim, hidden)),
        ("o_proj", (hidden, n_heads * head_dim)),
    ]:
        w = torch.randn(*shape)
        getattr(hf, proj).weight = torch.nn.Parameter(w.clone())
        hf_sd[f"{proj}.weight"] = w
        if has_bias:
            b = torch.randn(shape[0])
            getattr(hf, proj).bias = torch.nn.Parameter(b.clone())
            hf_sd[f"{proj}.bias"] = b

    for norm in ("q_norm", "k_norm"):
        w = torch.randn(head_dim)
        getattr(hf, norm).weight = torch.nn.Parameter(w.clone())
        hf_sd[f"{norm}.weight"] = w

    _load_via_mutate(our, hf_sd, config, layer_idx=our.layer_idx, submodule="self_attn")


def _setup_gated_delta_net_weights(
    our: GatedDeltaNet, hf: HFGatedDeltaNet, config: Qwen3_5TextConfig
) -> None:
    """As ``_setup_attention_weights``, but for the gated delta net."""
    hidden = config.hidden_size
    num_v = config.linear_num_value_heads
    num_k = config.linear_num_key_heads
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    key_dim = head_k * num_k
    value_dim = head_v * num_v
    conv_kernel = config.linear_conv_kernel_dim
    conv_dim = key_dim * 2 + value_dim

    hf_sd: dict[str, torch.Tensor] = {}
    for attr, shape in [
        ("in_proj_qkv", (key_dim * 2 + value_dim, hidden)),
        ("in_proj_z", (value_dim, hidden)),
        ("in_proj_b", (num_v, hidden)),
        ("in_proj_a", (num_v, hidden)),
        ("out_proj", (hidden, value_dim)),
    ]:
        w = torch.randn(*shape)
        getattr(hf, attr).weight = torch.nn.Parameter(w.clone())
        hf_sd[f"{attr}.weight"] = w

    conv_w = torch.randn(conv_dim, 1, conv_kernel)
    hf.conv1d.weight = torch.nn.Parameter(conv_w.clone())
    hf_sd["conv1d.weight"] = conv_w

    norm_w = torch.randn(head_v)
    hf.norm.weight = torch.nn.Parameter(norm_w.clone())
    hf_sd["norm.weight"] = norm_w

    dt = torch.randn(num_v)
    hf.dt_bias = torch.nn.Parameter(dt.clone())
    hf_sd["dt_bias"] = dt

    A = torch.randn(num_v)
    hf.A_log = torch.nn.Parameter(A.clone())
    hf_sd["A_log"] = A

    _load_via_mutate(our, hf_sd, config, layer_idx=our.layer_idx, submodule="linear_attn")


def _setup_layernorm_weights(
    our_block: torch.nn.Module, hf_block: torch.nn.Module, hidden_size: int
) -> None:
    for attr in ("input_layernorm", "post_attention_layernorm"):
        w = torch.randn(hidden_size)
        getattr(our_block, attr).weight = torch.nn.Parameter(w.clone())
        getattr(hf_block, attr).weight = torch.nn.Parameter(w.clone())


def _setup_mlp_weights(
    our_block: torch.nn.Module,
    hf_block: torch.nn.Module,
    intermediate_size: int,
    hidden_size: int,
) -> None:
    for proj, shape in [
        ("gate_proj", (intermediate_size, hidden_size)),
        ("up_proj", (intermediate_size, hidden_size)),
        ("down_proj", (hidden_size, intermediate_size)),
    ]:
        w = torch.randn(*shape)
        getattr(our_block.mlp, proj).weight = torch.nn.Parameter(w.clone())
        getattr(hf_block.mlp, proj).weight = torch.nn.Parameter(w.clone())


@pytest.mark.parametrize(
    "heads, layer_idx, attention_bias, partial_rotary_factor",
    [
        ((1, 1), 0, False, 0.5),
        ((8, 8), 1, True, 1.0),
        ((8, 4), 0, False, 1.0),
    ],
)
class TestAttention:
    @staticmethod
    def _config(
        heads: tuple[int, int],
        layer_idx: int,
        attention_bias: bool,
        partial_rotary_factor: float,
    ) -> Qwen3_5TextConfig:
        config = _make_component_config()
        n_heads, n_kv = heads
        config.num_attention_heads = n_heads
        config.num_key_value_heads = n_kv
        config.attention_bias = attention_bias
        config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        config.num_hidden_layers = max(layer_idx + 1, 2)
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        return config

    def test_hf(
        self,
        heads: tuple[int, int],
        layer_idx: int,
        attention_bias: bool,
        partial_rotary_factor: float,
    ) -> None:
        config = self._config(heads, layer_idx, attention_bias, partial_rotary_factor)

        our_attn = Attention(config=config, layer_idx=layer_idx)
        hf_attn = HFAttention(config=config, layer_idx=layer_idx)
        _setup_attention_weights(our_attn, hf_attn, config)

        batch_size, seq_len = 1, 10
        offset = 3
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch_size, -1
        )
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hf_rotary = Qwen3_5TextRotaryEmbedding(config)
        cos, sin = hf_rotary(x, position_ids)
        hf_out = hf_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )[0]

        our_out = our_attn(x, position_ids)

        atol = 1e-3 if heads == (1, 1) else 1e-5
        torch.testing.assert_close(our_out, hf_out, atol=atol, rtol=0)

    def test_coreai(
        self,
        heads: tuple[int, int],
        layer_idx: int,
        attention_bias: bool,
        partial_rotary_factor: float,
    ) -> None:
        config = self._config(heads, layer_idx, attention_bias, partial_rotary_factor)

        model = Attention(config=config, layer_idx=layer_idx)
        hf_attn = HFAttention(config=config, layer_idx=layer_idx)
        _setup_attention_weights(model, hf_attn, config)

        batch_size, seq_len = 1, 10
        offset = 3
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch_size, -1
        )
        run_compare_coreai(model=model, inputs=(x, position_ids), atol=5e-3, rtol=5e-4)


class TestGatedDeltaNet:
    # Tolerances track low-precision noise, not implementation divergence: at fp32 the
    # two agree to ~3e-6, and at fp16/bf16 each deviates from its own fp32 result by
    # *more* than they differ from each other.
    @pytest.mark.parametrize(
        "precision, atol, rtol",
        [
            (torch.float32, 1e-3, 1e-3),
            (torch.float16, 2e-2, 1e-2),
            (torch.bfloat16, 1e-1, 5e-2),
        ],
    )
    def test_hf(self, precision: torch.dtype, atol: float, rtol: float) -> None:
        # Seeded: the low-precision tolerances sit near the noise floor for some draws.
        torch.manual_seed(0)
        config = _make_component_config()
        batch_size, seq_len = 1, 4
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        our = GatedDeltaNet(config=config, layer_idx=0)
        hf = HFGatedDeltaNet(config=config, layer_idx=0)
        _setup_gated_delta_net_weights(our, hf, config)

        our = our.to(precision)
        hf = hf.to(precision)
        x = x.to(precision)

        hf_out = hf(x)
        our_out = our(x)

        torch.testing.assert_close(our_out, hf_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("heads", [(1, 1), (8, 4)])
@pytest.mark.parametrize("layer_idx", [0, 1])
class TestTransformerBlockFullAttention:
    def test_hf(self, heads: tuple[int, int], layer_idx: int) -> None:
        config = _make_component_config()
        n_heads, n_kv = heads
        config.num_attention_heads = n_heads
        config.num_key_value_heads = n_kv
        config.num_hidden_layers = 2
        config.layer_types = ["full_attention", "full_attention"]

        our_block = TransformerBlock(config=config, layer_idx=layer_idx)
        hf_block = HFDecoderLayer(config=config, layer_idx=layer_idx)

        _setup_attention_weights(our_block.self_attn, hf_block.self_attn, config)
        _setup_layernorm_weights(our_block, hf_block, config.hidden_size)
        _setup_mlp_weights(our_block, hf_block, config.intermediate_size, config.hidden_size)

        assert isinstance(our_block.mlp, MLP)

        batch_size, seq_len = 1, 10
        offset = 3
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch_size, -1
        )
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hf_rotary = Qwen3_5TextRotaryEmbedding(config)
        cos, sin = hf_rotary(x, position_ids)

        hf_out = hf_block(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            position_ids=position_ids,
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        our_out = our_block(x, position_ids)

        atol = 4e-2 if heads == (1, 1) else 4e-3
        torch.testing.assert_close(our_out, hf_out, atol=atol, rtol=0)

    def test_coreai(self, heads: tuple[int, int], layer_idx: int) -> None:
        config = _make_component_config()
        n_heads, n_kv = heads
        config.num_attention_heads = n_heads
        config.num_key_value_heads = n_kv
        config.num_hidden_layers = 2
        config.layer_types = ["full_attention", "full_attention"]

        model = TransformerBlock(config=config, layer_idx=layer_idx)
        hf_block = HFDecoderLayer(config=config, layer_idx=layer_idx)
        _setup_attention_weights(model.self_attn, hf_block.self_attn, config)
        _setup_layernorm_weights(model, hf_block, config.hidden_size)
        _setup_mlp_weights(model, hf_block, config.intermediate_size, config.hidden_size)

        batch_size, seq_len = 1, 10
        offset = 3
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(
            batch_size, -1
        )
        run_compare_coreai(model=model, inputs=(x, position_ids), atol=5e-3, rtol=5e-4)


class TestTransformerBlockLinearAttention:
    def test_hf(self) -> None:
        config = _make_component_config()
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        layer_idx = 0

        our_block = TransformerBlock(config=config, layer_idx=layer_idx)
        hf_block = HFDecoderLayer(config=config, layer_idx=layer_idx)

        _setup_gated_delta_net_weights(our_block.linear_attn, hf_block.linear_attn, config)
        _setup_layernorm_weights(our_block, hf_block, config.hidden_size)
        _setup_mlp_weights(our_block, hf_block, config.intermediate_size, config.hidden_size)

        assert isinstance(our_block.mlp, MLP)

        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

        hf_rotary = Qwen3_5TextRotaryEmbedding(config)
        cos, sin = hf_rotary(x, position_ids)

        hf_out = hf_block(
            hidden_states=x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=position_ids,
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        our_out = our_block(x, position_ids)

        torch.testing.assert_close(our_out, hf_out, atol=1e-3, rtol=1e-3)


class TestExportGuards:
    """Unsupported export options are rejected before anything is downloaded."""

    @pytest.fixture(autouse=True)
    def _offline_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = Qwen3_5Config(text_config=_make_qwen3_5_config().to_dict())
        monkeypatch.setattr(export_pipeline.AutoConfig, "from_pretrained", lambda *_, **__: config)

        def no_download(*_args, **_kwargs):
            raise AssertionError("the guard should fire before loading the model")

        monkeypatch.setattr(Qwen3_5ForCausalLM, "from_hf_memory_efficient", no_download)

    def _export(self, **options) -> None:
        config = export_pipeline.ExportConfig(
            hf_model_id="Qwen/Qwen3.5-0.8B", model_type_override="qwen3_5", **options
        )
        asyncio.run(export_pipeline._async_export_model(config))

    def test_num_layers(self) -> None:
        with pytest.raises(ValueError, match="--num-layers is not currently supported"):
            self._export(num_layers=2)

    def test_graph_quantization(self) -> None:
        with pytest.raises(ValueError, match="Graph-mode quantization is not currently supported"):
            self._export(quantization_mode="graph")


class TestLoadStateDict:
    def _model_and_state_dict(self) -> tuple[Qwen3_5ForCausalLM, dict[str, torch.Tensor]]:
        config = _make_qwen3_5_config()
        hf_model = HFQwen3_5ForCausalLM(config).eval()
        model = Qwen3_5ForCausalLM(config, model_device="cpu").eval()
        sd = dict(hf_model.state_dict())
        model._mutate_state_dict(sd)
        return model, sd

    def test_returns_incompatible_keys_and_leaves_input_alone(self) -> None:
        model, sd = self._model_and_state_dict()
        stripped = {k.removeprefix("model."): v for k, v in sd.items()}
        keys_before = list(stripped)

        result = model.load_state_dict(stripped, strict=True)

        assert result.missing_keys == [] and result.unexpected_keys == []
        assert list(stripped) == keys_before

    def test_non_strict_reports_foreign_keys_instead_of_raising(self) -> None:
        model, sd = self._model_and_state_dict()
        sd["mtp.fc.weight"] = torch.zeros(1)

        result = model.load_state_dict(sd, strict=False)

        assert result.unexpected_keys == ["mtp.fc.weight"]
        with pytest.raises(ValueError, match="Unexpected key"):
            model.load_state_dict(sd, strict=True)


class TestExportContract:
    """The hybrid graph carries two states beyond the base KV pair.

    ``test_export/test_export_contract.py`` covers the generic invariants (name/shape
    plumbing, signature order); these pin the Qwen3.5-specific overrides.
    """

    MAX_CONTEXT_LENGTH = 8192

    @pytest.fixture
    def config(self) -> Qwen3_5TextConfig:
        config = _make_qwen3_5_config()
        config.max_position_embeddings = self.MAX_CONTEXT_LENGTH
        return config

    @pytest.fixture
    def model(self, config: Qwen3_5TextConfig) -> Qwen3_5ForCausalLM:
        return Qwen3_5ForCausalLM(config, model_device="meta")

    @pytest.fixture
    def spec(self) -> TraceSpec:
        return TraceSpec(max_context_length=self.MAX_CONTEXT_LENGTH)

    def test_state_names(self, model: Qwen3_5ForCausalLM) -> None:
        assert model.export_state_names()[MAIN_GRAPH_NAME] == (
            KEY_CACHE_NAME,
            VALUE_CACHE_NAME,
            CONV_STATES_NAME,
            RECURRENT_STATES_NAME,
        )

    def test_ssm_state_shapes_and_dtype(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig, spec: TraceSpec
    ) -> None:
        refs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
        conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(
            config, dtype=torch.float16
        )
        assert refs["conv_states"].shape == conv_states.shape
        assert refs["recurrent_states"].shape == recurrent_states.shape
        for name in ("conv_states", "recurrent_states"):
            assert refs[name].dtype == torch.float16

    def test_states_are_sized_by_layer_type_not_depth(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig, spec: TraceSpec
    ) -> None:
        """Each state carries one row per layer of its own type, not one per layer."""
        refs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
        n_full = sum(1 for t in config.layer_types if t == "full_attention")
        n_linear = sum(1 for t in config.layer_types if t == "linear_attention")
        assert 0 < n_full < config.num_hidden_layers, "fixture must be a hybrid stack"

        assert refs["k_cache"].shape[0] == n_full
        assert refs["v_cache"].shape[0] == n_full
        assert refs["conv_states"].shape[0] == n_linear
        assert refs["recurrent_states"].shape[0] == n_linear

    def test_ssm_states_are_pinned(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig, spec: TraceSpec
    ) -> None:
        """Neither grows with sequence length, so both trace at fixed shape."""
        shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
        assert shapes["conv_states"] is None
        assert shapes["recurrent_states"] is None
        # The KV pair still has room to grow.
        assert shapes["k_cache"] is not None

    def test_contract_validates(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig, spec: TraceSpec
    ) -> None:
        model.validate_export_contract(
            model.build_reference_inputs(config, torch.float16, spec),
            model.build_dynamic_shapes(config, spec),
        )


class TestmacOSQwen3_5ForCausalLM:
    def test_output_shape(self) -> None:
        config = _make_qwen3_5_config()
        our_model = Qwen3_5ForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len, vocab = 1, 6, config.vocab_size
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = Qwen3_5ForCausalLM.create_kv_cache_tensors(config, dtype=torch.float32)
        conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(
            config, dtype=torch.float32
        )

        with torch.no_grad():
            out = our_model(
                input_ids, position_ids, k_cache, v_cache, conv_states, recurrent_states
            )

        assert out.shape == (batch, seq_len, vocab)

    @pytest.mark.parametrize(
        "make_config, hf_class",
        [
            (_make_qwen3_5_config, HFQwen3_5ForCausalLM),
            (_make_qwen3_5_moe_config, HFQwen3_5MoeForCausalLM),
        ],
        ids=["dense", "moe"],
    )
    def test_forward_parity(self, make_config, hf_class) -> None:
        """Prefill then one decode step match HF on a random-init model, dense and MoE."""
        torch.manual_seed(0)
        config = make_config()
        hf_model = hf_class(config).to(torch.float32).eval()
        our_model = Qwen3_5ForCausalLM(config, model_device="cpu").to(torch.float32).eval()
        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)

        k_cache, v_cache = Qwen3_5ForCausalLM.create_kv_cache_tensors(config, dtype=torch.float32)
        conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(
            config, dtype=torch.float32
        )
        states = (k_cache, v_cache, conv_states, recurrent_states)
        input_ids = torch.randint(0, config.vocab_size, (1, 7))
        prompt, next_id = input_ids[:, :6], input_ids[:, 6:]
        positions = torch.arange(7, dtype=torch.int32).unsqueeze(0)

        with torch.no_grad():
            our_prefill = our_model(prompt, positions[:, :6], *states)
            # position_ids spans the whole sequence so far; the model reads the
            # new token's position off its end.
            our_decode = our_model(next_id, positions, *states)
            hf_logits = hf_model(input_ids=input_ids, position_ids=positions.long()).logits

        torch.testing.assert_close(our_prefill, hf_logits[:, :6], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(our_decode, hf_logits[:, 6:], atol=1e-4, rtol=1e-4)

    def test_num_layers_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not currently supported for hybrid models"):
            Qwen3_5ForCausalLM._get_reauthored_config(_make_qwen3_5_config(), num_layers=1)

    def test_states_count_only_built_layers(self) -> None:
        config = _make_qwen3_5_config()
        config.layer_types = ["linear_attention", "full_attention", "full_attention"]
        config.num_hidden_layers = 2
        assert Qwen3_5ForCausalLM.kv_cache_layer_count(config) == 1


class TestPrefillGraph:
    """The model-side half of the optional ``prefill`` entrypoint.

    ``test_export/test_macos_prefill_graph.py`` covers what the exporter emits and how it
    converts; these pin what this model's ``forward`` has to do for that to be sound,
    which is all the exporter's second trace reads: return nothing under
    :attr:`prefill_mode`, and still write every state ``main`` writes.
    """

    PROMPT_LEN = 6

    @staticmethod
    def _states(config, dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
        """A fresh set of all four states, in ``forward`` argument order."""
        k_cache, v_cache = Qwen3_5ForCausalLM.create_kv_cache_tensors(config, dtype=dtype)
        conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(
            config, dtype=dtype
        )
        return [k_cache, v_cache, conv_states, recurrent_states]

    @pytest.fixture
    def config(self) -> Qwen3_5TextConfig:
        return _make_qwen3_5_config()

    @pytest.fixture
    def model(self, config: Qwen3_5TextConfig) -> Qwen3_5ForCausalLM:
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLM(config, model_device="cpu")
        # `conv_weight` is declared as zeros, and a zero depthwise filter drives the whole
        # gated-delta rule to zero -- the recurrent state would stay at its zeroed initial
        # value and every "was written" assertion below would pass vacuously.
        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                torch.nn.init.normal_(module.conv_weight, std=0.1)
        return model.to(torch.float32).eval()

    def _run(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5TextConfig,
        prefill_mode: bool,
        tokens: torch.Tensor,
        states: list[torch.Tensor] | None = None,
        offset: int = 0,
    ) -> tuple[object, list[torch.Tensor]]:
        """Run ``forward`` over ``tokens`` and return ``(output, states)``.

        ``states`` is mutated in place, so passing the previous call's states back in is
        how a chunked prompt is driven -- exactly what the runner does per chunk.
        """
        if states is None:
            states = self._states(config)
        query_len = tokens.shape[-1]
        # `position_ids` spans the whole processed prefix, the way the runner builds it:
        # `forward` derives its write offset from `len(position_ids) - len(input_ids)`.
        position_ids = torch.arange(offset + query_len, dtype=torch.int32).unsqueeze(0)
        model.set_prefill_mode(prefill_mode)
        try:
            with torch.no_grad():
                out = model(tokens, position_ids, *states)
        finally:
            model.set_prefill_mode(False)
        return out, states

    def test_opts_in(self) -> None:
        assert Qwen3_5ForCausalLM.exports_prefill_graph is True
        # Off until the exporter sets it, so `main` keeps its LM head.
        assert Qwen3_5ForCausalLM.prefill_mode is False

    def test_prefill_mode_returns_empty_tuple(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig
    ) -> None:
        """An empty tuple, not ``None``: a bare ``return`` traces a ``None`` leaf node."""
        tokens = torch.randint(0, config.vocab_size, (1, self.PROMPT_LEN))
        out, _ = self._run(model, config, prefill_mode=True, tokens=tokens)
        assert out == ()
        assert model.prefill_mode is False

    def test_prefill_mode_writes_the_same_states_as_decode(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig
    ) -> None:
        """Dropping the LM head must not drop any state write.

        The prefill graph declares no outputs, so these four tensors are its only
        product: a forward that skipped a cache write would look identical from outside.
        """
        tokens = torch.randint(0, config.vocab_size, (1, self.PROMPT_LEN))
        _, decode_states = self._run(model, config, prefill_mode=False, tokens=tokens)
        _, prefill_states = self._run(model, config, prefill_mode=True, tokens=tokens)

        for name, decoded, prefilled in zip(
            (KEY_CACHE_NAME, VALUE_CACHE_NAME, CONV_STATES_NAME, RECURRENT_STATES_NAME),
            decode_states,
            prefill_states,
            strict=True,
        ):
            # Not vacuous: prefill really wrote something, so two zeroed states can't pass.
            assert torch.any(prefilled != 0), f"{name} was never written"
            torch.testing.assert_close(prefilled, decoded, msg=f"{name} disagrees")

    def test_chunked_prefill_matches_whole_prompt(
        self, model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig
    ) -> None:
        """Prefilling in chunks leaves the same states as prefilling in one call.

        With a prefill graph the runner chunks every prompt, at any length, so this is
        the path Qwen3.5 actually takes. The linear-attention layers are what make it
        hold: their conv window and recurrent state carry the chunk boundary, and a
        chunk's KV writes land at the offset ``position_ids`` implies.
        """
        tokens = torch.randint(0, config.vocab_size, (1, self.PROMPT_LEN))
        _, whole = self._run(model, config, prefill_mode=True, tokens=tokens)

        split = self.PROMPT_LEN // 2
        _, chunked = self._run(model, config, prefill_mode=True, tokens=tokens[:, :split], offset=0)
        self._run(
            model,
            config,
            prefill_mode=True,
            tokens=tokens[:, split:],
            states=chunked,
            offset=split,
        )

        for name, one_shot, in_chunks in zip(
            (KEY_CACHE_NAME, VALUE_CACHE_NAME, CONV_STATES_NAME, RECURRENT_STATES_NAME),
            whole,
            chunked,
            strict=True,
        ):
            assert torch.any(in_chunks != 0), f"{name} was never written"
            torch.testing.assert_close(
                in_chunks, one_shot, atol=1e-4, rtol=1e-4, msg=f"{name} disagrees"
            )


@pytest.mark.slow
class TestQwen3_5ForCausalLMParity:
    """Logit parity against HF on real weights. Downloads Qwen3.5-0.8B."""

    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    KV_CACHE_LEN = 512
    # fp32 so the comparison checks the model, not low-precision noise (fp16 logits
    # differ from HF by more than any sensible tolerance on some tokens).
    DTYPE = torch.float32

    @pytest.fixture(autouse=True)
    def _require_weights(self) -> None:
        # A cached checkpoint (HF_HUB_CACHE may point at an external drive) needs no
        # network; otherwise skip unless the Hub can supply it.
        needed = ("config.json", "model.safetensors.index.json", "tokenizer.json")
        self.local_files_only = all(
            isinstance(try_to_load_from_cache(self.MODEL_ID, name), str) for name in needed
        )
        if not self.local_files_only and not _hf_hub_reachable(self.MODEL_ID):
            pytest.skip(f"{self.MODEL_ID!r} is neither cached nor reachable on the Hub")
        torch.manual_seed(0)

    def _load_pair(self) -> tuple[Qwen3_5ForCausalLM, HFQwen3_5ForCausalLM, Qwen3_5TextConfig]:
        hf_model = HFQwen3_5ForCausalLM.from_pretrained(
            self.MODEL_ID,
            local_files_only=self.local_files_only,
            dtype=self.DTYPE,
            attn_implementation="sdpa",
        ).eval()

        config = hf_model.config
        if hasattr(config, "text_config"):
            config = config.text_config

        our_model = Qwen3_5ForCausalLM(config, model_device="cpu").to(self.DTYPE).eval()
        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=True)
        return our_model, hf_model, config

    def _caches(self, config: Qwen3_5TextConfig) -> tuple[torch.Tensor, ...]:
        n_kv, head_dim = config.num_key_value_heads, config.head_dim
        n_full = Qwen3_5ForCausalLM.kv_cache_layer_count(config)
        shape = (n_full, 1, n_kv, self.KV_CACHE_LEN, head_dim)
        k_cache = torch.zeros(*shape, dtype=self.DTYPE)
        v_cache = torch.zeros(*shape, dtype=self.DTYPE)
        conv_states, recurrent_states = Qwen3_5ForCausalLM.create_delta_cache_tensors(
            config, dtype=self.DTYPE
        )
        return k_cache, v_cache, conv_states, recurrent_states

    def test_forward_parity_single_token(self) -> None:
        our_model, hf_model, config = self._load_pair()

        input_ids = torch.randint(0, config.vocab_size, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *self._caches(config))
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_multi_token(self) -> None:
        """Multi-token prefill followed by decode steps, both matching HF logits."""
        decode_steps = 5
        our_model, hf_model, config = self._load_pair()
        k_cache, v_cache, conv_states, recurrent_states = self._caches(config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, local_files_only=self.local_files_only
        )
        prompt = [{"role": "user", "content": "Hello"}]
        input_ids = torch.tensor(
            tokenizer.apply_chat_template(prompt, add_generation_prompt=True).input_ids
        ).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.int32).unsqueeze(0)

        with torch.no_grad():
            our_out = our_model(
                input_ids, position_ids, k_cache, v_cache, conv_states, recurrent_states
            )
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())
        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

        hf_inputs = input_ids
        new_ids = torch.argmax(our_out[:, -1:, :], dim=-1).to(torch.int32)
        for _ in range(decode_steps):
            new_position_id = torch.tensor([[position_ids.shape[-1]]], dtype=torch.int32)
            position_ids = torch.cat([position_ids, new_position_id], dim=-1)
            hf_inputs = torch.cat([hf_inputs, new_ids], dim=-1)

            with torch.no_grad():
                our_out = our_model(
                    new_ids, position_ids, k_cache, v_cache, conv_states, recurrent_states
                )
                hf_out = hf_model(input_ids=hf_inputs, position_ids=position_ids.long())

            torch.testing.assert_close(our_out, hf_out.logits[:, -1:, :], atol=1e-4, rtol=1e-4)
            new_ids = torch.argmax(our_out, dim=-1).to(torch.int32)
            if new_ids == config.eos_token_id:
                break


@pytest.mark.slow
class TestQwen3_5ForCausalLM(ForCausalLMTestBase):
    _toy_model_id = "yujiepan/qwen3.5-tiny-random"
    _model_class = Qwen3_5ForCausalLM
