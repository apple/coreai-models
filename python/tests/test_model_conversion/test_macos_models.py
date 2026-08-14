# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end macOS model conversion tests.

The autouse ``use_hf_impl`` fixture and the per-test ``disable_hf_impl_for_coreai``
fixture come from the directory ``conftest.py``.
"""

import os

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM as HFGemma3ForCausalLM,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM as HFGptOssForCausalLM,
)
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM as HFMistralForCausalLM,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM as HFMixtralForCausalLM,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM,
)

from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
from coreai_models.models.macos.gpt_oss import GptOssForCausalLM
from coreai_models.models.macos.mistral import MistralForCausalLM
from coreai_models.models.macos.mixtral import MixtralForCausalLM
from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM
from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
from coreai_models.models.macos.qwen3_moe import Qwen3MoeForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from tests._runner_infra.testing_utils import (
    create_dynamic_shapes_for_explicit_kv_coreai_test,
    create_test_inputs,
    load_state_dict_from_ref_model,
    run_compare_coreai_explicit_kv_cache,
    run_torch_prompt_extend_test,
    switch_block_to_scatter,
)


@pytest.fixture(autouse=True, scope="module")
def disable_bfloat16_cast():
    """Disable bfloat16 to float16 casting for model conversion tests.

    This fixture ensures that the cast_bfloat16_to_float16 decorator doesn't
    interfere with the model conversion tests, which need to test models
    in their original dtypes including bfloat16.
    """
    original = os.environ.get("DISABLE_BFLOAT16_CAST_FOR_LOGITS")
    os.environ["DISABLE_BFLOAT16_CAST_FOR_LOGITS"] = "1"
    yield
    if original is None:
        os.environ.pop("DISABLE_BFLOAT16_CAST_FOR_LOGITS", None)
    else:
        os.environ["DISABLE_BFLOAT16_CAST_FOR_LOGITS"] = original


class TestQwen2EndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen2.5-tiny-random",
            pytest.param("Qwen/Qwen2.5-0.5B", marks=pytest.mark.slow),
            pytest.param("Qwen/Qwen2.5-1.5B-Instruct", marks=pytest.mark.slow),
        ],
    )
    def test_hf(hf_model_id: str) -> None:
        """Test model comparison with prompt / extend."""
        ref_model = HFQwen2ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        model = Qwen2ForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)
        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=torch.float32,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen2.5-tiny-random",
            pytest.param("Qwen/Qwen2.5-0.5B", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_coreai(hf_model_id: str) -> None:
        max_seq_len = 4096
        ref_model = HFQwen2ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = max_seq_len
        model = Qwen2ForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-4,
            rtol=1e3,
        )


def _load_gemma3_ref_model(hf_model_id: str) -> HFGemma3ForCausalLM:
    """Load the Gemma3 reference model in float32.

    `from_pretrained` defaults to the checkpoint dtype (bfloat16 for
    google/gemma-3-*), and the cast reaches `embed_tokens.embed_scale` — a
    non-persistent buffer holding `hidden_size ** 0.5`. For the 1B model that
    rounds sqrt(1152) = 33.94113 to bfloat16's 34.0, a 0.17% error baked into
    every embedding, which stays even after the caller casts the model back up.
    Our `Embedding` derives the scale at its own dtype, so it never carries that
    rounding. Loading in float32 keeps both sides deriving the scale the same
    way; the caller casts to the precision under test afterwards.
    """
    return HFGemma3ForCausalLM.from_pretrained(hf_model_id, dtype=torch.float32).eval()


class TestGemma3EndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/gemma-3-tiny-random",
            pytest.param("google/gemma-3-1b-it", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize(
        "precision_tol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 1e-4, 1e-4),
            (torch.bfloat16, 5e-3, 1e-2),
        ],
    )
    def test_hf(hf_model_id: str, precision_tol: tuple[torch.dtype, float, float]) -> None:
        """Test model comparison with prompt / extend."""
        precision, atol, rtol = precision_tol

        ref_model = _load_gemma3_ref_model(hf_model_id)
        model = Gemma3ForCausalLM(ref_model.config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        if hf_model_id == "google/gemma-3-1b-it" and precision == torch.float16:
            # 1B model pass the prompt step but could results in inaccurate prediction after
            # several steps of extend.
            extend_steps = 1
        else:
            extend_steps = 3

        if hf_model_id == "google/gemma-3-1b-it" and precision == torch.float32:
            rtol = 1e2

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            atol=atol,
            rtol=rtol,
            extend_steps=extend_steps,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/gemma-3-tiny-random",
            pytest.param("google/gemma-3-1b-it", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize("use_bfp16", [True, False])
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_coreai(hf_model_id: str, use_bfp16: bool) -> None:
        # create models from config
        max_seq_len = 4096
        ref_model = _load_gemma3_ref_model(hf_model_id)
        config = ref_model.config
        config.max_position_embeddings = max_seq_len
        model = Gemma3ForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        if use_bfp16:
            model = model.bfloat16()

        input_ids, position_ids = create_test_inputs(config)
        dtype = torch.bfloat16 if use_bfp16 else torch.float32
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=dtype)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        atol, rtol = (10, 1e8) if use_bfp16 else (1e-4, 1e3)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=atol,
            rtol=rtol,
        )


class TestGptOssEndtoEnd:
    @pytest.fixture(autouse=True, scope="class")
    def _use_hf_impl_false(self):
        """Override the module-scope `USE_HF_IMPL=true` to false for these tests."""
        original = os.environ.get("USE_HF_IMPL")
        os.environ["USE_HF_IMPL"] = "false"
        yield
        if original is None:
            os.environ.pop("USE_HF_IMPL", None)
        else:
            os.environ["USE_HF_IMPL"] = original

    @staticmethod
    @pytest.mark.parametrize("hf_model_id", ["yujiepan/gpt-oss-tiny-random"])
    @pytest.mark.parametrize(
        "precision_tol",
        [
            (torch.float32, 1e-4, 1e-1),
            (torch.bfloat16, 5, 1e6),
        ],
    )
    def test_hf(
        hf_model_id: str,
        precision_tol: tuple[torch.dtype, float, float],
    ) -> None:
        """Test model comparison with prompt / extend."""
        precision, atol, rtol = precision_tol

        ref_model = HFGptOssForCausalLM.from_pretrained(hf_model_id).eval()
        model = GptOssForCausalLM(ref_model.config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        extend_steps = 1 if precision == torch.float16 else 3

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            atol=atol,
            rtol=rtol,
            extend_steps=extend_steps,
        )

    @staticmethod
    @pytest.mark.parametrize("hf_model_id", ["yujiepan/gpt-oss-tiny-random-mxfp4"])
    @pytest.mark.parametrize(
        "precision_tol",
        [
            (torch.bfloat16, 5, 1e6),
        ],
    )
    def test_hf_mxfp4(
        hf_model_id: str,
        precision_tol: tuple[torch.dtype, float, float],
    ) -> None:
        """Test MXFP4-quantized GPT-OSS model loading and inference."""
        precision, atol, rtol = precision_tol

        model = GptOssForCausalLM.from_hf(hf_model_id, target_dtype=precision).eval()

        ref_model = HFGptOssForCausalLM.from_pretrained(hf_model_id, torch_dtype=precision).eval()

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            atol=atol,
            rtol=rtol,
            extend_steps=1,
            skip_dtype_cast=True,
        )

    @staticmethod
    @pytest.mark.parametrize("hf_model_id", ["yujiepan/gpt-oss-tiny-random-mxfp4"])
    def test_coreai_mxfp4(hf_model_id: str) -> None:
        """Test MXFP4 GPT-OSS 20b model (1 layer): torch forward vs Core AI runtime."""
        pytest.xfail("CPU runtime produces NaN on gpt-oss toy model")
        precision = torch.float16
        max_seq_len = 4096

        model = GptOssForCausalLM.from_hf(hf_model_id, target_dtype=precision, num_layers=1).eval()

        config = model.config
        config.max_position_embeddings = max_seq_len

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=precision)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-2,
            rtol=1e3,
        )


class TestQwen3EndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen3-tiny-random",
            pytest.param("Qwen/Qwen3-0.6B", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize(
        "precision_tol",
        [
            (torch.float32, 1e-4, 1e-1),
            (torch.bfloat16, 2, 1e5),
        ],
    )
    def test_hf(hf_model_id: str, precision_tol: tuple[torch.dtype, float, float]) -> None:
        """Test model comparison with prompt / extend."""
        precision, atol, rtol = precision_tol

        ref_model = HFQwen3ForCausalLM.from_pretrained(hf_model_id).eval()
        model = Qwen3ForCausalLM(ref_model.config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            atol=atol,
            rtol=rtol,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen3-tiny-random",
            pytest.param("Qwen/Qwen3-0.6B", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_coreai(hf_model_id: str) -> None:
        # create models from config
        max_seq_len = 40960
        ref_model = HFQwen3ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = max_seq_len
        model = Qwen3ForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-4,
            rtol=1e3,
        )


class TestQwen3MoeEndtoEnd:
    @staticmethod
    @pytest.mark.parametrize("hf_model_id", ["yujiepan/qwen3-moe-tiny-random"])
    @pytest.mark.parametrize(
        "precision_tol",
        [
            (torch.float32, 1e-4, 1e-1),
            (torch.bfloat16, 5, 1e6),
        ],
    )
    @pytest.mark.parametrize("switch_sparse_block_to_vanilla", [True, False])
    def test_hf(
        hf_model_id: str,
        precision_tol: tuple[torch.dtype, float, float],
        switch_sparse_block_to_vanilla: bool,
    ) -> None:
        """Test model comparison with prompt / extend."""
        precision, atol, rtol = precision_tol

        ref_model = HFQwen3MoeForCausalLM.from_pretrained(hf_model_id).eval()
        model = Qwen3MoeForCausalLM(ref_model.config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        if switch_sparse_block_to_vanilla:
            model = switch_block_to_scatter(model)

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            atol=atol,
            rtol=rtol,
            strict_compare_numerical=bool(switch_sparse_block_to_vanilla),
        )


class TestMixtralEndtoEnd:
    @staticmethod
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("switch_sparse_block_to_vanilla", [False])
    def test_hf(precision: torch.dtype, switch_sparse_block_to_vanilla: bool) -> None:
        """Test model comparison with prompt / extend."""
        config = MixtralConfig(
            hidden_size=512,
            intermediate_size=256,
            max_position_embeddings=4096,
            num_hidden_layers=2,
        )
        ref_model = HFMixtralForCausalLM(config).eval()
        model = MixtralForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        if switch_sparse_block_to_vanilla:
            model = switch_block_to_scatter(model)

        extend_steps = 1 if precision == torch.float16 else 3

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            rtol=1e-1,
            extend_steps=extend_steps,
            strict_compare_numerical=bool(switch_sparse_block_to_vanilla),
        )

    @staticmethod
    @pytest.mark.slow
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_coreai() -> None:
        max_seq_len = 4096
        config = MixtralConfig(
            hidden_size=512,
            intermediate_size=256,
            max_position_embeddings=max_seq_len,
            num_hidden_layers=1,
        )
        ref_model = HFMixtralForCausalLM(config).eval()
        model = MixtralForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
        )


class TestMistralEndtoEnd:
    @staticmethod
    def get_7b_like_model() -> torch.nn.Module:
        """mistralai/Mistral-7B-Instruct-v0.3 is too big so we create a smaller one."""
        config = MistralConfig(
            num_hidden_layers=3,
            sliding_window=None,
            max_position_embeddings=40960,
        )
        return HFMistralForCausalLM(config).eval()

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/mistral-v0.3-tiny-random",
            pytest.param("mistralai/Mistral-7B-Instruct-v0.3", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
    def test_hf(hf_model_id: str, precision: torch.dtype) -> None:
        """Test model comparison with prompt / extend."""
        if hf_model_id == "mistralai/Mistral-7B-Instruct-v0.3":
            ref_model = TestMistralEndtoEnd.get_7b_like_model()
        else:
            ref_model = HFMistralForCausalLM.from_pretrained(hf_model_id).eval()

        if hf_model_id == "mistralai/Mistral-7B-Instruct-v0.3" and precision == torch.float16:
            # 7B model pass the prompt step but could results in inaccurate prediction after
            # several steps of extend.
            extend_steps = 1
        else:
            extend_steps = 3

        model = MistralForCausalLM(ref_model.config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        run_torch_prompt_extend_test(
            model,
            ref_model,
            precision=precision,
            rtol=1e-1,
            extend_steps=extend_steps,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/mistral-v0.3-tiny-random",
            pytest.param("mistralai/Mistral-7B-Instruct-v0.3", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_coreai(hf_model_id: str) -> None:
        # create models from config
        max_seq_len = 4096
        if hf_model_id == "mistralai/Mistral-7B-Instruct-v0.3":
            ref_model = TestMistralEndtoEnd.get_7b_like_model()
            config = ref_model.config
            config.max_position_embeddings = max_seq_len
        else:
            ref_model = HFMistralForCausalLM.from_pretrained(hf_model_id).eval()
            config = ref_model.config
            config.max_position_embeddings = max_seq_len

        model = MistralForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config)
        inputs = (input_ids, position_ids, k_cache, v_cache)
        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(max_seq_len)

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-4,
            rtol=1e3,
        )


# TODO: Make dynamic KV cache the default behavior for all models once mainstream
class TestDynamicKVCacheCoreAI:
    """Tests for Core AI export with dynamic-sized KV cache."""

    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen2.5-tiny-random",
        ],
    )
    @pytest.mark.usefixtures("disable_hf_impl_for_coreai")
    def test_qwen2(hf_model_id: str) -> None:
        max_seq_len = 4096
        ref_model = HFQwen2ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = max_seq_len
        model = Qwen2ForCausalLM(config).eval()
        load_state_dict_from_ref_model(model, ref_model)

        input_ids, position_ids = create_test_inputs(config)
        k_cache, v_cache = KVCache.create_cache_tensors(config)
        inputs = (input_ids, position_ids, k_cache, v_cache)

        dynamic_shapes = create_dynamic_shapes_for_explicit_kv_coreai_test(
            max_seq_len,
        )

        run_compare_coreai_explicit_kv_cache(
            model=model,
            inputs=inputs,
            dynamic_shapes=dynamic_shapes,
            rtol=1e3,
        )
