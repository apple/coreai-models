# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base class for all ForCausalLM model implementations."""

import collections.abc
import gc
import json
import os
import re
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from typing_extensions import Self

from coreai_models.primitives.ios.embedding import GatherEmbeddings, LoadEmbeddings
from coreai_models.primitives.macos.cache import KVCache

T = TypeVar("T", bound="BaseForCausalLM")


def _is_layer_key_beyond(key: str, num_layers: int) -> bool:
    """Return True if `key` refers to a transformer layer with index >= num_layers.

    Used to filter HuggingFace state dicts when loading a truncated model.

    Args:
        key: State dict key, e.g. "model.layers.3.self_attn.q_proj.weight"
        num_layers: Maximum number of layers to keep (0-indexed, exclusive upper bound)

    Returns:
        True if the key should be dropped (layer index >= num_layers)
    """
    match = re.search(r"\.layers\.(\d+)\.", key)
    if match is None:
        return False
    return int(match.group(1)) >= num_layers


def move_model_to_disk(model: torch.nn.Module, path: str = "temp_weights.pt") -> torch.nn.Module:
    """
    Moves a model's parameters and buffers from RAM to disk-backed mmap tensors.

    This function:
    1. Saves state dict (parameters + buffers) to disk
    2. Reloads as mmap'd tensors (zero-copy from disk)

    Excludes:
    - KV cache buffers (runtime buffers, not model weights)

    Args:
        model: The model whose state should be moved to disk
        path: Path to save the weights file

    Returns:
        The same model, now with mmap-backed state
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 1. Save full state dict (except KVCache buffers)
    exclude_buffers = {KVCache.HF_K_BUFFER_NAME, KVCache.HF_V_BUFFER_NAME}
    param_names = {name for name, _ in model.named_parameters()}

    # Build filtered state dict, excluding KV cache
    state_dict = model.state_dict()
    filtered_state_dict = {
        name: tensor for name, tensor in state_dict.items() if name not in exclude_buffers
    }
    torch.save(filtered_state_dict, path)

    # 2. Load the raw tensors (mmap) & re-wrap appropriately
    mmap_sd = torch.load(path, map_location="cpu", mmap=True)
    new_state_dict = {}

    for name, tensor in mmap_sd.items():
        # Wrap as Parameter if it's a parameter
        if name in param_names:
            new_state_dict[name] = torch.nn.Parameter(tensor, requires_grad=False)
        else:
            # Keep buffers as regular tensors
            new_state_dict[name] = tensor

    # 3. Assign the state dict (strict=False since KVCache buffers are excluded)
    model.load_state_dict(new_state_dict, assign=True, strict=False)
    return model


def _save_and_mmap_safetensors(
    module: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    path: str,
) -> None:
    """Save tensors as safetensors and reload as mmap-backed, then assign to module.

    Saves the provided tensors to a safetensors file, then reloads them via
    ``safe_open`` so the module's tensors are backed by file-mapped pages the
    OS can evict freely.

    Args:
        module: The module to assign mmap-backed tensors to.
        tensors: Dict of {key: tensor} to save. Keys should be relative to
            ``module``'s state dict namespace.
        path: File path for the safetensors file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    param_names = {name for name, _ in module.named_parameters()}

    # safetensors `save_file` rejects entries that share storage (e.g. the iOS
    # mutate ties `extend.emb_scale` to `gather_embeddings.scale`). Clone any
    # entry that aliases an earlier one so each key is written independently.
    # The mmap round-trip below returns independent tensors per key regardless,
    # so this does not change the reloaded model.
    seen_storage: dict[int, str] = {}
    to_save: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        value = value.contiguous()
        storage_ptr = value.untyped_storage().data_ptr()
        if storage_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage[storage_ptr] = key
        to_save[key] = value

    save_file(to_save, path)

    new_sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            tensor = f.get_slice(key)[...]
            if key in param_names:
                new_sd[key] = torch.nn.Parameter(tensor, requires_grad=False)
            else:
                new_sd[key] = tensor

    module.load_state_dict(new_sd, assign=True, strict=False)


def _resolve_safetensors_files(model_dir: str) -> list[str]:
    """Resolve the safetensors file paths in a local HuggingFace model directory."""
    single_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        if "weight_map" not in index:
            raise RuntimeError(f"Malformed index at {index_path}: missing 'weight_map'")

        shard_filenames = sorted(set(index["weight_map"].values()))
        paths = [os.path.join(model_dir, fn) for fn in shard_filenames]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"Safetensors shards listed in index but missing on disk: {missing}"
            )
        return paths
    elif os.path.isfile(single_path):
        return [single_path]
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )


def _build_safetensors_key_index(
    safetensors_files: list[str],
    num_layers: int | None = None,
    hf_state_dict_prefix: str = "",
) -> tuple[dict[int, dict[str, str]], dict[str, str]]:
    """Build a key-to-file index from safetensors files without loading tensors.

    Keys that do not start with ``hf_state_dict_prefix`` are skipped. Use this
    to load only a sub-model from multimodal checkpoints (e.g., set
    ``hf_state_dict_prefix="language_model."`` to ignore vision/projector keys).

    Returns ``(per_layer_index, shared_index)`` keyed by *original* safetensors
    keys (prefix not stripped); callers must strip before assigning.
    """
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    per_layer: dict[int, dict[str, str]] = {}
    shared: dict[str, str] = {}
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if not key.startswith(hf_state_dict_prefix):
                    continue
                stripped = key.removeprefix(hf_state_dict_prefix)

                if num_layers is not None and _is_layer_key_beyond(stripped, num_layers):
                    continue
                match = layer_pattern.match(stripped)
                if match:
                    layer_idx = int(match.group(1))
                    per_layer.setdefault(layer_idx, {})[key] = path
                else:
                    shared[key] = path

    return per_layer, shared


def _load_tensors_for_keys(
    key_to_file: dict[str, str],
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load specific tensors from safetensors files by key.

    Each safetensors file is opened at most once. Tensors are cast to
    ``target_dtype`` except embedding tables and quantization zero-points.
    """
    file_to_keys: dict[str, list[str]] = {}
    for key, path in key_to_file.items():
        file_to_keys.setdefault(path, []).append(key)

    result: dict[str, torch.Tensor] = {}
    for path, keys in file_to_keys.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                if (
                    tensor.dtype != target_dtype
                    and "embedding_table" not in key
                    and "zero_point" not in key
                ):
                    tensor = tensor.to(target_dtype)
                result[key] = tensor

    return result


class BaseForCausalLM(torch.nn.Module):
    """Base class for all ForCausalLM implementations."""

    # Subclasses must override this with their specific HuggingFace model class
    _HF_MODEL_CLASS: type | None = None

    # When True, the memory-efficient streaming loader runs `_mutate_state_dict`
    # on the shared (non-layer) params too, so subclasses that transform
    # embeddings / lm_head / norms (e.g. the iOS models) produce their expected
    # keys. macOS models load shared params as-is and leave this False.
    _MUTATE_SHARED_PARAMS_ON_STREAM: bool = False

    @staticmethod
    def cast_logits_bfloat16_to_float16(forward_fn: Callable) -> Callable:
        """Decorator to cast torch.bfloat16 logits outputs to float16.

        This decorator checks if the output of a forward function is torch.bfloat16
        and casts it to float16 if needed.

        The casting behavior can be disabled by setting the environment variable
        DISABLE_BFLOAT16_CAST_FOR_LOGITS to "1" or "true" (case-insensitive).

        Args:
            forward_fn: The forward function to wrap

        Returns:
            Wrapped function that casts bfloat16 outputs to float16
        """

        @wraps(forward_fn)
        def wrapper(*args, **kwargs):
            output = forward_fn(*args, **kwargs)
            disable_cast = os.environ.get("DISABLE_BFLOAT16_CAST_FOR_LOGITS", "").lower() in (
                "1",
                "true",
            )

            if (
                not disable_cast
                and isinstance(output, torch.Tensor)
                and output.dtype == torch.bfloat16
            ):
                return output.to(torch.float16)
            return output

        return wrapper

    def __init__(self: Self, config, model_device: str = "cpu") -> None:
        """Initialize the model using template method pattern.

        Initializing the model on the meta device allows us to avoid
        allocating dummy tensors on cpu.

        Args:
            config: Model configuration object
            model_device: Device to use for initializing model components
                       (e.g., "cpu" or "meta")
        """
        super().__init__()
        self.config = config

        with torch.device(model_device):
            self._init_model(config)

    @abstractmethod
    def _init_model(self: Self, config) -> None:
        """Initialize model components on meta device."""
        ...

    @abstractmethod
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Sanitize the HuggingFace state dict in-place before loading.

        Subclasses can override this to perform model-specific transformations
        on the state dict (e.g., fusing weights, renaming keys, etc.).

        Args:
            state_dict: The state dict from HuggingFace model (modified in-place)
        """
        ...

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        """Convert HuggingFace config to model-specific config.

        Default implementation returns the HF config with max_position_embeddings
        modified if max_context_length is provided. Subclasses can override to
        convert to a custom config format.

        Args:
            hf_config: The HuggingFace model configuration
            max_context_length: Optional maximum context length to override
            num_layers: Optional number of transformer layers to override

        Returns:
            Config object to use for model initialization
        """
        if max_context_length is not None and hasattr(hf_config, "max_position_embeddings"):
            hf_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            if not hasattr(hf_config, "num_hidden_layers"):
                raise ValueError(
                    f"num_layers={num_layers} was specified but hf_config has no "
                    f"'num_hidden_layers' attribute (config type: {type(hf_config).__name__})"
                )
            hf_config.num_hidden_layers = num_layers
        return hf_config

    @classmethod
    def from_hf(
        cls: type[T],
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        disable_embedding_quantization: bool = False,
    ) -> T:
        """Load model from HuggingFace model hub.

        Args:
            huggingface_model_id: The HuggingFace model identifier
            max_context_length: Optional maximum context length to override config
            target_dtype: Target dtype for the model weights
            mmap_path: Optional path to use for mmaping the model weights to disk.
                       If provided, the model weights will be saved to this path
                       and memory-mapped to reduce RAM usage during import.
            num_layers: Optional number of transformer layers. When set, only layers
                        0..num_layers-1 are loaded and the config is truncated.
                        Useful for fast smoke tests.
            disable_embedding_quantization: iOS only. When True, the
                embedding table is not quantized to int8.
                Ignored for macOS model classes.

        Returns:
            Instance of the model class loaded with HuggingFace weights
        """
        if cls._HF_MODEL_CLASS is None:
            raise ValueError(f"{cls.__name__} must define _HF_MODEL_CLASS class attribute")

        # Load the HuggingFace model
        hf_model = cast(PreTrainedModel, cls._HF_MODEL_CLASS).from_pretrained(
            huggingface_model_id, dtype=target_dtype
        )

        # Convert config using the hook method (default: pass-through with context length)
        config = cls._get_reauthored_config(
            hf_model.config, max_context_length, num_layers=num_layers
        )

        # Create our model instance and load the state dict.
        # disable_embedding_quantization is only accepted by the iOS base class.
        init_kwargs: dict = {"config": config, "model_device": "meta"}
        if issubclass(cls, BaseForCausalLMForiOS):
            init_kwargs["disable_embedding_quantization"] = disable_embedding_quantization
        model = cls(**init_kwargs)
        model.to(dtype=target_dtype)
        state_dict = hf_model.state_dict()
        if not isinstance(state_dict, collections.abc.MutableMapping):
            # some HF models uses immutable state dict
            # (e.g. GPT-OSS uses collections.OrderedDict)
            # so we make a shallow copy into a mutable dict
            state_dict = dict(state_dict)
        del hf_model

        # Filter state dict to only include layers 0..num_layers-1.
        if num_layers is not None:
            state_dict = {
                k: v for k, v in state_dict.items() if not _is_layer_key_beyond(k, num_layers)
            }

        model._mutate_state_dict(state_dict)

        # check the state_dict is in the correct dtype
        for k, v in state_dict.items():
            if v.dtype != target_dtype and "embedding_table" not in k and "zero_point" not in k:
                err = f"tensor {k} in an incorrect dtype {v.dtype}. Supposed to be {target_dtype}."
                raise ValueError(err)

        strict = num_layers is None
        model.load_state_dict(state_dict, assign=True, strict=strict)

        # Move model weights to disk-backed mmap if path is provided
        if mmap_path is not None:
            move_model_to_disk(model, path=mmap_path)

        return model

    @classmethod
    def from_hf_memory_efficient(
        cls: type[T],
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = "",
        disable_embedding_quantization: bool = False,
    ) -> T:
        """Load model from HuggingFace with layer-by-layer memory offloading.

        Unlike :meth:`from_hf`, this method never loads the full HF model into
        RAM. It downloads the safetensors files, opens them via mmap, and
        processes one transformer layer at a time. When ``mmap_path`` is set
        the peak RAM is roughly *one layer + shared params*.

        Args:
            huggingface_model_id: HuggingFace model identifier.
            max_context_length: Optional override for the model's context length.
            target_dtype: Target dtype for the model weights.
            mmap_path: Directory for per-layer mmap files. When provided each
                layer is saved to ``<mmap_path>/layer_<i>.safetensors`` and
                reloaded mmap-backed before the next layer is processed.
                Shared params go to ``<mmap_path>/shared.safetensors``.
            num_layers: Optional number of transformer layers to load
                (truncates the config and skips layers >= num_layers).
            hf_config_attr: Optional attribute name on the top-level HF config
                to read the per-modality config from (e.g. ``"text_config"``
                for multimodal Gemma-3).
            hf_state_dict_prefix: Only safetensors keys starting with this
                prefix are loaded. The prefix is stripped before assigning.
                Use for multimodal checkpoints where text weights live under
                a prefix (e.g. ``"language_model."``).
            disable_embedding_quantization: iOS only. When True, the
                embedding table is not quantized to int8.
                Ignored for non-iOS model classes.
        """
        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )

        raw_config = AutoConfig.from_pretrained(model_dir)
        hf_config = getattr(raw_config, hf_config_attr) if hf_config_attr else raw_config

        config = cls._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)

        # disable_embedding_quantization is only accepted by the iOS base class.
        init_kwargs: dict = {"config": config, "model_device": "meta"}
        if issubclass(cls, BaseForCausalLMForiOS):
            init_kwargs["disable_embedding_quantization"] = disable_embedding_quantization
        model = cls(**init_kwargs)
        model.to(dtype=target_dtype)

        safetensors_files = _resolve_safetensors_files(model_dir)
        per_layer_index, shared_index = _build_safetensors_key_index(
            safetensors_files,
            num_layers=num_layers,
            hf_state_dict_prefix=hf_state_dict_prefix,
        )

        # Shared params first (embeddings, norm, lm_head, ...).
        shared_dict = _load_tensors_for_keys(shared_index, target_dtype)
        shared_dict = {k.removeprefix(hf_state_dict_prefix): v for k, v in shared_dict.items()}
        del shared_index

        # Models that transform shared params (iOS embedding quantization,
        # lm_head / norm renaming) mutate the shared slice here. macOS models
        # load shared params unchanged.
        if model._MUTATE_SHARED_PARAMS_ON_STREAM:
            model._mutate_state_dict(shared_dict)

        if mmap_path is not None:
            os.makedirs(mmap_path, exist_ok=True)
            shared_path = os.path.join(mmap_path, "shared.safetensors")
            _save_and_mmap_safetensors(model, shared_dict, shared_path)
        else:
            model.load_state_dict(shared_dict, assign=True, strict=False)

        del shared_dict
        gc.collect()

        # One transformer layer at a time.
        for layer_idx in sorted(per_layer_index.keys()):
            layer_key_to_file = per_layer_index.pop(layer_idx)
            layer_sd = _load_tensors_for_keys(layer_key_to_file, target_dtype)
            layer_sd = {k.removeprefix(hf_state_dict_prefix): v for k, v in layer_sd.items()}
            del layer_key_to_file

            # Per-model fusion (qkv, qk_norm, MoE expert stacking, Conv2d
            # reshapes, ...). Subclass `_mutate_state_dict` is layer-keyed and
            # safe on a single-layer slice; it rewrites keys into the model's
            # own namespace (e.g. iOS wraps layers under `extend.`), so the
            # mutated slice is assigned against the whole model.
            model._mutate_state_dict(layer_sd)

            if mmap_path is not None:
                layer_path = os.path.join(mmap_path, f"layer_{layer_idx}.safetensors")
                _save_and_mmap_safetensors(model, layer_sd, layer_path)
            else:
                model.load_state_dict(layer_sd, assign=True, strict=False)

            del layer_sd
            gc.collect()

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")

        return model

    @classmethod
    def from_pretrained(
        cls: type[T],
        model_path: str,
        config=None,
        max_context_length: int = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
    ) -> T:
        """Create model from pretrained weights on disk.

        Args:
            model_path: Path to saved model weights (.pt file)
            config: Config object for model initialization. Required.
            max_context_length: Optional maximum context length
            target_dtype: Target dtype for the model weights
            mmap_path: Optional path for memory-mapped weights

        Returns:
            Instance of the model class loaded with pretrained weights
        """
        if config is None:
            raise ValueError("config must be provided for from_pretrained")

        if max_context_length is not None and hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = max_context_length

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)

        state_dict = torch.load(model_path, map_location="cpu")
        model._mutate_state_dict(state_dict)
        model.load_state_dict(state_dict, assign=True)

        if mmap_path is not None:
            move_model_to_disk(model, path=mmap_path)

        return model

    def _reassign_cache(self: Self) -> None:
        if not hasattr(self, "cache"):
            return
        self.cache._k_cache = getattr(self, KVCache.HF_K_BUFFER_NAME)
        self.cache._v_cache = getattr(self, KVCache.HF_V_BUFFER_NAME)

    def half(self: Self) -> Self:
        super().half()
        self._reassign_cache()
        return self

    def bfloat16(self: Self) -> Self:
        super().bfloat16()
        self._reassign_cache()
        return self

    def float(self: Self) -> Self:
        super().float()
        self._reassign_cache()
        return self

    def to(self: Self, dtype) -> Self:
        super().to(dtype)
        self._reassign_cache()
        return self


class BaseForCausalLMForiOS(BaseForCausalLM):
    # iOS models rewrite the shared embedding / lm_head / norm keys inside
    # `_mutate_state_dict`, so the streaming loader must mutate the shared slice.
    _MUTATE_SHARED_PARAMS_ON_STREAM: bool = True

    def __init__(self: Self, config, model_device: str, disable_embedding_quantization=False):
        super().__init__(config, model_device)
        self.load_embeddings = LoadEmbeddings(
            config,
            embedding_table_dtype=torch.float32 if disable_embedding_quantization else torch.int8,
        )
        self.gather_embeddings = GatherEmbeddings()
        self.disable_embedding_quantization = disable_embedding_quantization

    def set_prefill_mode(self, prefill_mode: bool):
        self.extend.prefill_mode = prefill_mode
