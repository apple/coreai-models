# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base class for all ForCausalLM model implementations."""

import collections.abc
import gc
import inspect
import json
import os
import re
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast

import torch
from coreai.authoring.types import AllocationType, HardwareConstraints
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from typing_extensions import Self, override

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    GATHERED_EMBEDDINGS_OUTPUT_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    KEY_CACHE_NAME,
    KEY_CACHE_OUTPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    LOAD_EMBEDDINGS_OUTPUT_NAME,
    MAIN_GRAPH_NAME,
    OUTPUT_LOGITS_NAME,
    POSITION_IDS_INPUT_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TOKEN_IDS_INPUT_NAME,
    TRACE_KV_CACHE_SEQ_LEN,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    VALUE_CACHE_NAME,
    VALUE_CACHE_OUTPUT_NAME,
)
from coreai_models.primitives.ios.embedding import GatherEmbeddings, LoadEmbeddings
from coreai_models.primitives.macos.cache import KVCache

T = TypeVar("T", bound="BaseForCausalLM")


@dataclass(frozen=True)
class TraceSpec:
    """Shapes for one ``torch.export`` trace of a causal LM forward.

    Passed to both ``build_reference_inputs`` and ``build_dynamic_shapes`` so the
    tensors and their declared dims cannot disagree.

    Attributes:
        max_context_length: Upper bound for the dynamic seq/cache dims. Must be at
            least ``query_len + 2``.
        cache_seq_len: Length the caches are *traced* at, to bound peak trace memory.
            Unrelated to the inference cache size. Must not exceed
            ``max_context_length``.
        query_len: Trace-time ``input_ids`` length.
        offset: Already-cached positions, so ``position_ids`` is
            ``query_len + offset`` long.
    """

    max_context_length: int
    cache_seq_len: int = TRACE_KV_CACHE_SEQ_LEN
    query_len: int = QUANT_TRACE_QUERY_LEN
    offset: int = QUANT_TRACE_OFFSET

    def __post_init__(self) -> None:
        # Below this there is no legal `Dim(min=query_len, max=max_context_length - 1)`.
        if self.max_context_length < self.query_len + 2:
            raise ValueError(
                f"max_context_length={self.max_context_length} is too small to trace: "
                f"it must be at least query_len + 2 = {self.query_len + 2}."
            )
        if self.cache_seq_len > self.max_context_length:
            raise ValueError(
                "cache_seq_len must not be greater than max_context_length. Received "
                f"cache_seq_len = {self.cache_seq_len}, "
                f"max_context_length = {self.max_context_length}"
            )

    @property
    def caches_are_static(self) -> bool:
        """Whether the cache dims must be pinned rather than declared dynamic.

        A cache traced at the full context has nowhere to grow, and ``Dim(min=max=n)``
        is illegal anyway.
        """
        return self.cache_seq_len == self.max_context_length


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

    save_file({k: v.contiguous() for k, v in tensors.items()}, path)

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

    #: Whether the macOS exporter emits a second, LM-head-less ``prefill`` graph
    #: beside ``main``. Opt in per model: ``forward`` must honour
    #: :attr:`prefill_mode`.
    exports_prefill_graph: bool = False

    #: Set while the exporter traces the prefill graph. See :meth:`set_prefill_mode`.
    prefill_mode: bool = False

    def set_prefill_mode(self, prefill_mode: bool) -> None:
        """Toggle prefill mode for the next trace.

        The exporters trace one module twice -- decode, then prefill -- to emit two
        entrypoints from it. A model that opts into :attr:`exports_prefill_graph` checks
        :attr:`prefill_mode` in ``forward`` and returns nothing when it is set: the
        prefill graph only has to fill the KV cache, so the LM head and the tail of the
        model become dead code.
        """
        self.prefill_mode = prefill_mode

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

    # ------------------------------------------------------------------
    # Export contract
    #
    # Everything the exporters need to trace this model, keyed by graph name. A macOS
    # model has one traced signature; iOS has several. These hooks supply only names and
    # tensors; which callable each graph traces stays the exporter's business. A macOS
    # model that opts into `exports_prefill_graph` emits a second entrypoint from that one
    # signature, differing only in that it declares no outputs.
    #
    # Reference inputs bind to the traced signature, so they must be in its EXACT
    # order. Names are looked up by name, so each list carries only the RELATIVE order
    # of its own kind: for forward(input_ids, key_cache, position_ids, value_cache),
    # input_names is (input_ids, position_ids) and state_names is (key_cache,
    # value_cache).
    # ------------------------------------------------------------------

    @classmethod
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        """Graph input names per graph, in relative order among the non-state args."""
        return {MAIN_GRAPH_NAME: ("input_ids", "position_ids")}

    @classmethod
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        """Runner-visible state names per graph, in relative order among the state args.

        State args are mutated in place and surfaced through the runtime ``state=``
        kwarg rather than as ordinary inputs/outputs.
        """
        return {MAIN_GRAPH_NAME: (KEY_CACHE_NAME, VALUE_CACHE_NAME)}

    @classmethod
    def export_state_classification(cls) -> dict[str, str] | None:
        """Optional per-state classification for heterogeneous cache models.

        Returns a dict mapping each state name (from :meth:`export_state_names`) to
        a classification string (e.g. ``"kv_cache"``, ``"sliding_kv_cache"``).  The
        runtime uses this to allocate caches with different growth policies.

        Returns ``None`` by default, meaning all states share the same (standard
        KV-cache) treatment.
        """
        return None

    @classmethod
    def export_output_names(cls) -> dict[str, tuple[str, ...]]:
        """Graph output names per graph, in return order."""
        return {MAIN_GRAPH_NAME: ("logits",)}

    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        """Reference tensors to trace with, per graph, keyed by parameter name.

        The inner dicts bind to the traced callable, so their keys must be its
        parameters in *exact* signature order. Pass ``spec`` to
        :meth:`build_dynamic_shapes` too, so the tensors and their dims cannot disagree.
        """
        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        position_ids = (
            torch.arange(spec.query_len + spec.offset, dtype=torch.int32)
            .unsqueeze(0)
            .expand(1, spec.query_len + spec.offset)
        )
        k_cache, v_cache = KVCache.create_cache_tensors(
            config, dtype=target_dtype, seq_len=spec.cache_seq_len
        )
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "k_cache": k_cache,
                "v_cache": v_cache,
            }
        }

    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        """``dynamic_shapes`` per graph, matching :meth:`build_reference_inputs`.

        Keyed like the reference inputs; ``None`` pins that input to its traced shape.
        """
        max_ctx = spec.max_context_length
        shapes: dict[str, Any] = {
            "input_ids": {1: torch.export.Dim("seq_ids", max=max_ctx - 2)},
            "position_ids": {1: torch.export.Dim("seq_pos", min=spec.query_len, max=max_ctx - 1)},
        }
        seq_dim = KVCache.seq_len_dim()
        if spec.caches_are_static:
            shapes["k_cache"] = None
            shapes["v_cache"] = None
        else:
            shapes["k_cache"] = {
                seq_dim: torch.export.Dim("k_seq_len", min=spec.cache_seq_len, max=max_ctx)
            }
            shapes["v_cache"] = {
                seq_dim: torch.export.Dim("v_seq_len", min=spec.cache_seq_len, max=max_ctx)
            }
        return {MAIN_GRAPH_NAME: shapes}

    def validate_export_contract(
        self,
        reference_inputs: dict[str, dict[str, Any]],
        dynamic_shapes: dict[str, Any],
    ) -> None:
        """Check the five hooks agree with each other.

        The converter compares name *counts* only, so it would accept a contract whose
        graphs disagree. Called by the exporters before tracing.

        Raises:
            ValueError: On any disagreement between the hooks.
        """
        cls_name = type(self).__name__
        inputs, states, outputs = (
            self.export_input_names(),
            self.export_state_names(),
            self.export_output_names(),
        )
        named = {
            "export_input_names": set(inputs),
            "export_state_names": set(states),
            "export_output_names": set(outputs),
            "build_reference_inputs": set(reference_inputs),
            "build_dynamic_shapes": set(dynamic_shapes),
        }
        graphs = named["export_input_names"]
        for hook, keys in named.items():
            if keys != graphs:
                raise ValueError(
                    f"{cls_name}: {hook} covers graphs {sorted(keys)} but "
                    f"export_input_names covers {sorted(graphs)}. Every hook must "
                    "describe the same graphs."
                )

        for graph in sorted(graphs):
            refs = reference_inputs[graph]
            declared = len(inputs[graph]) + len(states[graph])
            if declared != len(refs):
                raise ValueError(
                    f"{cls_name}, graph {graph!r}: {len(inputs[graph])} input names + "
                    f"{len(states[graph])} state names = {declared}, but "
                    f"build_reference_inputs supplies {len(refs)} tensors "
                    f"{tuple(refs)}."
                )
            shapes = dynamic_shapes[graph]
            if shapes is not None and set(shapes) != set(refs):
                raise ValueError(
                    f"{cls_name}, graph {graph!r}: dynamic_shapes keys "
                    f"{tuple(shapes)} do not match reference inputs {tuple(refs)}."
                )

    def reference_inputs_as_args(self, reference_inputs: dict[str, Any]) -> tuple[Any, ...]:
        """One graph's reference inputs as positional args, validating the order.

        For the coreai-opt quantizer, which takes a tuple rather than kwargs; a dict
        ordered differently from the signature would silently bind the wrong tensors.

        Raises:
            ValueError: If the keys are not a contiguous in-order prefix of ``forward``.
        """
        params = list(inspect.signature(self.forward).parameters)
        keys = list(reference_inputs)
        if keys != params[: len(keys)]:
            raise ValueError(
                f"{type(self).__name__}.build_reference_inputs returned keys {keys}, "
                f"which are not a contiguous in-order prefix of forward's parameters "
                f"{params}. Positional conversion would bind tensors to the wrong "
                f"parameters; fix the dict's insertion order or the signature."
            )
        return tuple(reference_inputs.values())

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

        if mmap_path is not None:
            os.makedirs(mmap_path, exist_ok=True)
            shared_path = os.path.join(mmap_path, "shared.safetensors")
            _save_and_mmap_safetensors(model, shared_dict, shared_path)
        else:
            model.load_state_dict(shared_dict, assign=True, strict=False)

        del shared_dict
        gc.collect()

        # One transformer layer at a time.
        exclude_buffers = {KVCache.HF_K_BUFFER_NAME, KVCache.HF_V_BUFFER_NAME}
        for layer_idx in sorted(per_layer_index.keys()):
            layer_key_to_file = per_layer_index.pop(layer_idx)
            layer_sd = _load_tensors_for_keys(layer_key_to_file, target_dtype)
            layer_sd = {k.removeprefix(hf_state_dict_prefix): v for k, v in layer_sd.items()}
            del layer_key_to_file

            # Per-model fusion (qkv, qk_norm, MoE expert stacking, ...).
            # Subclass `_mutate_state_dict` is layer-keyed and safe on a
            # single-layer slice.
            model._mutate_state_dict(layer_sd)

            if mmap_path is not None:
                layer_prefix = f"model.layers.{layer_idx}."
                relative_sd = {
                    k.removeprefix(layer_prefix): v
                    for k, v in layer_sd.items()
                    if k.removeprefix(layer_prefix) not in exclude_buffers
                }
                layer_module = model.model.layers[layer_idx]
                layer_path = os.path.join(mmap_path, f"layer_{layer_idx}.safetensors")
                _save_and_mmap_safetensors(layer_module, relative_sd, layer_path)
                del relative_sd
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
    def __init__(self: Self, config, model_device: str, disable_embedding_quantization=False):
        super().__init__(config, model_device)
        self.load_embeddings = LoadEmbeddings(
            config,
            embedding_table_dtype=torch.float32 if disable_embedding_quantization else torch.int8,
        )
        self.gather_embeddings = GatherEmbeddings()
        self.disable_embedding_quantization = disable_embedding_quantization

    @override
    def set_prefill_mode(self, prefill_mode: bool):
        # iOS composes the transformer as a submodule, so the flag lives there rather
        # than on the top-level module.
        self.extend.prefill_mode = prefill_mode

    # ------------------------------------------------------------------
    # Export contract
    #
    # iOS exports four entrypoints over three callables: the embedding-table loader,
    # the token gather, and the transformer, the last traced twice with prefill off and
    # on. The two transformer entrypoints have identical inputs, states and outputs, so
    # the contract carries three entries and `export/ios.py` uses the transformer entry
    # for both. Which callable each graph traces, and the prefill toggle between them,
    # is the exporter's business.
    #
    # `model.forward` composes the three for eager use and is never exported, so the
    # inherited single-graph defaults would describe the wrong graph entirely.
    #
    # iOS also needs two things macOS does not: the static shape ladder each graph is
    # specialized over, and the layout constraints on the buffers it shares with the
    # runner. Both are per-graph and shaped by this model's graphs, so they live here
    # alongside the names.
    # ------------------------------------------------------------------

    #: Query length the iOS graphs are traced at.
    IOS_QUERY_LEN = 8

    #: Query lengths the graphs are specialized for.
    IOS_STATIC_QUERY_LENS = (8, 16, 64)

    #: Smallest cache length in the static ladder; it doubles up to the context.
    IOS_STATIC_MIN_CACHE_LEN = 256

    #: Interleave factor for the KV cache's embedding dim.
    KV_CACHE_INTERLEAVE_FACTOR = 8

    @classmethod
    def _head_dim(cls, config) -> int:
        """``config.head_dim`` when it is set, else derived from hidden size and heads."""
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = config.hidden_size // config.num_attention_heads
        return head_dim

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (
                TOKEN_IDS_INPUT_NAME,
                EMBEDDING_TABLE_INPUT_NAME,
            ),
            EXTEND_FUNCTION_NAME: (
                TRANSFORMER_INPUT_NAME,
                POSITION_IDS_INPUT_NAME,
                IN_STEP_INPUT_NAME,
                CAUSAL_MASK_INPUT_NAME,
                EMBEDDING_TABLE_INPUT_NAME,
            ),
        }

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (),
            EXTEND_FUNCTION_NAME: (KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME),
        }

    @classmethod
    def export_state_output_names(cls) -> dict[str, tuple[str, ...]]:
        """State output names per graph.

        iOS names them because the hardware constraints attach to the mutated output as
        well as the input; the macOS converter surfaces state in/out implicitly.
        """
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (),
            EXTEND_FUNCTION_NAME: (KEY_CACHE_OUTPUT_NAME, VALUE_CACHE_OUTPUT_NAME),
        }

    @classmethod
    @override
    def export_output_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: (LOAD_EMBEDDINGS_OUTPUT_NAME,),
            GATHER_EMBEDDINGS_FUNCTION_NAME: (GATHERED_EMBEDDINGS_OUTPUT_NAME,),
            EXTEND_FUNCTION_NAME: (OUTPUT_LOGITS_NAME,),
        }

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, Any]]:
        max_ctx = spec.max_context_length
        query_len = self.IOS_QUERY_LEN
        head_dim = self._head_dim(config)

        embedding_table = self.load_embeddings.embedding_table
        input_ids = torch.randint(1, config.vocab_size, (1, query_len), dtype=torch.int32)
        # One graph's reference input is produced by running another.
        transformer_input = self.gather_embeddings(input_ids, embedding_table)

        key_cache = torch.zeros(
            config.num_hidden_layers,
            1,
            config.num_key_value_heads * head_dim,
            1,
            max_ctx,
            dtype=torch.float16,
        )

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                "input_ids": input_ids,
                "embedding_table": embedding_table,
            },
            # Exact signature order of Qwen3Extend.forward and friends.
            EXTEND_FUNCTION_NAME: {
                "transformer_input": transformer_input,
                "position_ids": torch.arange(query_len).to(torch.uint16).unsqueeze(0),
                "in_step": torch.zeros((1,), dtype=torch.int32),
                "causal_mask": torch.zeros(1, max_ctx, 1, query_len, dtype=torch.float16),
                "key_cache": key_cache,
                "value_cache": key_cache.clone(),
                "embedding_table": embedding_table,
            },
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, Any]:
        max_ctx = spec.max_context_length
        seq_len_dim = torch.export.Dim("seq_len", max=max_ctx)
        cache_len_dim = torch.export.Dim("cache_len", max=max_ctx)
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                "input_ids": {1: seq_len_dim},
                "embedding_table": None,
            },
            EXTEND_FUNCTION_NAME: {
                "transformer_input": {1: seq_len_dim},
                "position_ids": {1: seq_len_dim},
                "in_step": None,
                "causal_mask": {1: cache_len_dim, 3: seq_len_dim},
                "key_cache": {4: cache_len_dim},
                "value_cache": {4: cache_len_dim},
                "embedding_table": None,
            },
        }

    @classmethod
    def export_static_shape_configs(
        cls, config, max_context_length: int
    ) -> dict[str, dict[str, dict[str, tuple[int, ...]]]]:
        """Static shape specializations per graph, keyed like the name hooks.

        iOS compiles for fixed shapes, so each graph is built once per shape it must
        serve: the transformer over (cache length, query length), the gather over query
        length alone. The inner keys are the specialization labels, quoted because that
        is the form the compiler expects. An empty dict means no specialization.
        """
        kv_embed_size = config.num_key_value_heads * cls._head_dim(config)

        transformer: dict[str, dict[str, tuple[int, ...]]] = {}
        cache_len = cls.IOS_STATIC_MIN_CACHE_LEN
        while cache_len <= max_context_length:
            for q_len in cls.IOS_STATIC_QUERY_LENS:
                transformer[f'"{cache_len}_{q_len}"'] = {
                    TRANSFORMER_INPUT_NAME: (1, q_len, 1, config.hidden_size),
                    POSITION_IDS_INPUT_NAME: (1, q_len),
                    CAUSAL_MASK_INPUT_NAME: (1, cache_len, 1, q_len),
                    KEY_CACHE_INPUT_NAME: (
                        config.num_hidden_layers,
                        1,
                        kv_embed_size,
                        1,
                        cache_len,
                    ),
                    VALUE_CACHE_INPUT_NAME: (
                        config.num_hidden_layers,
                        1,
                        kv_embed_size,
                        1,
                        cache_len,
                    ),
                }
            cache_len *= 2

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                f'"{q}"': {TOKEN_IDS_INPUT_NAME: (1, q)} for q in cls.IOS_STATIC_QUERY_LENS
            },
            EXTEND_FUNCTION_NAME: transformer,
        }

    @classmethod
    def export_hardware_constraints(
        cls, max_context_length: int
    ) -> dict[str, dict[str, HardwareConstraints]]:
        """Per-input hardware constraints per graph, keyed like the name hooks.

        These pin the layout of the buffers the runner shares with the compiled graphs,
        so both sides agree. The caches are constrained on the mutated output as well as
        the input, which is why :meth:`export_state_output_names` exists.
        """
        embedding_table = HardwareConstraints(
            AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
        )
        cache = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, cls.KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, cls.KV_CACHE_INTERLEAVE_FACTOR * max_context_length, 1],
        )

        transformer = {EMBEDDING_TABLE_INPUT_NAME: embedding_table}
        for name in (
            *cls.export_state_names()[EXTEND_FUNCTION_NAME],
            *cls.export_state_output_names()[EXTEND_FUNCTION_NAME],
        ):
            transformer[name] = cache

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {EMBEDDING_TABLE_INPUT_NAME: embedding_table},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {EMBEDDING_TABLE_INPUT_NAME: embedding_table},
            EXTEND_FUNCTION_NAME: transformer,
        }
