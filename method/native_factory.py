"""Native GigaBrain/DuoFlow adapter construction for the System2 selector.

The factory keeps the S1 object untouched.  It loads a separate native model
instance for CPU bootstrap, applies the released DuoFlow FT2T actor LoRA to
that instance, and exposes only the official preprocessing objects needed by
``System2Backend``.
"""

from __future__ import annotations

import importlib.util
import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch

try:
    from .system2_backend import NativeInterfaceError, System2Backend, load_gigabrain07_policy
except ImportError:  # Direct ``python native_factory.py`` / bootstrap import.
    from system2_backend import NativeInterfaceError, System2Backend, load_gigabrain07_policy


_LOCAL_GIGA_ROOT = Path(__file__).resolve().parents[1] / "gigabrain07_execution_comm_20261009"
_LOCAL_TWINTIME_ROOT = Path(__file__).resolve().parents[1] / "twintimefm_official_retest_20261019"
_PLANNER_MAX_LENGTH = 512


def _unique_paths(values: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        value = value.expanduser()
        key = str(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _asset_roots(reference_root: str | Path) -> list[Path]:
    requested = Path(reference_root).expanduser()
    roots = [
        requested,
        requested / "model_assets",
        requested / "runtime",
        requested / "runtime" / "XPolicyLab",
        requested.parent / "gigabrain07_execution_comm_20261009",
        requested.parent / "twintimefm_official_retest_20261019",
        _LOCAL_GIGA_ROOT,
        _LOCAL_TWINTIME_ROOT,
    ]
    return _unique_paths(roots)


def _model_candidates(root: Path) -> list[Path]:
    candidates = [
        root,
        root / "model",
        root / "model_ema",
        root / "model_assets",
        root / "model_assets" / "model",
        root / "model_assets" / "model_ema",
        root / "checkpoints" / "model",
    ]
    if root.exists() and root.is_dir():
        candidates.extend(path for path in root.glob("*/model") if path.is_dir())
        candidates.extend(path for path in root.glob("*/model_ema") if path.is_dir())
    return _unique_paths(candidates)


def _find_model_dir(roots: Iterable[Path]) -> Path:
    for root in roots:
        for candidate in _model_candidates(root):
            if (candidate / "config.json").is_file() and (candidate / "diffusion_pytorch_model.bin").is_file():
                return candidate
    raise FileNotFoundError(
        "cannot locate a complete Clean GigaBrain model (config.json + "
        "diffusion_pytorch_model.bin); checked local assets and reference_root"
    )


def _find_file(roots: Iterable[Path], names: Iterable[str], *, required: bool = True) -> Path | None:
    names = tuple(names)
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
        for parent in (root / "model_assets", root / "runtime", root / "runtime" / "XPolicyLab"):
            for name in names:
                candidate = parent / name
                if candidate.is_file():
                    return candidate
    if required:
        raise FileNotFoundError(f"cannot locate asset {names!r}; checked {[str(root) for root in roots]}")
    return None


def _find_adapter(roots: Iterable[Path]) -> Path:
    explicit_names = (
        "runtime/XPolicyLab/policy/TwinTimeFM/checkpoints/twintimefm/twintimefm_adapter.safetensors",
        "runtime/XPolicyLab/policy/TwinTimeFM/checkpoints/twintimefm_adapter.safetensors",
        "XPolicyLab/policy/TwinTimeFM/checkpoints/twintimefm/twintimefm_adapter.safetensors",
        "XPolicyLab/policy/TwinTimeFM/checkpoints/twintimefm_adapter.safetensors",
    )
    for root in roots:
        for relative in explicit_names:
            candidate = root / relative
            if candidate.is_file():
                return candidate
        # The release has used both a nested ``twintimefm/`` folder and a
        # flat checkpoint folder.  Limit recursive lookup to these known
        # runtime roots so a model asset scan cannot traverse the workspace.
        if root.is_dir() and root.name in {"twintimefm_official_retest_20261019", "runtime", "XPolicyLab"}:
            matches = sorted(root.rglob("twintimefm_adapter.safetensors"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "DuoFlow FT2T adapter twintimefm_adapter.safetensors was not found; "
        "the CPU factory refuses to silently fall back to Clean-only features"
    )


def _load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise NativeInterfaceError(f"cannot import Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _planner_prompt_transform(prompt_transform: Any, *, max_length: int = _PLANNER_MAX_LENGTH) -> Any:
    """Clone the official tokenizer transform with a planner-only budget."""

    try:
        clone = copy.deepcopy(prompt_transform)
    except Exception:
        # Tokenizer internals may not support deepcopy in older transformers;
        # a shallow copy still keeps attribute writes off the actor object.
        clone = copy.copy(prompt_transform)
    clone.max_length = max(int(getattr(prompt_transform, "max_length", 0)), int(max_length))
    clone.is_train = False
    if hasattr(clone, "sample_generator"):
        clone.sample_generator = None
    return clone


def _load_duoflow_adapter(policy: Any, adapter_path: Path, *, device: torch.device | str) -> dict[str, Any]:
    """Install the released FT2T LoRA into this independently loaded policy."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - depends on the model environment
        raise NativeInterfaceError("DuoFlow FT2T loading requires safetensors in the active model environment") from exc

    lora_module_path = next(
        (
            ancestor / "lora_policy.py"
            for ancestor in adapter_path.parents
            if (ancestor / "lora_policy.py").is_file()
        ),
        _LOCAL_TWINTIME_ROOT / "runtime" / "XPolicyLab" / "policy" / "TwinTimeFM" / "lora_policy.py",
    )
    if not lora_module_path.is_file():
        raise FileNotFoundError(f"TwinTimePolicy source is missing: {lora_module_path}")
    module = _load_python_module(lora_module_path, "system2_duoflow_lora_policy")
    twin_policy = module.TwinTimePolicy(policy, mode="FT")
    config = twin_policy.configure_trainable(rank=8, alpha=16.0)
    state = {"lora": load_file(str(adapter_path), device="cpu"), "adapter": {}}
    if sum(value.numel() for value in state["lora"].values()) != 4_046_848:
        raise NativeInterfaceError(
            f"unexpected DuoFlow FT2T adapter size: {sum(value.numel() for value in state['lora'].values())}"
        )
    twin_policy.load_trainable_state_dict(state, strict=True)
    twin_policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return {
        "policy": twin_policy,
        "config": config,
        "adapter_path": str(adapter_path),
        "adapter_parameter_count": int(sum(value.numel() for value in state["lora"].values())),
    }


def _build_official_transforms(
    policy: Any,
    model_dir: Path,
    tokenizer_dir: Path,
    fast_tokenizer_dir: Path,
    norm_path: Path,
    *,
    embodiment_id: int,
    robot_type: str,
    source_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
        ImageTransform,
        Normalize,
        PadStatesAndActions,
        PromptTokenizerTransform,
        infer_end_effector_type_from_delta_mask,
    )

    inference_config = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
    image_cfg = inference_config["image_cfg"]
    prompt_cfg = inference_config["prompt_cfg"]
    norm_cfg = inference_config.get("norm_cfg", {})
    delta_mask = inference_config["delta_action_cfg"]["mask"][robot_type]
    present_img_keys = list(image_cfg["present_img_keys"])
    image_kwargs: dict[str, Any] = {
        "is_train": False,
        "resize_imgs_with_padding": image_cfg["resize_imgs_with_padding"],
        "present_img_keys": present_img_keys,
        "enable_image_aug": False,
        "enable_depth_img": bool(image_cfg.get("enable_depth_img", False)),
        "vlm_type": image_cfg.get("vlm_type", "paligemma2"),
    }
    if "high_res_cam" in inspect.signature(ImageTransform).parameters:
        image_kwargs["high_res_cam"] = image_cfg.get("high_res_cam")
    image_transform = ImageTransform(**image_kwargs)
    state_input_mode = str(inference_config.get("state_input_mode", prompt_cfg.get("state_input_mode", "prompt")))
    control_mode = "end effector" if embodiment_id in (3, 4, 5) else "joint"
    prompt_kwargs: dict[str, Any] = {
        "is_train": False,
        "tokenizer_model_path": str(tokenizer_dir),
        "fast_tokenizer_path": str(fast_tokenizer_dir),
        "max_length": prompt_cfg["max_length"],
        "discrete_state_input": prompt_cfg["discrete_state_input"],
        "discrete_state_input_for_pose_embodiments": prompt_cfg.get(
            "discrete_state_input_for_pose_embodiments", False
        ),
        "encode_action_input": prompt_cfg["encode_action_input"],
        "fast_token_vocab_mode": getattr(policy, "fast_token_vocab_mode", None),
        "fast_token_tail_skip_tokens": getattr(policy, "fast_token_tail_skip_tokens", 128),
        "fast_token_tail_vocab_size": getattr(policy, "fast_token_tail_vocab_size", None),
        "encoded_action_horizon": prompt_cfg.get("encoded_action_horizon"),
        "encode_sub_task_input": prompt_cfg["encode_sub_task_input"],
        "enable_control_mode_token": prompt_cfg.get("enable_control_mode_token", False),
        "control_mode_override": control_mode,
        "enable_end_effector_token": prompt_cfg.get("enable_end_effector_token", False),
        "end_effector_override": infer_end_effector_type_from_delta_mask(delta_mask),
        "vlm_type": prompt_cfg.get("vlm_type", "paligemma2"),
        "prefix_lm_text": prompt_cfg.get("prefix_lm_text", False),
        "image_attn": prompt_cfg.get("image_attn", "bidirectional"),
        "use_chat_template": prompt_cfg.get("use_chat_template", False),
        "chat_system_prompt": prompt_cfg.get("chat_system_prompt"),
        "prompt_filler_text": prompt_cfg.get("prompt_filler_text"),
        "resize_imgs_with_padding": image_cfg["resize_imgs_with_padding"],
        "state_input_mode": state_input_mode,
    }
    # This transform is owned by the new selector.  The deployed actor's
    # closure keeps its original prompt length; candidate descriptions,
    # retrieved memory and history are formatted into this longer context.
    prompt_kwargs["max_length"] = max(int(prompt_kwargs["max_length"]), _PLANNER_MAX_LENGTH)
    prompt_transform = PromptTokenizerTransform(**prompt_kwargs)
    if state_input_mode == "proprio_anchor" and prompt_transform.propri_token_id != policy.propri_token_id:
        raise NativeInterfaceError(
            f"official tokenizer/model <|propri|> mismatch: {prompt_transform.propri_token_id} vs {policy.propri_token_id}"
        )
    if hasattr(prompt_transform, "to"):
        prompt_transform.to(device)

    norm_data = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    state_normalize = Normalize(
        {embodiment_id: norm_data["observation.state"]},
        use_quantiles=bool(norm_cfg.get("use_quantiles", False)),
        enable_clamp=bool(norm_cfg.get("enable_clamp", False)),
    ).to(device)
    pad_states_and_actions = PadStatesAndActions(action_dim=int(policy.max_action_dim))
    return {
        "image_transform": image_transform,
        "prompt_transform": prompt_transform,
        "state_normalize": state_normalize,
        "pad_states_and_actions": pad_states_and_actions,
        "embodiment_id": int(embodiment_id),
        "robot_type": robot_type,
        "present_img_keys": present_img_keys,
        "state_input_mode": state_input_mode,
        "observation_memory_size": int(inference_config.get("observation_memory_size", 1)),
    }


def attach_native_preprocess(policy: Any) -> SimpleNamespace:
    """Expose official closure transforms without mutating ``policy``.

    ``get_policy`` binds its image/prompt/state transforms in the
    ``inference`` closure and the native method itself hardcodes CUDA moves.
    This wrapper only reads those closure locals so System2 can run the same
    transforms on an explicitly selected device.
    """

    inference = getattr(policy, "inference", None)
    if inference is None or not callable(inference):
        raise NativeInterfaceError("native policy must expose callable inference for closure inspection")
    function = inspect.unwrap(getattr(inference, "__func__", inference))
    try:
        closure = inspect.getclosurevars(function).nonlocals
    except (TypeError, ValueError) as exc:
        raise NativeInterfaceError("cannot inspect native inference closure") from exc

    required = (
        "image_transform",
        "prompt_transform",
        "state_normalize",
        "pad_states_and_actions",
        "embodiment_id",
        "present_img_keys",
    )
    values = {name: closure.get(name, getattr(policy, name, None)) for name in required}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise NativeInterfaceError(f"native inference closure lacks official fields: {missing}")
    actor_prompt_transform = values["prompt_transform"]
    values["prompt_transform"] = _planner_prompt_transform(actor_prompt_transform)
    optional_names = ("prepare_state_memory", "state_input_mode", "observation_memory_size")
    optional = {name: closure.get(name, getattr(policy, name, None)) for name in optional_names}
    try:
        device = next(policy.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    server = SimpleNamespace(
        policy=policy,
        device=device,
        system2_feature_dim=getattr(policy, "vlm_hidden_size", None),
        planner_features="frozen DuoFlow VLM including actor LoRA",
        actor_prompt_transform=actor_prompt_transform,
        planner_prompt_max_length=int(getattr(values["prompt_transform"], "max_length", _PLANNER_MAX_LENGTH)),
        **values,
        **{name: value for name, value in optional.items() if value is not None},
    )
    return server


def load_cpu_backend(
    reference_root: str | Path,
    *,
    device: torch.device | str = "cpu",
    scorer_checkpoint: str | Path | None = None,
    embodiment_id: int = 0,
    robot_type: str = "agilex_cobot_magic",
) -> System2Backend:
    """Load Clean + DuoFlow FT2T native features and official CPU transforms."""

    device = torch.device(device)
    roots = _asset_roots(reference_root)
    model_dir = _find_model_dir(roots)
    source_root = next(
        (
            candidate / "giga_models_src"
            for candidate in roots
            if (candidate / "giga_models_src" / "giga_models").is_dir()
        ),
        _LOCAL_GIGA_ROOT / "giga_models_src",
    )
    if not (source_root / "giga_models").is_dir():
        raise NativeInterfaceError(f"official source tree unavailable for CPU factory: {source_root}")
    tokenizer_dir = _find_file(roots, ("paligemma2_tokenizer/tokenizer.json",), required=False)
    if tokenizer_dir is not None:
        tokenizer_dir = tokenizer_dir.parent
    else:
        tokenizer_dir = _find_file(roots, ("paligemma2_tokenizer",), required=False)
    if tokenizer_dir is None or not tokenizer_dir.is_dir():
        tokenizer_dir = _LOCAL_GIGA_ROOT / "paligemma2_tokenizer"
    fast_tokenizer_dir = _find_file(roots, ("fast_tokenizer/tokenizer.json",), required=False)
    if fast_tokenizer_dir is not None:
        fast_tokenizer_dir = fast_tokenizer_dir.parent
    else:
        fast_tokenizer_dir = _find_file(roots, ("fast_tokenizer",), required=False)
    if fast_tokenizer_dir is None or not fast_tokenizer_dir.is_dir():
        fast_tokenizer_dir = _LOCAL_GIGA_ROOT / "fast_tokenizer"
    norm_path = model_dir / "norm_stats.json"
    if not norm_path.is_file():
        norm_path = _find_file(roots, ("norm_stats.json",))  # type: ignore[assignment]
    adapter_path = _find_adapter(roots)

    # The official loader casts the complete PaliGemma2/action-expert
    # backbone to bf16 before deployment.  The native dual-stream forward
    # also casts prefix embeddings to bf16, so leaving CPU weights in fp32
    # causes F.linear dtype mismatches before the first selector feature.
    policy = load_gigabrain07_policy(
        model_dir,
        device=device,
        dtype=torch.bfloat16,
        source_root=source_root,
    )
    adapter_info = _load_duoflow_adapter(policy, adapter_path, device=device)
    transform_info = _build_official_transforms(
        policy,
        model_dir,
        tokenizer_dir,
        fast_tokenizer_dir,
        norm_path,
        embodiment_id=embodiment_id,
        robot_type=robot_type,
        source_root=source_root,
        device=device,
    )
    native = SimpleNamespace(
        policy=policy,
        device=device,
        system2_feature_dim=getattr(policy, "vlm_hidden_size", None),
        planner_features="frozen DuoFlow VLM including actor LoRA",
        model_checkpoint=str(model_dir),
        actor_adapter=str(adapter_path),
        **transform_info,
    )
    backend = System2Backend.from_native(native, checkpoint=scorer_checkpoint)
    backend.planner_features = native.planner_features
    backend.model_checkpoint = native.model_checkpoint
    backend.actor_adapter = native.actor_adapter
    backend.adapter_parameter_count = adapter_info["adapter_parameter_count"]
    return backend


__all__ = ["attach_native_preprocess", "load_cpu_backend"]
