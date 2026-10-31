"""TwinTimeFM 的共同 FT5k 起点和单时刻 FT 对照训练。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

try:
    from . import runtime as backbone_runtime
    from .dataset import DemoDataset, collate_official
    from .lora_policy import TwinTimePolicy
except ImportError:  # 直接执行文件时使用同目录导入
    import runtime as backbone_runtime
    from dataset import DemoDataset, collate_official
    from lora_policy import TwinTimePolicy


ROOT = Path(__file__).resolve().parent
TASKS = ("dump_bin_bigbin", "handover_block", "lift_pot")
PROTOCOL = "gigabrain07_sr_adapter_v1"
NATIVE_DIM = 14
MODEL_DIM = 32
ACTION_STEPS = 50
ADAPTER_DTYPE = torch.float32
MODEL_DTYPE = torch.bfloat16
SEED = 0
LEARNING_RATE = 2.5e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 1e-10
SAVE_EVERY = 100
LOG_EVERY = 25


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _resolve_task_root(data_root: Path, task: str) -> Path:
    candidates = (data_root / task, data_root / f"{task}_clean50")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"缺少 {task} clean50 数据；候选路径: " + ", ".join(str(path) for path in candidates)
    )


def _build_loader(
    *,
    data_root: Path,
    norm_stats: Path,
    tokenizer: Any,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, dict[str, Any]]:
    datasets = []
    records = []
    for task in TASKS:
        task_root = _resolve_task_root(data_root, task)
        dataset = DemoDataset(
            task_root,
            split="train",
            norm_stats_path=norm_stats,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )
        datasets.append(dataset)
        records.append({"task": task, "path": str(task_root.resolve()), "examples": len(dataset)})

    combined = ConcatDataset(datasets)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(SEED)
    permutation = torch.randperm(len(combined), generator=order_generator).tolist()
    ordered = Subset(combined, permutation)

    # 该 generator 与 flow 全局 RNG 分离，恢复时重建 iterator 也保持稳定。
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(SEED)
    loader = DataLoader(
        ordered,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        collate_fn=collate_official,
    )
    if len(loader) == 0:
        raise ValueError("训练数据不足一个完整 batch")
    manifest = {
        "root": str(data_root.resolve()),
        "norm_stats": str(norm_stats.resolve()),
        "split": "train episodes 0:45 per task",
        "batch_size": int(batch_size),
        "drop_last": True,
        "tasks": records,
        "loader_batches": len(loader),
        "examples": len(ordered),
        "sample_order": {
            "kind": "torch.randperm",
            "seed": SEED,
            "concat_task_order": list(TASKS),
            "loader_shuffle": False,
            "indices": permutation,
        },
    }
    return loader, manifest


def _cycle(loader: DataLoader, iterator: Iterable[Mapping[str, Any]] | None = None):
    current = iter(loader) if iterator is None else iterator
    while True:
        try:
            yield next(current)
        except StopIteration:
            current = iter(loader)


def _skip_batches(loader: DataLoader, batches: int):
    iterator = iter(loader)
    for _ in range(batches % max(1, len(loader))):
        next(iterator)
    return _cycle(loader, iterator)


def _flow_inputs(
    batch: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions32 = batch["actions32"].to(device=device, dtype=torch.float32)
    if tuple(actions32.shape[1:]) != (ACTION_STEPS, MODEL_DIM):
        raise ValueError(f"actions32 必须是 [B,50,32]，得到 {tuple(actions32.shape)}")
    # 官方 joint forward 保持 32D；padding 的 noise 也按官方 mask 保持为零。
    # 先完整采样 32D 再置零，确保 FT/M 消耗完全相同的 flow 随机数。
    noise = torch.randn(actions32.shape, device=device, dtype=torch.float32)
    noise[..., NATIVE_DIM:] = 0.0
    uniform = torch.rand((actions32.shape[0],), device=device, dtype=torch.float32)
    beta_sample = uniform.pow(2.0 / 3.0)
    timestep = beta_sample * 0.999 + 0.001
    x_t = timestep[:, None, None] * noise + (1.0 - timestep[:, None, None]) * actions32
    target = noise - actions32
    return x_t, timestep, target


def _native_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_steps: torch.Tensor,
) -> torch.Tensor:
    if tuple(prediction.shape[1:]) != (ACTION_STEPS, NATIVE_DIM):
        raise ValueError(
            f"TwinTimePolicy 输出必须是 [B,50,14]，得到 {tuple(prediction.shape)}"
        )
    if tuple(target.shape[1:]) != (ACTION_STEPS, MODEL_DIM):
        raise ValueError(f"flow target 必须是 [B,50,32]，得到 {tuple(target.shape)}")
    target = target[..., :NATIVE_DIM]
    valid = valid_steps.to(device=prediction.device, dtype=torch.bool)
    if tuple(valid.shape) != prediction.shape[:2]:
        raise ValueError(f"action_valid_mask 必须是 [B,50]，得到 {tuple(valid.shape)}")
    mask = valid[:, :, None].expand_as(prediction)
    if int(mask.sum().item()) == 0:
        raise ValueError("batch 没有有效 action timestep")
    error = (prediction.float() - target.float()).square()
    return error.masked_select(mask).mean()


def _paired_native_loss(
    prediction_one: torch.Tensor,
    prediction_two: torch.Tensor,
    target: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算两个 flow 时刻的物理动作损失，并返回可记录指标。

    ``consistency_weight=0`` 保持 FT2T 的普通双时刻目标；EC 只在同一
    两次有梯度预测上增加速度一致性项。padding 维度仍由 ``_native_loss``
    统一排除，避免训练钩子改变原 flow 语义。
    """
    if prediction_one.shape != prediction_two.shape:
        raise ValueError(
            "两个 flow prediction 必须同形状，"
            f"得到 {tuple(prediction_one.shape)} / {tuple(prediction_two.shape)}"
        )
    fm_one = _native_loss(prediction_one, target, valid_steps)
    fm_two = _native_loss(prediction_two, target, valid_steps)
    fm_loss = 0.5 * (fm_one + fm_two)
    if float(consistency_weight) == 0.0:
        consistency_loss = fm_loss.new_zeros(())
    else:
        valid = valid_steps.to(device=prediction_one.device, dtype=torch.bool)
        if tuple(valid.shape) != prediction_one.shape[:2]:
            raise ValueError(
                f"action_valid_mask 必须是 [B,50]，得到 {tuple(valid.shape)}"
            )
        mask = valid[:, :, None].expand_as(prediction_one)
        if int(mask.sum().item()) == 0:
            raise ValueError("batch 没有有效 action timestep")
        difference = (prediction_one.float() - prediction_two.float()).square()
        consistency_loss = difference.masked_select(mask).mean()
    total = fm_loss + float(consistency_weight) * consistency_loss
    return total, {
        "loss": total.detach(),
        "flow_mse": fm_loss.detach(),
        "consistency_mse": consistency_loss.detach(),
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=MODEL_DTYPE)
    return nullcontext()


def _make_trainables_fp32(policy: TwinTimePolicy) -> None:
    for parameter in policy.parameters():
        if parameter.requires_grad and parameter.is_floating_point():
            parameter.data = parameter.data.float()


def _state(policy: TwinTimePolicy) -> dict[str, dict[str, torch.Tensor]]:
    raw_state = policy.trainable_state_dict()
    if not isinstance(raw_state, Mapping):
        raise TypeError("TwinTimePolicy.trainable_state_dict() 必须返回 mapping")
    if "lora" in raw_state or "adapter" in raw_state:
        groups = {
            "lora": raw_state.get("lora", {}),
            "adapter": raw_state.get("adapter", {}),
        }
    else:
        groups = {
            "lora": {
                name: value for name, value in raw_state.items() if "lora" in str(name).lower()
            },
            "adapter": {
                name: value for name, value in raw_state.items() if "adapter" in str(name).lower()
            },
        }
    return {
        group: {
            str(name): value.detach().cpu().clone()
            for name, value in values.items()
        }
        for group, values in groups.items()
    }


def _lora_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    values = state.get("lora", state)
    if not isinstance(values, Mapping):
        raise TypeError("LoRA state 必须是 mapping")
    result = {str(name): value for name, value in values.items() if "lora" in str(name).lower()}
    if not result:
        raise RuntimeError("TwinTimePolicy 未暴露 LoRA trainable state")
    return result


def _parameter_count(state: Mapping[str, Any]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, Mapping):
            total += sum(parameter.numel() for parameter in value.values())
        else:
            total += value.numel()
    return int(total)


def _publish_common(
    path: Path,
    state: Mapping[str, torch.Tensor],
    *,
    seed: int,
    official_model_source: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "protocol": PROTOCOL,
                "kind": "shared_lora_initialization",
                "seed": int(seed),
                "official_model_source": official_model_source,
                "adapter_state": {"lora": dict(state)},
                "parameter_names": sorted(state),
            },
            temporary,
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL or payload.get("kind") != "shared_lora_initialization":
        raise ValueError(f"共同 LoRA 初始化协议不匹配: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"共同 LoRA 初始化 seed 不匹配: {path}")
    common_state = payload.get("adapter_state")
    if isinstance(common_state, Mapping) and "lora" in common_state:
        common_values = common_state["lora"]
    else:
        common_values = common_state
    if not isinstance(common_values, Mapping) or not common_values:
        raise ValueError(f"共同 LoRA 初始化为空: {path}")
    if any("lora" not in str(name).lower() for name in common_values):
        raise ValueError("共同初始化文件只能包含 LoRA 参数")
    return payload


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _device_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_dtype": str(MODEL_DTYPE),
        "adapter_dtype": str(ADAPTER_DTYPE),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        result.update({"gpu_name": None, "gpu_uuid": None})
        return result
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    result["gpu_name"] = str(properties.name)
    result["gpu_uuid"] = str(getattr(properties, "uuid", None))
    result["cuda_device_index"] = int(index)
    return result


def _observed_dtypes(policy: TwinTimePolicy) -> dict[str, Any]:
    frozen = next(
        (parameter for parameter in policy.parameters() if not parameter.requires_grad),
        None,
    )
    trainable = next(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        None,
    )
    return {
        "observed_base_parameter_dtype": None if frozen is None else str(frozen.dtype),
        "observed_adapter_parameter_dtype": None if trainable is None else str(trainable.dtype),
    }


def _checkpoint_payload(
    *,
    policy: TwinTimePolicy,
    optimizer: torch.optim.Optimizer,
    mode: str,
    seed: int,
    update: int,
    target_updates: int,
    batches_consumed: int,
    loader_batches: int,
    official_model_source: str,
    data_manifest: Mapping[str, Any],
    trainable_parameter_count: int,
    elapsed_seconds: float,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "adapter_state": _state(policy),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "target_updates": int(target_updates),
        "rng_state": _capture_rng(),
        "mode": mode,
        "seed": int(seed),
        "official_model_source": official_model_source,
        "data_cursor": {
            "batches_consumed": int(batches_consumed),
            "loader_batches": int(loader_batches),
        },
        "data_manifest": dict(data_manifest),
        "trainable_parameter_count": int(trainable_parameter_count),
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_hours_approx": float(elapsed_seconds / 3600.0 if runtime["device"].startswith("cuda") else 0.0),
        "runtime": dict(runtime),
    }


def _run_phase(
    *,
    policy: TwinTimePolicy,
    common_state: Mapping[str, torch.Tensor],
    loader: DataLoader,
    data_manifest: Mapping[str, Any],
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    official_model_source: str,
    common_path: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    phase_root = args.output_root / f"{mode}_seed{args.seed}"
    phase_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = phase_root / "checkpoint_last.pt"
    receipt_path = phase_root / "receipt.json"
    log_path = phase_root / "train.jsonl"
    if not args.resume and (checkpoint_path.exists() or receipt_path.exists() or log_path.exists()):
        raise FileExistsError(f"拒绝覆盖已有 {mode} 训练产物: {phase_root}")

    policy.mode = mode
    _seed_everything(args.seed)
    current_state = _state(policy)
    current_lora = _lora_state(current_state)
    missing_common = sorted(set(common_state) - set(current_lora))
    if missing_common:
        raise RuntimeError(f"当前 {mode} 缺少共同 LoRA 参数: {missing_common[:5]}")
    policy.load_trainable_state_dict({"lora": common_state}, strict=False)
    _make_trainables_fp32(policy)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("TwinTimePolicy 没有可训练参数")
    full_state = _state(policy)
    trainable_parameter_count = _parameter_count(full_state)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    start_update = 0
    batches_consumed = 0
    prior_elapsed = 0.0
    resume_payload: Mapping[str, Any] | None = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume 找不到 checkpoint: {checkpoint_path}")
        resume_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume_payload.get("protocol") != PROTOCOL:
            raise ValueError(f"checkpoint 协议不匹配: {checkpoint_path}")
        if resume_payload.get("mode") != mode or int(resume_payload.get("seed", -1)) != args.seed:
            raise ValueError(f"checkpoint mode/seed 不匹配: {checkpoint_path}")
        policy.load_trainable_state_dict(resume_payload["adapter_state"], strict=True)
        _make_trainables_fp32(policy)
        optimizer.load_state_dict(resume_payload["optimizer"])
        _optimizer_to_device(optimizer, device)
        start_update = int(resume_payload.get("update", 0))
        cursor = resume_payload.get("data_cursor", {})
        batches_consumed = int(cursor.get("batches_consumed", start_update * args.gradient_accumulation))
        prior_elapsed = float(resume_payload.get("elapsed_seconds", 0.0))
        if resume_payload.get("rng_state") is not None:
            # 先推进独立 DataLoader iterator，再恢复 RNG，避免改变 flow 噪声/时间序列。
            pass
        if start_update > args.updates:
            raise ValueError(f"checkpoint update {start_update} exceeds target {args.updates}")

    # 官方基础模型保持 eval；梯度只经过适配参数。
    if hasattr(policy, "model"):
        policy.model.eval()
    iterator = _skip_batches(loader, batches_consumed)
    if resume_payload is not None and resume_payload.get("rng_state") is not None:
        _restore_rng(resume_payload["rng_state"])

    started = time.time()
    receipt: dict[str, Any] = {
        "protocol": PROTOCOL,
        "mode": mode,
        "seed": int(args.seed),
        "status": "RUNNING",
        "run_complete": False,
        "updates": int(start_update),
        "target_updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "gradient_accumulation": int(args.gradient_accumulation),
        "effective_batch": int(args.batch_size * args.gradient_accumulation),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": EPS,
        "weight_decay": WEIGHT_DECAY,
        "data_manifest": dict(data_manifest),
        "sample_order": dict(data_manifest["sample_order"]),
        "common_initialization": str(common_path.resolve()),
        "common_lora_parameter_count": _parameter_count(common_state),
        "official_model_source": official_model_source,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_names": sorted(
            name for group in full_state.values() for name in group
        ),
        "shared_lora_only_initialization": True,
        "old_execution_included": False,
        "rl_data_included": False,
        "runtime": dict(runtime),
        "checkpoint": str(checkpoint_path.resolve()),
        "started_at": started,
    }
    _write_json(receipt_path, receipt)
    log_mode = "a" if args.resume and log_path.exists() else "w"
    log = log_path.open(log_mode, encoding="utf-8")
    last_loss: float | None = None
    try:
        for update in range(start_update + 1, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            batch_ids: list[str] = []
            losses: list[float] = []
            for _ in range(args.gradient_accumulation):
                raw_batch = next(iterator)
                batch_ids.extend(str(value) for value in raw_batch["stableexampleindex"])
                batch = _move_to_device(raw_batch, device)
                noisy_actions, timestep, target = _flow_inputs(batch, device)
                with _autocast(device):
                    prediction = policy.training_prediction(batch, noisy_actions, timestep)
                loss = _native_loss(prediction, target, batch["action_valid_mask"])
                (loss / args.gradient_accumulation).backward()
                losses.append(float(loss.detach().cpu()))
            optimizer.step()
            last_loss = float(sum(losses) / len(losses))
            batches_consumed += args.gradient_accumulation
            if update % LOG_EVERY == 0 or update == 1 or update == args.updates:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = {
                    "event": "update",
                    "mode": mode,
                    "seed": int(args.seed),
                    "update": update,
                    "target_updates": int(args.updates),
                    "loss": last_loss,
                    "update_seconds": time.perf_counter() - update_started,
                    "elapsed_seconds": prior_elapsed + time.time() - started,
                    "batch_ids": batch_ids,
                    "effective_batch": int(args.batch_size * args.gradient_accumulation),
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
            if update % SAVE_EVERY == 0 or update == args.updates:
                elapsed = prior_elapsed + time.time() - started
                payload = _checkpoint_payload(
                    policy=policy,
                    optimizer=optimizer,
                    mode=mode,
                    seed=args.seed,
                    update=update,
                    target_updates=args.updates,
                    batches_consumed=batches_consumed,
                    loader_batches=len(loader),
                    official_model_source=official_model_source,
                    data_manifest=data_manifest,
                    trainable_parameter_count=trainable_parameter_count,
                    elapsed_seconds=elapsed,
                    runtime=runtime,
                )
                payload["run_complete"] = update == args.updates
                _atomic_torch_save(payload, checkpoint_path)
                receipt.update(
                    {
                        "updates": update,
                        "last_loss": last_loss,
                        "checkpoint_update": update,
                        "elapsed_seconds": elapsed,
                        "gpu_hours_approx": elapsed / 3600.0 if device.type == "cuda" else 0.0,
                    }
                )
                _write_json(receipt_path, receipt)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"最终 checkpoint 未写入: {checkpoint_path}")
        elapsed = prior_elapsed + time.time() - started
        receipt.update(
            {
                "status": "COMPLETE",
                "run_complete": True,
                "updates": int(args.updates),
                "completed_at": time.time(),
                "elapsed_seconds": elapsed,
                "gpu_hours_approx": elapsed / 3600.0 if device.type == "cuda" else 0.0,
                "last_loss": last_loss,
            }
        )
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        elapsed = prior_elapsed + time.time() - started
        receipt.update(
            {
                "status": "FAILED",
                "run_complete": False,
                "updates": int(receipt.get("updates", start_update)),
                "elapsed_seconds": elapsed,
                "gpu_hours_approx": elapsed / 3600.0 if device.type == "cuda" else 0.0,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": time.time(),
            }
        )
        _write_json(receipt_path, receipt)
        raise
    finally:
        log.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("FT",), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "prepared_demo_clean")
    parser.add_argument("--norm-stats", type=Path, default=backbone_runtime.MODEL_ROOT / "norm_stats.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seed != SEED:
        raise SystemExit("本合同固定 seed=0")
    if args.updates <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise SystemExit("updates、batch-size、gradient-accumulation 必须为正")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"请求的 device 不可用: {device}")
    args.data_root = args.data_root.resolve()
    args.norm_stats = args.norm_stats.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    common_path = args.output_root / "common_initialization.pt"
    official_model_source = "backbone_runtime.load_official (frozen official O)"

    # 模型与 LoRA 初始化使用 seed0；共同初始化载入/发布后再次重置训练 RNG。
    _seed_everything(args.seed)
    model = backbone_runtime.load_official()
    model.to(device)
    transforms = backbone_runtime.build_demo_transforms(model)
    if not isinstance(transforms, (tuple, list)) or len(transforms) != 2:
        raise ValueError("build_demo_transforms(model) 必须返回 (image_transform, tokenizer)")
    image_transform, tokenizer = transforms
    loader, data_manifest = _build_loader(
        data_root=args.data_root,
        norm_stats=args.norm_stats,
        tokenizer=tokenizer,
        image_transform=image_transform,
        batch_size=args.batch_size,
        device=device,
    )
    _write_json(args.output_root / "DATA_MANIFEST.json", data_manifest)

    policy = TwinTimePolicy(model, mode=args.mode)
    policy.to(device)
    # 在 LoRA 注入前恢复固定 seed，保持共同初始化。
    _seed_everything(args.seed)
    train_config = policy.configure_trainable(8, 16)
    _make_trainables_fp32(policy)
    full_state = _state(policy)
    local_lora_state = _lora_state(full_state)
    common_payload = _publish_common(
        common_path,
        local_lora_state,
        seed=args.seed,
        official_model_source=official_model_source,
    )
    common_payload_state = common_payload["adapter_state"]
    if isinstance(common_payload_state, Mapping) and "lora" in common_payload_state:
        common_values = common_payload_state["lora"]
    else:
        common_values = common_payload_state
    common_state = {
        str(name): value.detach().cpu().clone()
        for name, value in common_values.items()
    }
    current_names = set(_lora_state(full_state))
    if not set(common_state).issubset(current_names):
        missing = sorted(set(common_state) - current_names)
        raise ValueError(f"共同 LoRA 初始化参数不在当前 {args.mode} policy 中: {missing[:5]}")
    policy.load_trainable_state_dict({"lora": common_state}, strict=False)
    _make_trainables_fp32(policy)
    runtime = _device_metadata(device)
    runtime.update(_observed_dtypes(policy))
    _write_json(
        args.output_root / "TRAIN_CONFIG.json",
        {
            "protocol": PROTOCOL,
            "mode": args.mode,
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch": args.batch_size * args.gradient_accumulation,
            "train_config": train_config,
            "common_initialization": str(common_path.resolve()),
            "official_model_source": official_model_source,
            "runtime": runtime,
        },
    )
    _run_phase(
        policy=policy,
        common_state=common_state,
        loader=loader,
        data_manifest=data_manifest,
        mode=args.mode,
        args=args,
        device=device,
        official_model_source=official_model_source,
        common_path=common_path,
        runtime=runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
