"""轻量的 RoboTwin clean50 -> GigaBrain 训练样本适配器。

该文件只依赖 PyTorch、PyArrow 和 Pillow，不依赖 ``giga_train``。源数据应是
``prepare_data.py`` 生成的单任务 staging 根目录，或等价的
LeRobot v2.1 根目录。``train`` 固定使用 episode 0--44，``val`` 使用
episode 45--49；不按 Python hash 或随机数生成样本 id。

示例（``tokenizer`` 是调用官方 PromptTokenizerTransform 的薄适配器）：

    ds = DemoDataset(path, split="train", norm_stats_path=release_norm,
                     tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(ds, batch_size=2,
                                         collate_fn=ds.collate_fn)

返回 batch 保留官方训练入口所需的 ``images``, ``img_masks``,
``lang_tokens``, ``lang_masks``, ``lang_att_masks``, ``emb_ids``,
``proprioception``、``agent_pos_mask``、``proprioception_present``；监督动作
同时以 ``actions32`` 和官方兼容别名 ``action`` 给出，形状为 ``[B,50,32]``。
``image_transform`` 若提供，应一次接收三路 camera dict 并返回官方
``(images, img_masks, image_transform_params)``；不提供时使用同一 resize/pad
的确定性 fallback。
"""

from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_SLOTS = {
    "dump_bin_bigbin": 0,
    "handover_block": 1,
    "lift_pot": 2,
}
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
PARQUET_COLUMNS = (
    "observation.state",
    "action",
    *CAMERA_KEYS,
    "frame_index",
    "episode_index",
    "task_index",
)
ACTION_HORIZON = 50
MODEL_DIM = 32
NATIVE_DIM = 14
DEFAULT_DELTA_MASK = (True, True, True, True, True, True, False,
                      True, True, True, True, True, True, False)


def stableexampleindex(task: str, episode_index: int, frame_index: int) -> str:
    """返回跨进程稳定且可读的样本索引。"""

    return f"{task}/episode_{int(episode_index):06d}/frame_{int(frame_index):06d}"


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on caller runtime
        raise RuntimeError("DemoDataset 需要 PyTorch") from exc
    return torch


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on caller runtime
        raise RuntimeError("DemoDataset 需要 pyarrow.parquet 读取 LeRobot Parquet") from exc
    return pq


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on caller runtime
        raise RuntimeError("DemoDataset 需要 numpy 解码 Parquet 数值列") from exc
    return np


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"无法读取 JSONL: {path}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {line_no} 行无效: {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_no} 行不是对象: {path}")
        rows.append(value)
    return rows


def _as_vector(value: Any, *, name: str, length: int = NATIVE_DIM):
    np = _numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} 必须是 [{length}]，得到 {array.shape}")
    return array


def _as_stats(stats: Mapping[str, Any], name: str) -> tuple[Any, Any]:
    np = _numpy()
    if "q01" not in stats or "q99" not in stats:
        raise ValueError(f"{name} 缺少官方 quantile 字段 q01/q99")
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    if low.shape != (NATIVE_DIM,) or high.shape != (NATIVE_DIM,):
        raise ValueError(f"{name} q01/q99 必须都是 [14]，得到 {low.shape}/{high.shape}")
    return low, high


class _QuantileNorm:
    def __init__(self, state_stats: Mapping[str, Any], action_stats: Mapping[str, Any]):
        self.state_low, self.state_high = _as_stats(state_stats, "observation.state")
        self.action_low, self.action_high = _as_stats(action_stats, "action")

    def apply(self, value, *, kind: str):
        torch = _torch()
        if kind == "state":
            low, high = self.state_low, self.state_high
        elif kind == "action":
            low, high = self.action_low, self.action_high
        else:
            raise ValueError(f"未知 norm kind: {kind}")
        low_t = torch.as_tensor(low, dtype=torch.float32, device=value.device)
        high_t = torch.as_tensor(high, dtype=torch.float32, device=value.device)
        return (value - low_t[..., : value.shape[-1]]) / (high_t[..., : value.shape[-1]] - low_t[..., : value.shape[-1]] + 1e-6) * 2.0 - 1.0


@dataclass(frozen=True)
class _EpisodeFile:
    episode_index: int
    paths: tuple[Path, ...]
    length: int


@dataclass(frozen=True)
class _Example:
    episode_index: int
    row_index: int
    frame_index: int
    task_index: int


def _decode_image(value: Any, *, dataset_root: Path):
    """把 LeRobot bytes/path image 转成 [C,H,W] float32 的 [0,1] tensor。"""

    torch = _torch()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on caller runtime
        raise RuntimeError("DemoDataset 需要 Pillow 解码 Parquet image bytes") from exc

    payload = value
    relative_path: str | None = None
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        relative_path = value.get("path")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not payload and relative_path:
        candidates = (
            dataset_root / relative_path,
            dataset_root / "images" / relative_path,
            dataset_root / "meta" / relative_path,
        )
        for candidate in candidates:
            if candidate.is_file():
                payload = candidate.read_bytes()
                break
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError(f"Parquet image 缺少 bytes/path: {relative_path!r}")
    with Image.open(io.BytesIO(bytes(payload))) as image:
        image = image.convert("RGB")
        np = _numpy()
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB image 必须是 HWC×3，得到 {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div(255.0)


def _resize_with_pad(image, width: int = 224, height: int = 224):
    """复现官方 resize_with_pad：先保持比例缩放，再以 0 填充。"""

    torch = _torch()
    import torch.nn.functional as F

    current_h, current_w = image.shape[-2:]
    ratio = max(current_w / width, current_h / height)
    resized_h = min(height, max(1, round(current_h / ratio)))
    resized_w = min(width, max(1, round(current_w / ratio)))
    resized = F.interpolate(
        image.unsqueeze(0), size=(resized_h, resized_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_h = int(height - resized_h)
    pad_w = int(width - resized_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return padded * 2.0 - 1.0


def _stack(values: Sequence[Any], name: str):
    torch = _torch()
    if any(value is None for value in values):
        raise ValueError(f"batch 缺少 {name}；请给 DemoDataset 传入 tokenizer")
    return torch.stack([value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values])


def _normalize_tokenizer_output(value: Any) -> dict[str, Any]:
    """接收官方 tokenizer tuple，或接收已命名的 token mapping。"""

    if isinstance(value, Mapping):
        output = dict(value)
        if "lang_tokens" not in output and "input_ids" in output:
            output["lang_tokens"] = output["input_ids"]
        if "lang_masks" not in output and "attention_mask" in output:
            output["lang_masks"] = output["attention_mask"]
        return output
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) < 2:
            raise ValueError("tokenizer tuple 至少需要 (lang_tokens, lang_masks)")
        names = (
            "lang_tokens",
            "lang_masks",
            "lang_att_masks",
            "lang_loss_masks",
            "fast_action_indicator",
            "subtask_indicator",
        )
        return {name: value[index] for index, name in enumerate(names) if index < len(value)}
    raise TypeError(f"tokenizer 必须返回 mapping 或 tuple，得到 {type(value).__name__}")


def collate_official(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把 ``DemoDataset`` 样本整理成官方 batch 命名和形状。"""

    if not samples:
        raise ValueError("不能 collate 空 batch")
    torch = _torch()
    batch: dict[str, Any] = {}
    batch["images"] = [
        torch.stack([sample["images"][camera] for sample in samples])
        for camera in range(len(samples[0]["images"]))
    ]
    batch["img_masks"] = [
        torch.stack([sample["img_masks"][camera] for sample in samples])
        for camera in range(len(samples[0]["img_masks"]))
    ]
    batch["image_masks"] = batch["img_masks"]
    batch["image_transform_params"] = [sample.get("image_transform_params", {}) for sample in samples]
    batch["lang_tokens"] = _stack([sample["lang_tokens"] for sample in samples], "lang_tokens")
    batch["lang_masks"] = _stack([sample["lang_masks"] for sample in samples], "lang_masks")
    for name in (
        "lang_att_masks",
        "lang_loss_masks",
        "fast_action_indicator",
        "subtask_indicator",
    ):
        values = [sample.get(name) for sample in samples]
        if any(value is not None for value in values):
            batch[name] = _stack(values, name)
    batch["emb_ids"] = _stack([sample["emb_ids"] for sample in samples], "emb_ids").to(torch.long)
    batch["embodiment_id"] = batch["emb_ids"]
    batch["proprioception"] = _stack([sample["proprioception"] for sample in samples], "proprioception")
    batch["agent_pos_mask"] = _stack([sample["agent_pos_mask"] for sample in samples], "agent_pos_mask")
    batch["proprioception_present"] = _stack(
        [sample["proprioception_present"] for sample in samples], "proprioception_present"
    ).to(torch.bool)
    batch["actions32"] = _stack([sample["actions32"] for sample in samples], "actions32")
    batch["action"] = batch["actions32"]
    batch["action_valid_mask"] = _stack(
        [sample["action_valid_mask"] for sample in samples], "action_valid_mask"
    ).to(torch.bool)
    batch["action_loss_mask"] = batch["action_valid_mask"]
    batch["action_is_pad"] = ~batch["action_valid_mask"]
    batch["stableexampleindex"] = [sample["stableexampleindex"] for sample in samples]
    batch["stable_example_index"] = batch["stableexampleindex"]
    for name in ("task", "instruction"):
        batch[name] = [sample[name] for sample in samples]
    for name in ("task_index", "episode_index", "frame_index"):
        batch[name] = _stack([sample[name] for sample in samples], name).to(torch.long)
    return batch


class DemoDataset:
    """读取一个 staged clean50 task，并产生官方 Giga 训练 batch 样本。"""

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        *,
        norm_stats_path: str | Path | None = None,
        tokenizer: Callable[[Mapping[str, Any]], Any] | None = None,
        image_transform: Callable[[Mapping[str, Any]], Any] | None = None,
        action_horizon: int = ACTION_HORIZON,
        model_dim: int = MODEL_DIM,
        delta_mask: Sequence[bool] = DEFAULT_DELTA_MASK,
        use_delta_actions: bool = True,
    ):
        self.root = Path(path)
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.action_horizon = int(action_horizon)
        self.model_dim = int(model_dim)
        self.delta_mask = tuple(bool(value) for value in delta_mask)
        self.use_delta_actions = bool(use_delta_actions)
        if self.action_horizon <= 0 or self.model_dim < NATIVE_DIM:
            raise ValueError("action_horizon 必须为正且 model_dim 至少为 14")
        if len(self.delta_mask) != NATIVE_DIM:
            raise ValueError("delta_mask 必须是 14 个布尔值")
        info = _read_json(self.root / "meta" / "info.json")
        if info.get("total_episodes") != 50:
            raise ValueError(f"clean50 dataset 必须有 50 episodes，得到 {info.get('total_episodes')!r}")
        self.info = info
        self.task = self.root.name.removesuffix("_clean50")
        self.task_slot = TASK_SLOTS.get(self.task, 99)
        tasks_path = self.root / "meta" / "tasks.jsonl"
        self.tasks = {}
        if tasks_path.is_file():
            for row in _read_jsonl(tasks_path):
                key = row.get("task_index", row.get("index", len(self.tasks)))
                self.tasks[int(key)] = str(row.get("task", row.get("language_instruction", self.task)))
        self.episodes = self._discover_episodes()
        self._episode_by_index = {episode.episode_index: episode for episode in self.episodes}
        if set(self._episode_by_index) != set(range(50)):
            raise ValueError(f"Parquet episode index 必须覆盖 0..49，得到 {sorted(self._episode_by_index)}")
        self.norm = _QuantileNorm(*self._load_norm_stats(norm_stats_path))
        if split == "train":
            selected = range(0, 45)
        elif split in ("val", "validation"):
            selected = range(45, 50)
        elif split == "all":
            selected = range(0, 50)
        else:
            raise ValueError("split 只能是 train、val 或 all")
        self.examples = [
            _Example(ep.episode_index, row_index, frame_index, task_index)
            for ep in self.episodes
            if ep.episode_index in selected
            for row_index, (frame_index, task_index) in enumerate(self._episode_index_rows(ep))
        ]

    def _load_norm_stats(self, path: str | Path | None) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        candidates = []
        if path is not None:
            candidates.append(Path(path))
        candidates.extend((self.root / "norm_stats.json", self.root.parent / "norm_stats.json"))
        stats_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if stats_path is None:
            raise FileNotFoundError(
                "未找到发布 Giga norm_stats.json；请通过 norm_stats_path 传入当前 O release 文件，"
                "不要使用 pi0 assets/{task}/norm_stats.json"
            )
        payload = _read_json(stats_path)
        stats = payload.get("norm_stats", payload)
        if not isinstance(stats, Mapping):
            raise ValueError(f"norm_stats 顶层无效: {stats_path}")
        state_stats = stats.get("observation.state")
        action_stats = stats.get("action")
        if not isinstance(state_stats, Mapping) or not isinstance(action_stats, Mapping):
            raise ValueError(f"norm_stats 必须包含 observation.state/action: {stats_path}")
        return state_stats, action_stats

    def _discover_episodes(self) -> list[_EpisodeFile]:
        pq = _pyarrow_parquet()
        grouped: dict[int, list[Path]] = defaultdict(list)
        for path in sorted(self.root.glob("data/**/*.parquet")):
            match = re.search(r"episode_(\d+)\.parquet$", path.name)
            if match is None:
                raise ValueError(f"无法从 parquet 文件名解析 episode: {path}")
            grouped[int(match.group(1))].append(path)
        if len(grouped) != 50:
            raise ValueError(f"clean50 必须发现 50 个 episode parquet，得到 {len(grouped)}")
        episodes = []
        for episode_index in sorted(grouped):
            paths = tuple(grouped[episode_index])
            length = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)
            if length <= 0:
                raise ValueError(f"episode {episode_index} 没有 frame: {paths}")
            episodes.append(_EpisodeFile(episode_index, paths, length))
        return episodes

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_index_rows(paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        pq = _pyarrow_parquet()
        rows: list[dict[str, Any]] = []
        for path in paths:
            table = pq.read_table(path, columns=["frame_index", "task_index"])
            rows.extend(table.to_pylist())
        return tuple(rows)

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_rows(paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        pq = _pyarrow_parquet()
        rows: list[dict[str, Any]] = []
        for path in paths:
            table = pq.read_table(path, columns=list(PARQUET_COLUMNS))
            rows.extend(table.to_pylist())
        return tuple(rows)

    def _episode_rows(self, episode: _EpisodeFile) -> tuple[dict[str, Any], ...]:
        return self._read_rows(tuple(str(path) for path in episode.paths))

    def _episode_index_rows(self, episode: _EpisodeFile) -> list[tuple[int, int]]:
        rows = self._read_index_rows(tuple(str(path) for path in episode.paths))
        result = []
        for row_index, row in enumerate(rows):
            frame_index = int(row.get("frame_index", row_index))
            task_index = int(row.get("task_index", 0))
            result.append((frame_index, task_index))
        return result

    def __len__(self) -> int:
        return len(self.examples)

    def _task_text(self, task_index: int) -> str:
        return self.tasks.get(task_index, self.task)

    def _tokenize(self, *, task: str, state, actions) -> dict[str, Any]:
        if self.tokenizer is None:
            return {
                "lang_tokens": None,
                "lang_masks": None,
                "lang_att_masks": None,
            }
        context = {
            "task": task,
            "observation.state": state,
            "action": actions,
            "embodiment_id": 0,
            "robot_type": "agilex_cobot_magic",
        }
        return _normalize_tokenizer_output(self.tokenizer(context))

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch()
        example = self.examples[index]
        episode = self._episode_by_index[example.episode_index]
        rows = self._episode_rows(episode)
        row = rows[example.row_index]
        raw_state = _as_vector(row["observation.state"], name="observation.state")
        raw_actions = []
        valid = []
        for offset in range(self.action_horizon):
            row_index = min(example.row_index + offset, len(rows) - 1)
            raw_actions.append(_as_vector(rows[row_index]["action"], name="action"))
            valid.append(example.row_index + offset < len(rows))
        np = _numpy()
        action_array = np.stack(raw_actions, axis=0)
        if self.use_delta_actions:
            delta = np.asarray(self.delta_mask, dtype=bool)
            action_array[:, delta] -= raw_state[delta]
        state14 = self.norm.apply(torch.as_tensor(raw_state), kind="state")
        action14 = self.norm.apply(torch.as_tensor(action_array), kind="action")
        state32 = torch.zeros(self.model_dim, dtype=torch.float32)
        state32[:NATIVE_DIM] = state14
        actions32 = torch.zeros(self.action_horizon, self.model_dim, dtype=torch.float32)
        actions32[:, :NATIVE_DIM] = action14
        task_text = self._task_text(example.task_index)
        tokens = self._tokenize(task=task_text, state=state14, actions=action14)

        decoded_images = {
            camera_key: _decode_image(row[camera_key], dataset_root=self.root)
            for camera_key in CAMERA_KEYS
        }
        if self.image_transform is not None:
            transformed = self.image_transform(decoded_images)
            if not isinstance(transformed, (tuple, list)) or len(transformed) != 3:
                raise ValueError(
                    "image_transform 必须返回官方 (images, img_masks, image_params) 三元组"
                )
            images, img_masks, image_params = transformed
            images = list(images)
            img_masks = list(img_masks)
            if len(images) != len(CAMERA_KEYS) or len(img_masks) != len(CAMERA_KEYS):
                raise ValueError("image_transform 必须返回三路 images 和三路 img_masks")
        else:
            images = [_resize_with_pad(decoded_images[camera_key]) for camera_key in CAMERA_KEYS]
            img_masks = [torch.tensor(True, dtype=torch.bool) for _ in images]
            image_params = {}
        agent_pos_mask = torch.zeros(1, self.model_dim, dtype=torch.float32)
        agent_pos_mask[:, :NATIVE_DIM] = 1.0
        stable_index = stableexampleindex(self.task, example.episode_index, example.frame_index)
        sample = {
            "images": images,
            "img_masks": img_masks,
            "image_transform_params": image_params,
            "emb_ids": torch.tensor(0, dtype=torch.long),
            "proprioception": state32.unsqueeze(0),
            "agent_pos_mask": agent_pos_mask,
            "proprioception_present": torch.tensor(True, dtype=torch.bool),
            "actions32": actions32,
            "action_valid_mask": torch.as_tensor(valid, dtype=torch.bool),
            "stableexampleindex": stable_index,
            "stable_example_index": stable_index,
            "task": task_text,
            "instruction": task_text,
            "task_index": torch.tensor(example.task_index, dtype=torch.long),
            "episode_index": torch.tensor(example.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(example.frame_index, dtype=torch.long),
        }
        sample.update(tokens)
        return sample

    @staticmethod
    def collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_official(samples)


__all__ = ["DemoDataset", "collate_official", "stableexampleindex"]
