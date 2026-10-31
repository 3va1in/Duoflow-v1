"""Leak-safe, frozen runtime patch selection for the independent RSI pilot.

The selector consumes only public RGB/qpos transition statistics and the
predicted plan state. Episode outcomes are accepted only by the offline
``update_episode`` API, where they receive weak episode-level credit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness_editor import _solve, candidate_patches


VECTOR_DIM = 32
SCHEMA_VERSION = "runtime_edit_policy_v1"
PATCH_IDS = ("no_change", "guard_2", "guard_3", "backtrack_1")
_STATIC_DIM = 8
_OBS_DIM = 12
_STATE_DIM = 12


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _nonnegative_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, number)


def _bounded(value: Any, scale: float = 1.0) -> float:
    """Map an unbounded finite public statistic into ``[-1, 1]``."""

    value = _finite_float(value)
    scale = max(abs(float(scale)), 1e-12)
    return math.tanh(value / scale)


def _unit(value: Any, denominator: float) -> float:
    value = max(0.0, _finite_float(value))
    denominator = max(float(denominator), 1e-12)
    return max(0.0, min(1.0, value / denominator))


def _normalise_block(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in values))
    if norm <= 1e-12:
        return [0.0] * len(values)
    return [float(item) / norm for item in values]


def _hash_bucket(text: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dimension
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def _state_value(state: Any) -> float:
    if state == "changed":
        return 1.0
    if state == "unchanged":
        return -1.0
    return 0.0


def _safe_stats(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in keys:
        item = source.get(key)
        if key == "state":
            state = str(item) if item is not None else "unknown"
            result[key] = state if state in {"changed", "unchanged", "unknown"} else "unknown"
        elif key in {"cameras_compared", "dimensions_compared"}:
            result[key] = _nonnegative_int(item)
        else:
            result[key] = _finite_float(item)
    return result


def _history_attempt_count(history: Any, current_index: int) -> int:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return 0
    count = 0
    for item in history:
        if not isinstance(item, Mapping) or item.get("event") != "execution_attempted":
            continue
        raw_index = item.get("plan_index")
        if isinstance(raw_index, bool):
            continue
        try:
            item_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if item_index == current_index:
            count += 1
    return count


def safe_context(
    task: str,
    condition: str,
    plan_export: Mapping[str, Any] | None,
    transition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a strict public-observation/runtime-state context whitelist.

    ``plan_export`` is not copied wholesale because it contains the current
    patch and selection history. Attempt count is reconstructed only from
    ``execution_attempted`` entries, matching ``PlanState``'s counter.
    """

    plan_export = plan_export if isinstance(plan_export, Mapping) else {}
    transition = transition if isinstance(transition, Mapping) else {}
    raw_plan = plan_export.get("plan")
    plan = raw_plan if isinstance(raw_plan, Sequence) and not isinstance(raw_plan, (str, bytes)) else []
    current_index = _nonnegative_int(plan_export.get("current_index", plan_export.get("current_plan_index", 0)))
    plan_length = len(plan)
    current_step = (
        plan[current_index]
        if current_index < plan_length and isinstance(plan[current_index], Mapping)
        else {}
    )
    current_base_skill = current_step.get("base_skill_id", current_step.get("skill_id"))
    current_base_skill = str(current_base_skill) if current_base_skill else "unknown"
    rgb = _safe_stats(
        transition.get("rgb"),
        ("state", "mean_abs_delta", "changed_fraction", "cameras_compared"),
    )
    qpos = _safe_stats(
        transition.get("qpos"),
        ("state", "mean_abs_delta", "max_abs_delta", "dimensions_compared"),
    )
    stagnation_state = transition.get("stagnation_state", "unknown")
    stagnation_state = str(stagnation_state) if stagnation_state is not None else "unknown"
    if stagnation_state not in {"changed", "stagnated", "unknown"}:
        stagnation_state = "unknown"
    stagnated = transition.get("stagnated")
    if not isinstance(stagnated, bool):
        stagnated = None
    return {
        "task": str(task),
        "condition": str(condition),
        "plan_index": current_index,
        "attempt_count": _history_attempt_count(plan_export.get("selection_history"), current_index),
        "stagnation_state": stagnation_state,
        "stagnated": stagnated,
        "stagnation_chunks": _nonnegative_int(plan_export.get("stagnation_chunks")),
        "backtrack_count": _nonnegative_int(plan_export.get("backtrack_count")),
        "current_base_skill": current_base_skill,
        "plan_length": plan_length,
        "final_frame_missing": transition.get("final_frame_missing") is True,
        "rgb": rgb,
        "qpos": qpos,
    }


def context_vector(context: Mapping[str, Any] | None) -> list[float]:
    """Build the fixed deterministic 32-D runtime vector.

    The three blocks are independently L2-scaled: 8 static hashed dimensions,
    12 public RGB/qpos dimensions, and 12 predicted-plan dimensions.
    """

    context = dict(context or {})
    static = [0.0] * _STATIC_DIM
    for field in ("task", "condition", "current_base_skill"):
        value = context.get(field)
        if value is not None:
            index, sign = _hash_bucket(f"{field}:{value}", _STATIC_DIM)
            static[index] += sign
    static[0] += 1.0

    rgb = context.get("rgb") if isinstance(context.get("rgb"), Mapping) else {}
    qpos = context.get("qpos") if isinstance(context.get("qpos"), Mapping) else {}
    observation = [
        _unit(rgb.get("mean_abs_delta"), 1.0),
        _unit(rgb.get("changed_fraction"), 1.0),
        _unit(rgb.get("cameras_compared"), 8.0),
        _bounded(qpos.get("mean_abs_delta"), 1.0),
        _bounded(qpos.get("max_abs_delta"), 1.0),
        _unit(qpos.get("dimensions_compared"), 32.0),
        _state_value(rgb.get("state")),
        _state_value(qpos.get("state")),
        1.0 if context.get("stagnated") is True else 0.0,
        _unit(context.get("stagnation_chunks"), 3.0),
        1.0 if context.get("final_frame_missing") is True else 0.0,
        1.0 if context.get("stagnation_state") == "unknown" else 0.0,
    ]

    plan_length = _nonnegative_int(context.get("plan_length"))
    plan_index = _nonnegative_int(context.get("plan_index"))
    state = [
        _unit(plan_index, max(plan_length - 1, 1)),
        _unit(plan_length, 32.0),
        _unit(context.get("attempt_count"), 3.0),
        _unit(context.get("stagnation_chunks"), 3.0),
        _unit(context.get("backtrack_count"), 1.0),
        1.0 if _nonnegative_int(context.get("attempt_count")) > 0 else 0.0,
        1.0 if plan_index > 0 else 0.0,
        1.0 if plan_index + 1 < plan_length else 0.0,
        1.0 if plan_length > 0 else 0.0,
        1.0 if context.get("stagnation_state") == "stagnated" else 0.0,
        1.0 if context.get("stagnation_state") == "changed" else 0.0,
        1.0,
    ]
    return _normalise_block(static) + _normalise_block(observation) + _normalise_block(state)


def _is_stagnated(context: Mapping[str, Any]) -> bool:
    return bool(
        context.get("stagnated") is True
        or context.get("stagnation_state") == "stagnated"
        or _nonnegative_int(context.get("stagnation_chunks")) > 0
    )


class RuntimeEditPolicy:
    """Four-patch ridge policy with immutable evaluation mode."""

    def __init__(self, ridge: float = 1.0, *, frozen: bool = False):
        if not math.isfinite(float(ridge)) or float(ridge) <= 0:
            raise ValueError("ridge must be positive and finite")
        self.context_dim = VECTOR_DIM
        self.ridge = float(ridge)
        self._frozen = bool(frozen)
        self._stats: dict[str, dict[str, Any]] = {}
        self.training = {
            "credit_scope": "episode_weak_sign",
            "episodes": 0,
            "decision_updates": 0,
            "episode_weight_sum": 0.0,
            "raw_reward_sum": 0.0,
        }
        self.metadata: dict[str, Any] = {}
        self._reset_stats()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _reset_stats(self) -> None:
        self._stats = {}
        for patch_id in PATCH_IDS:
            matrix = [[0.0] * VECTOR_DIM for _ in range(VECTOR_DIM)]
            for index in range(VECTOR_DIM):
                matrix[index][index] = self.ridge
            self._stats[patch_id] = {
                "A": matrix,
                "b": [0.0] * VECTOR_DIM,
                "theta": [0.0] * VECTOR_DIM,
                "dirty": False,
                "updates": 0,
                "weight_sum": 0.0,
                "reward_sum": 0.0,
            }

    @staticmethod
    def _weak_credit(reward: Any) -> float:
        value = _finite_float(reward)
        return 1.0 if value > 0 else -1.0 if value < 0 else 0.0

    def _theta(self, patch_id: str) -> list[float]:
        stats = self._stats[patch_id]
        if not stats["dirty"]:
            return stats["theta"]
        theta = _solve(stats["A"], stats["b"])
        stats["theta"] = [float(item) for item in theta]
        stats["dirty"] = False
        return stats["theta"]

    def _score(self, patch_id: str, vector: Sequence[float]) -> float:
        theta = self._theta(patch_id)
        return float(sum(left * right for left, right in zip(theta, vector)))

    def score(self, context: Mapping[str, Any], patch_id: str) -> float:
        """Return the frozen ridge prediction for one canonical patch."""

        if patch_id not in PATCH_IDS:
            raise ValueError(f"unsupported patch_id: {patch_id!r}")
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping returned by safe_context")
        return self._score(patch_id, context_vector(context))

    def _eligible(self, context: Mapping[str, Any]) -> list[str]:
        plan_index = _nonnegative_int(context.get("plan_index"))
        plan_length = _nonnegative_int(context.get("plan_length"))
        attempts = _nonnegative_int(context.get("attempt_count"))
        eligible = ["no_change"]
        if attempts <= 0:
            return eligible
        has_next = plan_index + 1 < plan_length
        if has_next:
            if attempts < 2:
                eligible.append("guard_2")
            if attempts < 3:
                eligible.append("guard_3")
        if (
            _is_stagnated(context)
            and plan_index > 0
            and _nonnegative_int(context.get("backtrack_count")) < 1
        ):
            eligible.append("backtrack_1")
        return eligible

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Select a legal patch without updating the frozen policy."""

        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping returned by safe_context")
        vector = context_vector(context)
        scores = {patch_id: self._score(patch_id, vector) for patch_id in PATCH_IDS}
        eligible = self._eligible(context)
        baseline = scores["no_change"]
        if _nonnegative_int(context.get("attempt_count")) <= 0:
            selected = "no_change"
            reason = "first_or_unattempted"
        else:
            alternatives = [patch_id for patch_id in eligible if patch_id != "no_change"]
            selected = max(alternatives, key=lambda patch_id: scores[patch_id], default="no_change")
            if selected == "no_change" or not scores[selected] > baseline:
                selected = "no_change"
                reason = "no_positive_benefit"
            else:
                reason = "highest_positive_benefit"
        patch = next(dict(item) for item in candidate_patches() if item["patch_id"] == selected)
        return {
            "patch": patch,
            "selected": selected,
            "scores": scores,
            "eligible": eligible,
            "reason": reason,
            "context_schema_version": SCHEMA_VERSION,
        }

    def update(
        self,
        context: Mapping[str, Any],
        patch_id: str,
        reward: Any,
        *,
        weight: float = 1.0,
    ) -> None:
        """Apply one weighted weak episode-credit update offline."""

        if self._frozen:
            raise RuntimeError("RuntimeEditPolicy is frozen; online updates are disabled")
        if patch_id not in PATCH_IDS:
            raise ValueError(f"unsupported patch_id: {patch_id!r}")
        weight = _finite_float(weight)
        if weight < 0:
            raise ValueError("weight must be non-negative")
        vector = context_vector(context)
        credit = self._weak_credit(reward)
        stats = self._stats[patch_id]
        for row in range(VECTOR_DIM):
            stats["b"][row] += weight * credit * vector[row]
            for col in range(VECTOR_DIM):
                stats["A"][row][col] += weight * vector[row] * vector[col]
        stats["updates"] += 1
        stats["weight_sum"] += weight
        stats["reward_sum"] += weight * credit
        stats["dirty"] = True

    def update_episode(self, decisions: Sequence[Mapping[str, Any]], reward: Any) -> dict[str, Any]:
        """Give one weak episode credit to all decisions with weights summing to one."""

        if self._frozen:
            raise RuntimeError("RuntimeEditPolicy is frozen; online updates are disabled")
        items = list(decisions)
        if not items:
            raise ValueError("decisions must contain at least one decision")
        raw_weights = [_finite_float(item.get("weight", 1.0)) for item in items]
        if any(weight < 0 for weight in raw_weights):
            raise ValueError("decision weights must be non-negative")
        total = sum(raw_weights)
        weights = (
            [1.0 / len(items)] * len(items)
            if total <= 1e-12
            else [weight / total for weight in raw_weights]
        )
        for item, weight in zip(items, weights):
            self.update(item["context"], str(item["patch_id"]), reward, weight=weight)
        self.training["episodes"] += 1
        self.training["decision_updates"] += len(items)
        self.training["episode_weight_sum"] += sum(weights)
        self.training["raw_reward_sum"] += _finite_float(reward)
        return {"decision_weights": weights, "weight_sum": sum(weights), "credit": self._weak_credit(reward)}

    def to_dict(self) -> dict[str, Any]:
        patches = {}
        for patch_id, stats in self._stats.items():
            self._theta(patch_id)
            patches[patch_id] = {
                "theta": deepcopy(stats["theta"]),
                "A": deepcopy(stats["A"]),
                "b": deepcopy(stats["b"]),
                "updates": stats["updates"],
                "weight_sum": stats["weight_sum"],
                "reward_sum": stats["reward_sum"],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "policy_type": "runtime_contextual_ridge",
            "context_dim": VECTOR_DIM,
            "ridge": self.ridge,
            "patch_ids": list(PATCH_IDS),
            "frozen": self._frozen,
            "metadata": deepcopy(self.metadata),
            "training": deepcopy(self.training),
            "patches": patches,
        }

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            self.metadata = deepcopy(dict(metadata))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, frozen: bool = True) -> "RuntimeEditPolicy":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported RuntimeEditPolicy schema")
        if int(payload.get("context_dim", -1)) != VECTOR_DIM:
            raise ValueError("RuntimeEditPolicy requires a fixed 32-D context")
        if list(payload.get("patch_ids", [])) != list(PATCH_IDS):
            raise ValueError("patch set does not match RuntimeEditPolicy")
        policy = cls(float(payload.get("ridge", 1.0)), frozen=frozen)
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            policy.metadata = deepcopy(dict(metadata))
        policy.training = deepcopy(dict(payload.get("training", policy.training)))
        patches = payload.get("patches", {})
        if not isinstance(patches, Mapping):
            raise ValueError("serialized patches must be a mapping")
        for patch_id in PATCH_IDS:
            source = patches.get(patch_id)
            if not isinstance(source, Mapping):
                raise ValueError(f"serialized patch missing: {patch_id}")
            stats = policy._stats[patch_id]
            for key in ("A", "b", "theta"):
                value = source.get(key)
                if not isinstance(value, list):
                    raise ValueError(f"serialized {patch_id}.{key} is missing")
                stats[key] = deepcopy(value)
            stats["updates"] = int(source.get("updates", 0))
            stats["weight_sum"] = float(source.get("weight_sum", 0.0))
            stats["reward_sum"] = float(source.get("reward_sum", 0.0))
        return policy

    @classmethod
    def load(cls, path: str | Path, *, frozen: bool = True) -> "RuntimeEditPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), frozen=frozen)


__all__ = ["PATCH_IDS", "SCHEMA_VERSION", "VECTOR_DIM", "RuntimeEditPolicy", "context_vector", "safe_context"]
