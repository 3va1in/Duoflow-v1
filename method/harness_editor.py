"""Finite harness edits and a small persistent contextual edit policy.

The editor only emits four JSON patches.  It never executes arbitrary source
or invents a new low-level skill.  ``FixedEditPolicy`` is the non-learning
control; ``EditPolicy`` learns patch preferences from matched A/B/C outcomes
where ``delta = success(new) - success(old)`` is -1, 0, or 1.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from plan_state import PlanState, normalize_patch


_PATCHES = (
    {"patch_id": "no_change", "operation": "NO_CHANGE"},
    {"patch_id": "guard_2", "operation": "SET_GUARD", "min_attempts": 2, "stagnation_chunks": 2},
    {"patch_id": "guard_3", "operation": "SET_GUARD", "min_attempts": 3, "stagnation_chunks": 3},
    {"patch_id": "backtrack_1", "operation": "RETRY_OR_BACKTRACK", "min_attempts": 1, "max_backtracks": 1},
)
PATCH_IDS = tuple(item["patch_id"] for item in _PATCHES)


def candidate_patches() -> list[dict[str, Any]]:
    """Return fresh copies in the fixed, auditable candidate order."""

    return [normalize_patch(item) for item in _PATCHES]


def validate_patch(patch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a patch and return its canonical JSON representation."""

    return normalize_patch(patch)


def _as_floats(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)):
        return []
    if isinstance(value, Sequence):
        result: list[float] = []
        for item in value:
            result.extend(_as_floats(item))
        return result
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if math.isfinite(number) else []


def _bucket(text: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def context_vector(context: Mapping[str, Any] | None, dimension: int = 16) -> list[float]:
    """Build a deterministic low-dimensional vector from editor context.

    A 2304-D shared-VLM summary may be supplied under ``shared_feature`` or
    ``features``; hashed signed buckets avoid a torch dependency and preserve a
    stable interface for the controller.  Task/condition and evidence fields
    are always included, so a context with no VLM feature remains usable.
    """

    if dimension < 4:
        raise ValueError("dimension must be at least four")
    context = dict(context or {})
    vector = [0.0] * dimension
    vector[0] = 1.0
    raw = context.get("shared_feature", context.get("projection", context.get("features")))
    values = _as_floats(raw)
    if values:
        scale = math.sqrt(float(len(values)))
        for position, value in enumerate(values):
            if not math.isfinite(value):
                continue
            index, sign = _bucket(f"feature:{position}", dimension)
            vector[index] += sign * value / max(scale, 1.0)
    for key in (
        "task", "task_name", "condition", "failure_source", "stagnation_state",
        "semantic_state", "reference_success", "prior_failures",
    ):
        value = context.get(key)
        if value is not None:
            index, sign = _bucket(f"{key}:{value}", dimension)
            vector[index] += sign
    evidence = context.get("evidence")
    if isinstance(evidence, Mapping):
        for key in ("stagnated", "final_frame_missing", "stagnation_chunks"):
            if key in evidence:
                index, sign = _bucket(f"evidence:{key}:{evidence[key]}", dimension)
                vector[index] += sign * (1.0 if key != "stagnation_chunks" else float(evidence[key] or 0))
    # Keep the scale bounded for stable ridge scores while retaining signs.
    norm = math.sqrt(sum(item * item for item in vector))
    if norm > 0:
        vector = [item / norm for item in vector]
    return vector


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Small Gaussian solver used for the 16-D ridge systems."""

    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[pivot][col] = 1e-12
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[col])
                ]
    return [augmented[row][-1] for row in range(size)]


class FixedEditPolicy:
    """A deterministic, non-learning round-robin edit policy.

    The A arm can pass an explicit ``NO_CHANGE`` patch.  This class is the B
    arm: it explores the fixed finite candidates in order and never updates
    its preference from outcomes.
    """

    def __init__(self) -> None:
        self._cursor = 0

    def choose(self, context: Mapping[str, Any] | None = None, candidates: Iterable[Mapping[str, Any]] | None = None, explore: bool = True) -> dict[str, Any]:
        options = [validate_patch(item) for item in (candidates or candidate_patches())]
        if not options:
            raise ValueError("at least one candidate patch is required")
        chosen = options[self._cursor % len(options)]
        self._cursor += 1
        return chosen

    def predict(self, context: Mapping[str, Any] | None = None, candidates: Iterable[Mapping[str, Any]] | None = None, explore: bool = True) -> dict[str, Any]:
        return self.choose(context, candidates, explore=explore)

    def update(self, context: Mapping[str, Any] | None, patch_id: str, reward: int) -> None:
        # Fixed policy is deliberately independent of learning.
        return None


class EditPolicy:
    """Persistable contextual ridge/UCB policy for finite harness patches."""

    def __init__(self, context_dim: int = 16, ridge: float = 1.0, exploration: float = 0.15):
        if context_dim < 4 or ridge <= 0 or exploration < 0:
            raise ValueError("invalid edit-policy hyperparameters")
        self.context_dim = int(context_dim)
        self.ridge = float(ridge)
        self.exploration = float(exploration)
        self.update_count = 0
        self._stats: dict[str, dict[str, Any]] = {}
        self.experiences: list[dict[str, Any]] = []
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._stats = {}
        for patch_id in PATCH_IDS:
            matrix = [[0.0] * self.context_dim for _ in range(self.context_dim)]
            for index in range(self.context_dim):
                matrix[index][index] = self.ridge
            self._stats[patch_id] = {
                "A": matrix,
                "b": [0.0] * self.context_dim,
                "n": 0,
                "updates": 0,
                "reward_sum": 0.0,
                "negative_experiences": [],
            }

    def _vector(self, context: Mapping[str, Any] | None) -> list[float]:
        return context_vector(context, self.context_dim)

    def _score(self, patch_id: str, vector: list[float], *, explore: bool = True) -> float:
        stats = self._stats[patch_id]
        theta = _solve(stats["A"], stats["b"])
        mean = sum(left * right for left, right in zip(theta, vector))
        solved = _solve(stats["A"], vector)
        variance = max(0.0, sum(left * right for left, right in zip(vector, solved)))
        return mean + (self.exploration if explore else 0.0) * math.sqrt(variance)

    def predict(self, context: Mapping[str, Any] | None = None, candidates: Iterable[Mapping[str, Any]] | None = None, explore: bool = True) -> dict[str, Any]:
        options = [validate_patch(item) for item in (candidates or candidate_patches())]
        vector = self._vector(context)
        scored = [(self._score(item["patch_id"], vector, explore=explore), index, item) for index, item in enumerate(options)]
        _, _, chosen = max(scored, key=lambda row: (row[0], -row[1]))
        return dict(chosen, policy_score=float(self._score(chosen["patch_id"], vector, explore=explore)))

    def choose(self, context: Mapping[str, Any] | None = None, candidates: Iterable[Mapping[str, Any]] | None = None, explore: bool = True) -> dict[str, Any]:
        return self.predict(context, candidates, explore=explore)

    def update(self, context: Mapping[str, Any] | None, patch_id: str, reward: int) -> dict[str, Any]:
        if patch_id not in PATCH_IDS:
            raise ValueError(f"unsupported patch_id: {patch_id!r}")
        delta = reward
        if isinstance(delta, bool) or delta not in {-1, 0, 1}:
            raise ValueError("reward must be -1, 0, or 1")
        vector = self._vector(context)
        stats = self._stats[patch_id]
        for row in range(self.context_dim):
            stats["b"][row] += float(delta) * vector[row]
            for col in range(self.context_dim):
                stats["A"][row][col] += vector[row] * vector[col]
        stats["n"] += 1
        stats["updates"] += 1
        stats["reward_sum"] += float(delta)
        self.update_count += 1
        record = {
            "patch_id": patch_id,
            "delta": int(delta),
            "negative": delta < 0,
            "context_vector": vector,
            "update_index": self.update_count,
            "timestamp": time.time(),
        }
        self.experiences.append(record)
        if delta < 0:
            stats["negative_experiences"].append(record)
        return dict(record)

    def stats(self, patch_id: str | None = None) -> dict[str, Any]:
        if patch_id is not None:
            if patch_id not in PATCH_IDS:
                raise ValueError(f"unsupported patch_id: {patch_id!r}")
            item = self._stats[patch_id]
            return {key: deepcopy(value) for key, value in item.items() if key not in {"A", "b"}}
        return {key: self.stats(key) for key in PATCH_IDS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "policy_type": "contextual_ridge_ucb",
            "context_dim": self.context_dim,
            "ridge": self.ridge,
            "exploration": self.exploration,
            "update_count": self.update_count,
            "stats": deepcopy(self._stats),
            "experiences": deepcopy(self.experiences),
        }

    def serialize(self) -> dict[str, Any]:
        return self.to_dict()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EditPolicy":
        policy = cls(
            context_dim=int(payload.get("context_dim", 16)),
            ridge=float(payload.get("ridge", 1.0)),
            exploration=float(payload.get("exploration", 0.15)),
        )
        stats = payload.get("stats")
        if isinstance(stats, Mapping):
            for patch_id in PATCH_IDS:
                if isinstance(stats.get(patch_id), Mapping):
                    source = stats[patch_id]
                    for key in ("A", "b", "n", "updates", "reward_sum", "negative_experiences"):
                        if key in source:
                            policy._stats[patch_id][key] = deepcopy(source[key])
        policy.update_count = int(payload.get("update_count", len(payload.get("experiences", []))))
        policy.experiences = deepcopy(list(payload.get("experiences", [])))
        return policy

    @classmethod
    def load(cls, path: str | Path) -> "EditPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class HarnessEditor:
    """Convenience facade used by a controller for proposal/apply/update."""

    def __init__(self, policy: EditPolicy | FixedEditPolicy | None = None):
        self.policy = policy or EditPolicy()

    def candidates(self) -> list[dict[str, Any]]:
        return candidate_patches()

    def propose(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.choose(context, self.candidates(), explore=True)

    def choose(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.propose(context)

    def apply(self, plan: Sequence[Mapping[str, Any]] | PlanState, patch: Mapping[str, Any] | None) -> PlanState:
        normal = validate_patch(patch)
        if isinstance(plan, PlanState):
            return plan.with_patch(normal)
        return PlanState(plan, patch=normal)

    def apply_patch(self, plan: Sequence[Mapping[str, Any]] | PlanState, patch: Mapping[str, Any] | None) -> PlanState:
        return self.apply(plan, patch)

    def update(self, context: Mapping[str, Any] | None, patch_id: str, reward: int) -> dict[str, Any] | None:
        return self.policy.update(context, patch_id, reward)


# Names used by the independent RSI controller.  Keep the longer names above
# for readability in local tests and documentation.
patch_candidates = candidate_patches
EditBandit = EditPolicy


__all__ = [
    "PATCH_IDS",
    "candidate_patches",
    "patch_candidates",
    "validate_patch",
    "context_vector",
    "FixedEditPolicy",
    "EditPolicy",
    "EditBandit",
    "HarnessEditor",
]
