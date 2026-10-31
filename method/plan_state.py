"""Evidence-aware state machine for a public-recipe execution plan.

The state tracks predicted planner progress only.  It never treats a chosen
skill as successfully executed and never reads simulator truth.  The optional
finite patch surface is intentionally small: ``SET_GUARD`` can hold a
predicted transition after measured public-observation stagnation, while
``RETRY_OR_BACKTRACK`` can expose one preceding step once.  Neither operation
adds environment steps or skips a prerequisite.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


_PATCH_IDS = {"no_change", "guard_2", "guard_3", "backtrack_1"}
_PATCH_OPERATIONS = {"NO_CHANGE", "SET_GUARD", "RETRY_OR_BACKTRACK"}


def normalize_patch(patch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and canonicalise the finite executable patch language."""

    if patch is None:
        patch = {"patch_id": "no_change", "operation": "NO_CHANGE"}
    if not isinstance(patch, Mapping):
        raise TypeError("patch must be a mapping or None")
    patch_id = patch.get("patch_id")
    # Older local model configs used ``{"op": "NO_CHANGE"}``; accept this
    # spelling at the boundary while keeping every exported patch canonical.
    if patch_id is None and patch.get("op") is not None:
        operation_hint = str(patch.get("op")).upper()
        patch_id = {
            "NO_CHANGE": "no_change",
            "SET_GUARD": "guard_2" if patch.get("min_attempts", 2) == 2 else "guard_3",
            "RETRY_OR_BACKTRACK": "backtrack_1",
        }.get(operation_hint)
    if patch_id not in _PATCH_IDS:
        raise ValueError(f"unsupported patch_id: {patch_id!r}")
    expected = {
        "no_change": "NO_CHANGE",
        "guard_2": "SET_GUARD",
        "guard_3": "SET_GUARD",
        "backtrack_1": "RETRY_OR_BACKTRACK",
    }[patch_id]
    operation = str(patch.get("operation", expected)).upper()
    if operation != expected or operation not in _PATCH_OPERATIONS:
        raise ValueError(f"patch {patch_id!r} must use operation {expected!r}")

    def bounded_int(name: str, default: int, lower: int, upper: int) -> int:
        value = patch.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be in [{lower}, {upper}]")
        return value

    if patch_id == "no_change":
        min_attempts = max_backtracks = stagnation_chunks = 0
    elif patch_id in {"guard_2", "guard_3"}:
        expected_chunks = 2 if patch_id == "guard_2" else 3
        # A guard is a finite minimum-attempt rule.  It must not wait on an
        # uncertain/stagnant semantic estimate forever.
        min_attempts = bounded_int("min_attempts", expected_chunks, 2, 3)
        max_backtracks = bounded_int("max_backtracks", 0, 0, 0)
        stagnation_chunks = bounded_int("stagnation_chunks", expected_chunks, 2, 3)
        if min_attempts != expected_chunks:
            raise ValueError(f"{patch_id} fixes min_attempts={expected_chunks}")
        if stagnation_chunks != expected_chunks:
            raise ValueError(f"{patch_id} fixes stagnation_chunks={expected_chunks}")
    else:
        min_attempts = bounded_int("min_attempts", 1, 1, 1)
        max_backtracks = bounded_int("max_backtracks", 1, 0, 1)
        stagnation_chunks = bounded_int("stagnation_chunks", 1, 1, 1)
        if max_backtracks != 1:
            raise ValueError("backtrack_1 permits exactly one backtrack")
    return {
        "patch_id": patch_id,
        "operation": operation,
        "min_attempts": min_attempts,
        "max_backtracks": max_backtracks,
        "stagnation_chunks": stagnation_chunks,
        "environment_step_budget_delta": 0,
    }


class PlanState:
    """Expose the current step and legal next/recovery steps."""

    def __init__(
        self,
        plan: Sequence[Mapping[str, Any]],
        patch: Mapping[str, Any] | None = None,
    ):
        if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)) or not plan:
            raise ValueError("plan must be a non-empty sequence")
        self._plan = [deepcopy(dict(step)) for step in plan]
        for expected, step in enumerate(self._plan):
            if step.get("plan_index") != expected:
                raise ValueError("plan_index values must be contiguous and start at zero")
            if not step.get("skill_id") or not step.get("execution_prompt"):
                raise ValueError("each plan step needs skill_id and execution_prompt")
        self._current_index = 0
        self._attempted_current = False
        self._history: list[dict[str, Any]] = []
        self._patch = normalize_patch(patch)
        self._latest_transition: dict[str, Any] | None = None
        self._stagnation_chunks = 0
        self._backtrack_count = 0

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def plan(self) -> list[dict[str, Any]]:
        return deepcopy(self._plan)

    @property
    def patch(self) -> dict[str, Any]:
        return deepcopy(self._patch)

    @property
    def latest_transition_evidence(self) -> dict[str, Any] | None:
        return deepcopy(self._latest_transition)

    def _copy_step(self, index: int) -> dict[str, Any]:
        return deepcopy(self._plan[index])

    def _consume_transition(self, evidence: Mapping[str, Any] | None) -> None:
        if evidence is None:
            return
        if not isinstance(evidence, Mapping):
            raise TypeError("transition_evidence must be a mapping or None")
        item = deepcopy(dict(evidence))
        self._latest_transition = item
        state = item.get("stagnation_state")
        stagnant = item.get("stagnated")
        if stagnant is True or state == "stagnated":
            reported = item.get("stagnation_chunks")
            if isinstance(reported, int) and reported > 1:
                self._stagnation_chunks = max(self._stagnation_chunks, reported)
            else:
                self._stagnation_chunks += 1
        elif stagnant is False or state == "changed":
            self._stagnation_chunks = 0
        else:
            self._stagnation_chunks = 0

    def _attempted_current_count(self) -> int:
        return sum(
            1 for item in self._history
            if item.get("event") == "execution_attempted"
            and item.get("plan_index") == self._current_index
        )

    def _guard_blocks_advance(self) -> bool:
        return (
            self._patch["operation"] == "SET_GUARD"
            and self._attempted_current
            and self._attempted_current_count() < self._patch["min_attempts"]
        )

    def _backtrack_allowed(self) -> bool:
        return (
            self._patch["operation"] == "RETRY_OR_BACKTRACK"
            and self._current_index > 0
            and self._attempted_current
            and self._stagnation_chunks >= self._patch["stagnation_chunks"]
            and self._backtrack_count < self._patch["max_backtracks"]
        )

    def eligible(
        self,
        transition_evidence: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return legal current/next/recovery steps after public evidence."""

        self._consume_transition(transition_evidence)
        if self._current_index >= len(self._plan):
            return []
        indices = [self._current_index]
        if self._backtrack_allowed():
            indices.append(self._current_index - 1)
        if (
            self._attempted_current
            and not self._guard_blocks_advance()
            and self._current_index + 1 < len(self._plan)
        ):
            indices.append(self._current_index + 1)
        return [self._copy_step(index) for index in indices]

    def mark_executed(
        self,
        plan_index: int,
        transition_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an attempt, optionally consuming its public evidence."""

        if plan_index != self._current_index:
            raise ValueError("only the current plan step may be marked attempted")
        self._consume_transition(transition_evidence)
        self._attempted_current = True
        self._history.append(
            {
                "event": "execution_attempted",
                "plan_index": plan_index,
                "predicted_advance": False,
                "verified_complete": False,
                "transition_evidence": deepcopy(dict(transition_evidence))
                if transition_evidence is not None else None,
                "stagnation_chunks": self._stagnation_chunks,
                "patch_id": self._patch["patch_id"],
            }
        )

    def choose(self, plan_index: int) -> dict[str, Any]:
        """Choose the current step, successor, or one permitted predecessor."""

        current = self._current_index
        allowed = {current}
        if (
            self._attempted_current
            and not self._guard_blocks_advance()
            and current + 1 < len(self._plan)
        ):
            allowed.add(current + 1)
        if self._backtrack_allowed():
            allowed.add(current - 1)
        if plan_index not in allowed:
            raise ValueError("plan step is not currently eligible")

        if plan_index == current - 1:
            self._current_index = plan_index
            self._attempted_current = False
            self._backtrack_count += 1
            self._stagnation_chunks = 0
            self._history.append(
                {
                    "event": "choose",
                    "plan_index": plan_index,
                    "predicted_advance": False,
                    "verified_complete": False,
                    "backtracked": True,
                    "backtrack_count": self._backtrack_count,
                    "patch_id": self._patch["patch_id"],
                }
            )
            return self._copy_step(plan_index)

        advanced = plan_index == current + 1
        if advanced:
            self._current_index = plan_index
            self._attempted_current = False
            self._stagnation_chunks = 0
        self._history.append(
            {
                "event": "choose",
                "plan_index": plan_index,
                "predicted_advance": advanced,
                "verified_complete": False,
                "backtracked": False,
                "patch_id": self._patch["patch_id"],
            }
        )
        return self._copy_step(plan_index)

    def with_patch(self, patch: Mapping[str, Any] | None) -> "PlanState":
        """Clone this state with a validated patch, preserving history."""

        clone = PlanState(self._plan, patch=patch)
        clone._current_index = self._current_index
        clone._attempted_current = self._attempted_current
        clone._history = deepcopy(self._history)
        clone._latest_transition = deepcopy(self._latest_transition)
        clone._stagnation_chunks = self._stagnation_chunks
        clone._backtrack_count = self._backtrack_count
        return clone

    def apply_harness_patch(
        self,
        patch: Mapping[str, Any] | None,
        observation: Mapping[str, Any] | None = None,
        public_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply the validated patch in-place for the current episode.

        ``observation`` is accepted as an audit-only summary by the model
        wrapper.  It is intentionally not interpreted as semantic truth;
        transition evidence must arrive through :meth:`mark_executed` or
        :meth:`eligible`.
        """

        self._patch = normalize_patch(patch)
        self._history.append({
            "event": "harness_patch_applied",
            "patch": deepcopy(self._patch),
            "observation_audit_available": observation is not None or public_observation is not None,
            "environment_step_budget_delta": 0,
        })
        return {"applied": True, "patch": deepcopy(self._patch), "observation_used": False}

    def apply_patch(
        self,
        patch: Mapping[str, Any] | None,
        observation: Mapping[str, Any] | None = None,
        public_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Alias for callers using the shorter editor API name."""

        return self.apply_harness_patch(patch, observation, public_observation)

    def export(self) -> dict[str, Any]:
        """Return a lightweight, JSON-serializable snapshot."""

        return {
            "schema_version": "0.2",
            "plan": deepcopy(self._plan),
            "current_index": self._current_index,
            "current_plan_index": self._plan[self._current_index]["plan_index"],
            "attempted_current": self._attempted_current,
            "patch": deepcopy(self._patch),
            "latest_transition_evidence": deepcopy(self._latest_transition),
            "stagnation_chunks": self._stagnation_chunks,
            "backtrack_count": self._backtrack_count,
            "selection_history": deepcopy(self._history),
        }


__all__ = ["PlanState", "normalize_patch"]
