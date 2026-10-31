"""Persistent, leakage-safe experience memory for the DuoFlow skill pilot.

The module deliberately stores observations as references or short summaries.  It
does not invent a failure explanation and it never makes a pending correction
visible to retrieval.  A rollout driver can append one record per skill decision
and a terminal record for the actual task outcome.

The on-disk layout is JSONL based::

    events.jsonl          train/dev experience
    eval_events.jsonl     evaluation audit log (never retrieved/exported)
    corrections.jsonl     pending and resolved correction proposals
    snapshots/*.jsonl     immutable, versioned train/dev snapshots

``MemoryStore`` accepts either a directory or a path ending in ``.jsonl``.  The
directory form is recommended for a run directory.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None


SPLITS = {"train", "dev", "eval"}
FEEDBACK = {"success", "failure", "unknown"}
FAILURE_SOURCES = {"unknown", "planning", "execution", "infrastructure"}
EVENT_SOURCES = {"rollout", "curated", "self_evolved", "external"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        value = " ".join(str(v) for v in value.values())
    if value is None:
        return set()
    return set(re.findall(r"[\w]+", str(value).lower(), flags=re.UNICODE))


def _nonempty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _json_copy(value: Any) -> Any:
    """Return a JSON-safe copy and reject NaN/Infinity at the boundary."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class MemoryStore:
    """Append-only experience memory with isolated evaluation logging.

    Parameters
    ----------
    path:
        A run directory, or an explicit ``events.jsonl`` path.
    allowed_splits:
        Splits accepted for normal events.  ``eval`` is always written to its
        separate audit file when supplied to :meth:`append_event`, even if it is
        omitted from this tuple, but is never returned by retrieval/export APIs.
    memory_version:
        Version stamped on every appended event and snapshot header.  Start a
        new store/version when the memory rule or schema changes.
    """

    def __init__(
        self,
        path: str | Path,
        allowed_splits: Sequence[str] = ("train", "dev"),
        memory_version: str = "memory-v1",
    ) -> None:
        allowed = set(allowed_splits)
        if not allowed or not allowed.issubset(SPLITS):
            raise ValueError(f"allowed_splits must be a non-empty subset of {sorted(SPLITS)}")
        if not _nonempty(memory_version):
            raise ValueError("memory_version must be non-empty")

        requested = Path(path)
        if requested.suffix.lower() == ".jsonl":
            self.root = requested.parent
            self.events_path = requested
            stem = requested.stem
            self.eval_events_path = requested.with_name(f"{stem}.eval.jsonl")
            self.corrections_path = requested.with_name(f"{stem}.corrections.jsonl")
        else:
            self.root = requested
            self.events_path = self.root / "events.jsonl"
            self.eval_events_path = self.root / "eval_events.jsonl"
            self.corrections_path = self.root / "corrections.jsonl"
        self.snapshots_dir = self.root / "snapshots"
        self.lock_path = self.root / ".memory.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_splits = frozenset(allowed)
        self.memory_version = str(memory_version)
        self._lock = threading.RLock()

    # ---------- persistence and validation ----------

    @staticmethod
    def _read_jsonl_unlocked(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
                records.append(item)
        return records

    @contextmanager
    def _file_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Coordinate JSONL readers/writers across rollout processes."""

        if fcntl is None:  # pragma: no cover - retained for non-POSIX imports
            yield
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        # Read the complete file while holding the shared lock, then yield the
        # in-memory records after unlocking.  A writer can therefore never be
        # observed between two JSONL lines or while a line is being appended.
        with self._file_lock(exclusive=False):
            records = self._read_jsonl_unlocked(path)
        return iter(records)

    def _normal_records(self) -> list[dict[str, Any]]:
        return list(self._read_jsonl(self.events_path))

    def _eval_records(self) -> list[dict[str, Any]]:
        return list(self._read_jsonl(self.eval_events_path))

    def _all_records(self) -> list[dict[str, Any]]:
        with self._file_lock(exclusive=False):
            return self._all_records_unlocked()

    def _all_records_unlocked(self) -> list[dict[str, Any]]:
        return (
            self._read_jsonl_unlocked(self.events_path)
            + self._read_jsonl_unlocked(self.eval_events_path)
            + self._read_jsonl_unlocked(self.corrections_path)
        )

    def _next_sequence(self) -> int:
        maximum = 0
        for record in self._all_records_unlocked():
            try:
                maximum = max(maximum, int(record.get("_store_seq", 0)))
            except (TypeError, ValueError):
                continue
        return maximum + 1

    def _append_record_unlocked(self, path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
        item = _json_copy(dict(record))
        item.setdefault("_store_seq", self._next_sequence())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
        return item

    def _append_record(self, path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._file_lock(exclusive=True):
            return self._append_record_unlocked(path, record)

    @staticmethod
    def _normalise_observation(observation: Any) -> dict[str, Any]:
        if isinstance(observation, str):
            if not observation.strip():
                raise ValueError("observation summary cannot be empty")
            return {"summary": observation}
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a reference/summary object or string")
        normalised = _json_copy(dict(observation))
        keys = {"ref", "reference", "summary", "sha256", "uri"}
        if not any(_nonempty(normalised.get(key)) for key in keys):
            raise ValueError("observation needs a non-empty ref or summary")
        return normalised

    def _validate_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        item = _json_copy(dict(event))
        required = (
            "episode_id",
            "task",
            "condition",
            "seed",
            "split",
            "model_version",
            "instruction",
            "selected_skill_id",
            "observation",
        )
        missing = [name for name in required if name not in item]
        if missing:
            raise ValueError(f"event missing required fields: {', '.join(missing)}")
        for name in ("episode_id", "task", "instruction", "selected_skill_id", "model_version"):
            if not _nonempty(item[name]):
                raise ValueError(f"event field {name!r} must be non-empty")
        split = item["split"]
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if split != "eval" and split not in self.allowed_splits:
            raise ValueError(f"split {split!r} is not enabled for this store")
        item["observation"] = self._normalise_observation(item["observation"])

        feedback = item.get("execution_feedback", "unknown")
        if feedback not in FEEDBACK:
            raise ValueError(f"execution_feedback must be one of {sorted(FEEDBACK)}")
        item["execution_feedback"] = feedback

        failure_source = item.get("failure_source", "unknown")
        if failure_source not in FAILURE_SOURCES:
            raise ValueError(f"failure_source must be one of {sorted(FAILURE_SOURCES)}")
        # A non-failure event cannot silently acquire a planning/execution label.
        if feedback != "failure" and failure_source != "unknown":
            raise ValueError("failure_source must remain unknown unless execution_feedback=failure")
        item["failure_source"] = failure_source

        record_type = item.get("record_type", "episode_step")
        if not isinstance(record_type, str) or not record_type.strip():
            raise ValueError("record_type must be a non-empty string")
        item["record_type"] = record_type
        source = item.get("source", "rollout")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        item["source"] = source
        if item["record_type"] == "seed_recipe" and source != "curated":
            raise ValueError("seed_recipe records must explicitly use source='curated'")

        terminal = item["terminal"] if "terminal" in item else record_type == "episode_outcome"
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be boolean")
        if record_type == "episode_outcome" and not terminal:
            raise ValueError("episode_outcome records must set terminal=true")
        item["terminal"] = terminal
        step_index = item.get("step_index", 0)
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        item["step_index"] = step_index

        if "step_feedback" in item:
            step_feedback = item["step_feedback"]
            if not isinstance(step_feedback, Mapping):
                raise ValueError("step_feedback must be an object")
            label = step_feedback.get("label", "unknown")
            if label not in FEEDBACK:
                raise ValueError("step_feedback.label must be success/failure/unknown")
            item["step_feedback"] = _json_copy(dict(step_feedback))
        if "weak_label" in item and not isinstance(item["weak_label"], bool):
            raise ValueError("weak_label must be boolean when supplied")
        if "feedback_text" in item and not isinstance(item["feedback_text"], str):
            raise ValueError("feedback_text must be a string when supplied")

        supplied_version = item.get("memory_version")
        if supplied_version is not None and supplied_version != self.memory_version:
            raise ValueError(
                f"event memory_version={supplied_version!r} does not match store {self.memory_version!r}"
            )
        item["memory_version"] = self.memory_version
        item.setdefault("event_id", uuid.uuid4().hex)
        item.setdefault("timestamp_utc", _utc_now())
        if not _nonempty(item["event_id"]):
            raise ValueError("event_id must be non-empty")
        if not _nonempty(item["timestamp_utc"]):
            raise ValueError("timestamp_utc must be non-empty")
        return item

    def append_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one rollout/curated event and return its stamped record.

        ``failure_source`` defaults to ``unknown``.  The method intentionally
        does not infer planning failure from a failed episode.  Evaluation events
        are routed to ``eval_events.jsonl`` and are invisible to retrieval.
        """

        with self._lock:
            item = self._validate_event(event)
            with self._file_lock(exclusive=True):
                if item["event_id"] in self._event_index_unlocked(include_eval=True):
                    raise ValueError(f"duplicate event_id {item['event_id']!r}")
                destination = self.eval_events_path if item["split"] == "eval" else self.events_path
                return self._append_record_unlocked(destination, item)

    def append_seed_recipe(self, recipe: Mapping[str, Any], **event_fields: Any) -> dict[str, Any]:
        """Append a human-curated seed recipe with an explicit provenance tag."""

        fields = dict(event_fields)
        fields["record_type"] = "seed_recipe"
        fields["source"] = "curated"
        fields["seed_recipe"] = _json_copy(dict(recipe))
        return self.append_event(fields)

    def iter_events(
        self,
        splits: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield only train/dev events; requesting eval raises to prevent leakage."""

        selected = set(self.allowed_splits if splits is None else splits)
        if "eval" in selected:
            raise ValueError("evaluation events are isolated and cannot be iterated as training data")
        if not selected.issubset(self.allowed_splits):
            raise ValueError(f"requested splits {sorted(selected)} exceed allowed splits")
        for event in self._normal_records():
            if event.get("split") in selected:
                yield event

    def _event_index(self, include_eval: bool = False) -> dict[str, dict[str, Any]]:
        with self._file_lock(exclusive=False):
            return self._event_index_unlocked(include_eval=include_eval)

    def _event_index_unlocked(self, include_eval: bool = False) -> dict[str, dict[str, Any]]:
        records = self._read_jsonl_unlocked(self.events_path)
        if include_eval:
            records += self._read_jsonl_unlocked(self.eval_events_path)
        return {str(record["event_id"]): record for record in records if record.get("event_id")}

    # ---------- retrieval ----------

    @staticmethod
    def _event_projection(event: Mapping[str, Any]) -> dict[str, Any]:
        observation = event.get("observation", {})
        return {
            "task": event.get("task"),
            "condition": event.get("condition"),
            "selected_skill_id": event.get("selected_skill_id"),
            "instruction": event.get("instruction"),
            "observation": observation,
        }

    @staticmethod
    def _query_projection(query: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(query)
        if "skill_id" in result and "selected_skill_id" not in result:
            result["selected_skill_id"] = result.pop("skill_id")
        return result

    @classmethod
    def _correction_context_matches(cls, query: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
        query = cls._query_projection(query)
        projection = cls._event_projection(event)
        return all(
            query.get(field) == projection.get(field)
            for field in ("task", "condition", "instruction", "selected_skill_id")
        )

    @classmethod
    def _similarity(cls, query: Mapping[str, Any], record: Mapping[str, Any]) -> float:
        query = cls._query_projection(query)
        projection = cls._event_projection(record)
        score = 0.0
        if query.get("task") is not None and query.get("task") == projection.get("task"):
            score += 5.0
        if query.get("condition") is not None and query.get("condition") == projection.get("condition"):
            score += 3.0
        if query.get("selected_skill_id") is not None and query.get("selected_skill_id") == projection.get(
            "selected_skill_id"
        ):
            score += 4.0
        if query.get("instruction") is not None:
            q_tokens = _tokens(query["instruction"])
            r_tokens = _tokens(projection.get("instruction"))
            if q_tokens and r_tokens:
                overlap = len(q_tokens & r_tokens) / max(len(q_tokens), len(r_tokens))
                score += 4.0 * overlap
        q_obs = query.get("observation")
        r_obs = projection.get("observation")
        if isinstance(q_obs, Mapping) and isinstance(r_obs, Mapping):
            for key in ("ref", "reference", "uri", "sha256", "summary"):
                if _nonempty(q_obs.get(key)) and q_obs.get(key) == r_obs.get(key):
                    score += 2.0
                    break
        return score

    @staticmethod
    def _latest_corrections_from(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            correction_id = record.get("correction_id")
            if correction_id:
                latest[str(correction_id)] = dict(record)
        return latest

    def _latest_corrections(self) -> dict[str, dict[str, Any]]:
        with self._file_lock(exclusive=False):
            return self._latest_corrections_from(self._read_jsonl_unlocked(self.corrections_path))

    def _latest_corrections_unlocked(self) -> dict[str, dict[str, Any]]:
        return self._latest_corrections_from(self._read_jsonl_unlocked(self.corrections_path))

    def retrieve(
        self,
        query: Mapping[str, Any],
        limit: int = 20,
        split: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve similar non-eval episodes, counts, and validated experience.

        ``pending`` and ``rejected`` corrections are deliberately absent from
        ``validated_experiences``.  The returned counts are episode counts, so a
        multi-step episode is not over-counted.
        """

        if not isinstance(query, Mapping) or not query:
            raise ValueError("retrieve query must contain at least one field")
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = self._query_projection(query)
        requested_split = split or query.get("split")
        if requested_split == "eval":
            raise ValueError("evaluation memory cannot be retrieved")
        if requested_split is not None and requested_split not in self.allowed_splits:
            raise ValueError(f"split {requested_split!r} is not enabled")

        # Capture events and correction state under one shared lock so a
        # concurrent rollout cannot expose a half-written JSONL line or mix
        # correction revisions from different file reads.
        with self._file_lock(exclusive=False):
            normal_records = self._read_jsonl_unlocked(self.events_path)
            latest_corrections = self._latest_corrections_from(
                self._read_jsonl_unlocked(self.corrections_path)
            )
        events = [
            event
            for event in normal_records
            if event.get("split") in (set([requested_split]) if requested_split else self.allowed_splits)
        ]
        # Keep a complete per-episode view so a query matching one step still
        # returns the selected-skill sequence and feedback from all prior steps.
        episode_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for event in events:
            episode_events[str(event["episode_id"])].append(event)
            score = self._similarity(query, event)
            if score > 0:
                grouped[str(event["episode_id"])].append((score, event))

        summaries: list[dict[str, Any]] = []
        for episode_id, scored_events in grouped.items():
            scored_events.sort(key=lambda item: (item[0], int(item[1].get("_store_seq", 0))), reverse=True)
            all_events = sorted(
                episode_events[episode_id], key=lambda event: int(event.get("_store_seq", 0))
            )
            terminal = [
                event
                for event in all_events
                if event.get("terminal") and event.get("execution_feedback") in {"success", "failure"}
            ]
            outcome_event = terminal[-1] if terminal else all_events[-1]
            skill_history = [
                {
                    "step_index": event.get("step_index", 0),
                    "event_id": event.get("event_id"),
                    "selected_skill_id": event.get("selected_skill_id"),
                    "execution_prompt": event.get("execution_prompt"),
                    "plan_index": event.get("plan_index"),
                    "execution_feedback": event.get("execution_feedback", "unknown"),
                    "step_feedback": event.get("step_feedback"),
                    "feedback_text": event.get(
                        "feedback_text",
                        event.get("step_feedback", {}).get(
                            "text", event.get("step_feedback", {}).get("label", "")
                        )
                        if isinstance(event.get("step_feedback"), Mapping)
                        else "",
                    ),
                    "failure_source": event.get("failure_source", "unknown"),
                }
                for event in all_events
            ]
            summaries.append(
                {
                    "episode_id": episode_id,
                    "task": outcome_event.get("task"),
                    "condition": outcome_event.get("condition"),
                    "selected_skill_id": outcome_event.get("selected_skill_id"),
                    "execution_feedback": outcome_event.get("execution_feedback", "unknown")
                    if terminal
                    else "unknown",
                    "failure_source": outcome_event.get("failure_source", "unknown")
                    if outcome_event.get("execution_feedback") == "failure"
                    else "unknown",
                    "event_ids": [event["event_id"] for event in all_events],
                    "best_score": scored_events[0][0],
                    "evidence_event_count": len(all_events),
                    "selected_skill_id_sequence": [
                        item["selected_skill_id"] for item in skill_history
                    ],
                    "skill_history": skill_history,
                    "feedback_history": [
                        item["feedback_text"] or item["execution_feedback"] for item in skill_history
                    ],
                }
            )
        summaries.sort(key=lambda item: (item["best_score"], item["episode_id"]), reverse=True)
        selected = summaries[:limit]

        stats = {"episodes": len(summaries), "success": 0, "failure": 0, "unknown": 0}
        for summary in summaries:
            stats[summary["execution_feedback"]] += 1

        validated: list[dict[str, Any]] = []
        for correction in latest_corrections.values():
            if correction.get("status") != "validated":
                continue
            if self._similarity(query, correction.get("query", {})) <= 0:
                continue
            validated.append(
                {
                    "correction_id": correction["correction_id"],
                    "correction": correction.get("correction"),
                    "query": correction.get("query", {}),
                    "evidence_event_ids": correction.get("evidence_event_ids", []),
                    "status": "validated",
                }
            )
        return {
            "query": _json_copy(dict(query)),
            "memory_version": self.memory_version,
            "statistics": stats,
            "episodes": selected,
            "validated_experiences": validated,
        }

    # ---------- correction lifecycle ----------

    def propose_correction(
        self,
        source_failure_event_id: str,
        correction: str | Mapping[str, Any],
        query: Mapping[str, Any] | None = None,
        evidence_event_ids: Sequence[str] = (),
        source: str = "system_proposal",
    ) -> dict[str, Any]:
        """Create a pending correction; no pending correction is retrievable."""

        with self._lock:
            with self._file_lock(exclusive=True):
                normal = self._event_index_unlocked(include_eval=False)
                event = normal.get(str(source_failure_event_id))
                if event is None:
                    raise ValueError("source failure event must be a non-eval event")
                if event.get("execution_feedback") != "failure" or event.get("terminal") is not True:
                    raise ValueError("source_failure_event_id must reference a terminal failure")
                if not _nonempty(source):
                    raise ValueError("correction source must be non-empty")
                if isinstance(correction, str):
                    if not correction.strip():
                        raise ValueError("correction cannot be empty")
                    stored_correction: Any = correction
                elif isinstance(correction, Mapping):
                    stored_correction = _json_copy(dict(correction))
                    if not stored_correction:
                        raise ValueError("correction cannot be empty")
                else:
                    raise ValueError("correction must be a string or object")
                source_query = self._event_projection(event)
                stored_query = self._query_projection(query or source_query)
                required_context = ("task", "condition", "instruction", "selected_skill_id")
                if any(field not in stored_query for field in required_context):
                    raise ValueError(
                        "correction query must include task, condition, instruction, and selected_skill_id"
                    )
                if not self._correction_context_matches(stored_query, event):
                    raise ValueError("correction query must match the source task/condition/instruction/skill")
                evidence = list(dict.fromkeys(str(item) for item in evidence_event_ids))
                for evidence_id in evidence:
                    if evidence_id not in normal:
                        raise ValueError("correction evidence cannot reference eval or unknown events")
                item = {
                    "record_type": "correction",
                    "correction_id": uuid.uuid4().hex,
                    "status": "pending",
                    "source_failure_event_id": str(source_failure_event_id),
                    "query": _json_copy(stored_query),
                    "correction": stored_correction,
                    "evidence_event_ids": evidence,
                    "source": str(source),
                    "memory_version": self.memory_version,
                    "created_at_utc": _utc_now(),
                }
                return self._append_record_unlocked(self.corrections_path, item)

    def resolve_correction(
        self,
        correction_id: str,
        resolution: str,
        evidence_event_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Resolve a pending correction only with independent later evidence."""

        if resolution not in {"validated", "rejected"}:
            raise ValueError("resolution must be validated or rejected")
        evidence_ids = list(dict.fromkeys(str(item) for item in evidence_event_ids))
        if not evidence_ids:
            raise ValueError("resolution requires independent execution evidence")
        with self._lock:
            with self._file_lock(exclusive=True):
                current = self._latest_corrections_unlocked().get(str(correction_id))
                if current is None:
                    raise KeyError(f"unknown correction_id {correction_id!r}")
                if current.get("status") != "pending":
                    raise ValueError("only pending corrections can be resolved")
                events = self._event_index_unlocked(include_eval=False)
                source_id = str(current["source_failure_event_id"])
                source = events.get(source_id)
                if source is None:
                    raise ValueError("correction source failure is unavailable")
                if source.get("execution_feedback") != "failure" or source.get("terminal") is not True:
                    raise ValueError("correction source must remain a terminal failure")
                current_query = current.get("query", {})
                if not isinstance(current_query, Mapping):
                    raise ValueError("correction query is unavailable")
                query = self._query_projection(current_query)
                required_context = ("task", "condition", "instruction", "selected_skill_id")
                if any(field not in query for field in required_context):
                    raise ValueError(
                        "correction query must include task, condition, instruction, and selected_skill_id"
                    )
                if not self._correction_context_matches(query, source):
                    raise ValueError("correction query must match the source task/condition/instruction/skill")
                source_seq = int(source.get("_store_seq", 0))
                evidence = []
                for event_id in evidence_ids:
                    event = events.get(event_id)
                    if event is None:
                        raise ValueError("resolution evidence cannot reference eval or unknown events")
                    if int(event.get("_store_seq", 0)) <= source_seq:
                        raise ValueError("resolution evidence must be appended after the failure")
                    if event.get("episode_id") == source.get("episode_id"):
                        raise ValueError("resolution evidence must come from an independent episode")
                    if not event.get("terminal") or event.get("execution_feedback") not in {"success", "failure"}:
                        raise ValueError("resolution evidence must be a terminal success/failure")
                    if not self._correction_context_matches(query, event):
                        raise ValueError(
                            "resolution evidence must match correction task/condition/instruction/skill"
                        )
                    evidence.append(event)
                labels = {event["execution_feedback"] for event in evidence}
                if resolution == "validated" and "success" not in labels:
                    raise ValueError("validated correction requires a later successful execution")
                if resolution == "rejected" and "failure" not in labels:
                    raise ValueError("rejected correction requires a later failed execution")
                item = {
                    "record_type": "correction",
                    "correction_id": str(correction_id),
                    "status": resolution,
                    "source_failure_event_id": source_id,
                    "query": current.get("query", {}),
                    "correction": current.get("correction"),
                    "evidence_event_ids": evidence_ids,
                    "source": current.get("source", "system_proposal"),
                    "memory_version": self.memory_version,
                    "resolved_at_utc": _utc_now(),
                }
                return self._append_record_unlocked(self.corrections_path, item)

    # ---------- fixed snapshots ----------

    def freeze_snapshot(
        self,
        snapshot_version: str,
        splits: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Write an immutable JSONL snapshot for the selected allowed splits.

        Evaluation records are always excluded; the caller may restrict the
        snapshot further to any non-evaluation split enabled on this store.
        """

        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(snapshot_version)):
            raise ValueError("snapshot_version contains unsupported path characters")
        selected = set(self.allowed_splits if splits is None else splits)
        if "eval" in selected:
            raise ValueError("evaluation records cannot be frozen into a training snapshot")
        if not selected.issubset(self.allowed_splits):
            raise ValueError(f"snapshot splits {sorted(selected)} exceed allowed splits")
        with self._lock:
            with self._file_lock(exclusive=False):
                normal_records = self._read_jsonl_unlocked(self.events_path)
                latest_corrections = self._latest_corrections_from(
                    self._read_jsonl_unlocked(self.corrections_path)
                )
            events = [event for event in normal_records if event.get("split") in selected]
            event_ids = {event.get("event_id") for event in events}
            corrections = [
                correction
                for correction in latest_corrections.values()
                if correction.get("source_failure_event_id") in event_ids
            ]
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshots_dir / f"{snapshot_version}.jsonl"
            header = {
                "record_type": "snapshot_header",
                "snapshot_version": str(snapshot_version),
                "memory_version": self.memory_version,
                "splits": sorted(selected),
                "eval_included": False,
                "created_at_utc": _utc_now(),
                "event_count": len(events),
                "correction_count": len(corrections),
            }
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
                    for record in sorted(events, key=lambda item: int(item.get("_store_seq", 0))):
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    for record in sorted(corrections, key=lambda item: int(item.get("_store_seq", 0))):
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            except FileExistsError as exc:
                raise FileExistsError(f"snapshot version already exists: {snapshot_version}") from exc
            return {
                "snapshot_version": str(snapshot_version),
                "memory_version": self.memory_version,
                "path": str(path),
                "event_count": len(events),
                "correction_count": len(corrections),
                "eval_included": False,
            }


__all__ = ["MemoryStore", "SPLITS", "FEEDBACK", "FAILURE_SOURCES"]
