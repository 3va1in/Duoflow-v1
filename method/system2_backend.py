"""System2 skill selector built on the native GigaBrain/PaliGemma prefix.

This module deliberately implements a *new* System2 selector prototype.  It
does not claim to be the independent System2 model described in the paper.
The minimum model path embeds one candidate-conditioned prompt with the native
GigaBrain VLM and trains a small candidate/status scorer on top of the pooled
hidden state.  The scorer can also be trained from cached VLM features.

The native server integration is explicit.  ``from_native`` expects either a
``system2_feature_extractor`` callback, or a ``system2_preprocess`` callback
plus a native policy exposing ``embed_prefix`` and
``paligemma_with_expert.forward``.  The official inference server keeps its
preprocessing transforms in a closure, so callers must attach the callback (or
the equivalent transforms) to the server object; this module never silently
uses unprocessed images or a hand-written rule in their place.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


STATUS_TO_INDEX = {"continue": 0, "done": 1}
INDEX_TO_STATUS = {value: key for key, value in STATUS_TO_INDEX.items()}


class NativeInterfaceError(RuntimeError):
    """Raised when a native server has not exposed the required adapter."""


@dataclass(frozen=True)
class SkillCandidate:
    """A legal atomic skill offered to the selector."""

    id: str
    description: str
    args: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "SkillCandidate":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(id=value, description=value)
        if isinstance(value, Mapping):
            skill_id = value.get("id", value.get("skill_id"))
            if not isinstance(skill_id, str) or not skill_id:
                raise ValueError(f"skill candidate needs a non-empty id: {value!r}")
            description = value.get(
                "execution_prompt",
                value.get("description", value.get("text", skill_id)),
            )
            if not isinstance(description, str):
                description = str(description)
            args = value.get("args", value.get("bindings", {}))
            if not isinstance(args, Mapping):
                raise ValueError(f"skill candidate args must be a mapping: {value!r}")
            return cls(skill_id, description, dict(args))
        raise TypeError(f"unsupported skill candidate type: {type(value).__name__}")


@dataclass(frozen=True)
class MemoryRecord:
    """A retrieved public-memory item used as prompt context."""

    text: str
    skill_id: str | None = None
    outcome: str | None = None
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "MemoryRecord":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, Mapping):
            text = value.get("text", value.get("summary", value.get("failure_reason", "")))
            if not isinstance(text, str):
                text = str(text)
            metadata = value.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {"value": metadata}
            return cls(
                text=text,
                skill_id=value.get("skill_id"),
                outcome=value.get("outcome"),
                failure_reason=value.get("failure_reason"),
                metadata=dict(metadata),
            )
        raise TypeError(f"unsupported memory type: {type(value).__name__}")


class CandidateScorer(nn.Module):
    """Small trainable head over candidate-conditioned VLM features."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        hidden_dim = int(hidden_dim or max(32, min(512, feature_dim // 2)))
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.skill_head = nn.Linear(hidden_dim, 1)
        self.status_head = nn.Linear(hidden_dim, len(STATUS_TO_INDEX))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [N,{self.feature_dim}], got {tuple(features.shape)}"
            )
        hidden = self.projection(features)
        return self.skill_head(hidden).squeeze(-1), self.status_head(hidden)


class LoRALinear(nn.Module):
    """A frozen linear layer with an additive low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(
    module: nn.Module,
    target_suffixes: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Inject LoRA only into matching ``nn.Linear`` children of ``module``.

    ``module`` must be an independently loaded planner copy.  The caller owns
    that isolation boundary; this function never looks up or mutates an S1
    object by itself.
    """

    suffixes = tuple(str(item) for item in target_suffixes)
    if not suffixes:
        raise ValueError("target_suffixes cannot be empty")
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(path.endswith(suffix) for suffix in suffixes):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(path)
            else:
                visit(child, path)

    visit(module)
    if not replaced:
        raise ValueError(f"no nn.Linear matched LoRA target suffixes {list(suffixes)!r}")
    return replaced


def _as_tensor(value: Any, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _normalise_candidates(values: Iterable[Any]) -> list[SkillCandidate]:
    candidates = [SkillCandidate.from_value(value) for value in values]
    if not candidates:
        raise ValueError("skill_candidates cannot be empty")
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"skill_candidates contains duplicate ids: {ids!r}")
    return candidates


def _normalise_memory(values: Iterable[Any] | None) -> list[MemoryRecord]:
    return [MemoryRecord.from_value(value) for value in (values or [])]


def _format_context(
    task: str,
    candidate: SkillCandidate,
    memory: Sequence[MemoryRecord],
    history: Sequence[Any] | None,
) -> str:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    lines = [f"Task: {task.strip()}", f"Candidate skill id: {candidate.id}", f"Candidate skill: {candidate.description}"]
    if candidate.args:
        lines.append("Candidate arguments: " + json.dumps(dict(candidate.args), sort_keys=True, ensure_ascii=False))
    if memory:
        lines.append("Retrieved execution memory:")
        for item in memory:
            details = [item.text]
            if item.skill_id:
                details.append(f"skill_id={item.skill_id}")
            if item.outcome:
                details.append(f"outcome={item.outcome}")
            if item.failure_reason:
                details.append(f"failure={item.failure_reason}")
            if item.metadata:
                details.append("metadata=" + json.dumps(dict(item.metadata), ensure_ascii=False, sort_keys=True))
            lines.append("- " + " | ".join(details))
    if history:
        lines.append("Recent selector history:")
        for item in history:
            lines.append("- " + (item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)))
    return "\n".join(lines)


class System2Backend:
    """Candidate selector using native VLM features and a trainable scorer.

    ``rank`` returns a JSON-friendly mapping with the stable keys
    ``skill_id``, ``status``, ``confidence`` and ``scores``.  The additional
    ``update_scope`` field is intentionally explicit so cached-feature scorer
    updates cannot be mistaken for VLM LoRA updates.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        scorer: CandidateScorer | None = None,
        native_server: Any | None = None,
        policy: Any | None = None,
        preprocess: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        feature_extractor: Callable[[Mapping[str, Any]], Tensor] | None = None,
        update_scope: str = "scorer",
        device: torch.device | str | None = None,
    ):
        self.feature_dim = int(feature_dim)
        self.native_server = native_server
        self.policy = policy
        self.preprocess = preprocess
        self.feature_extractor = feature_extractor
        self.device = torch.device(device or self._infer_device(policy) or "cpu")
        self.scorer = scorer or CandidateScorer(self.feature_dim)
        self.scorer.to(self.device)
        self.update_scope = update_scope
        self.planner_policy: Any | None = getattr(native_server, "system2_policy", None)
        self.lora_modules: list[str] = []
        self.last_observation_mode: str | None = None

    @staticmethod
    def _infer_device(module: Any | None) -> torch.device | None:
        if module is None:
            return None
        declared = getattr(module, "device", None)
        if declared is not None:
            try:
                return torch.device(declared)
            except (TypeError, RuntimeError):
                pass
        if isinstance(module, nn.Module):
            try:
                return next(module.parameters()).device
            except StopIteration:
                pass
        return None

    @classmethod
    def from_native(cls, native_server: Any, checkpoint: str | Path | None = None) -> "System2Backend":
        """Construct a selector from a native server without changing S1.

        The server may expose ``policy`` or itself be the policy.  A native
        feature callback is preferred for integration.  Otherwise
        ``system2_preprocess`` must return the official prepared tensors, and
        this backend calls the policy's existing ``embed_prefix`` and dual
        stream forward directly.
        """

        if native_server is None:
            raise NativeInterfaceError("native_server is required")
        policy = getattr(native_server, "policy", native_server)
        if not hasattr(policy, "embed_prefix") or not hasattr(policy, "paligemma_with_expert"):
            raise NativeInterfaceError(
                "native_server must expose a GigaBrain policy with embed_prefix and paligemma_with_expert"
            )
        feature_extractor = getattr(native_server, "system2_feature_extractor", None)
        preprocess = getattr(native_server, "system2_preprocess", None)
        if feature_extractor is not None and not callable(feature_extractor):
            raise NativeInterfaceError("native_server.system2_feature_extractor must be callable")
        if preprocess is not None and not callable(preprocess):
            raise NativeInterfaceError("native_server.system2_preprocess must be callable")
        if feature_extractor is None and preprocess is None:
            # The official server keeps these objects as get_policy() closure
            # locals.  A direct, CPU-safe native loader can instead expose the
            # same objects on the server; build the adapter without touching
            # the S1 policy when that contract is present.
            preprocess = _build_direct_official_preprocess(native_server, policy)

        feature_dim = getattr(native_server, "system2_feature_dim", None)
        if feature_dim is None:
            feature_dim = getattr(policy, "vlm_hidden_size", None)
        if feature_dim is None:
            feature_dim = getattr(getattr(policy, "config", None), "vlm_hidden_size", None)
        if feature_dim is None:
            paligemma = getattr(policy, "paligemma_with_expert", None)
            feature_dim = getattr(paligemma, "paligemma_hidden_size", None)
        if feature_dim is None:
            raise NativeInterfaceError(
                "native_server must provide system2_feature_dim or a policy VLM hidden size"
            )
        backend = cls(
            int(feature_dim),
            native_server=native_server,
            policy=policy,
            preprocess=preprocess,
            feature_extractor=feature_extractor,
            update_scope="scorer",
            device=getattr(native_server, "device", None) or cls._infer_device(policy),
        )
        if checkpoint is not None:
            backend.load_checkpoint(checkpoint)
        return backend

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint must contain a mapping, got {type(payload).__name__}")
        expected = set(self.scorer.state_dict())

        def _normalise_state(raw: Any) -> Mapping[str, Any] | None:
            if not isinstance(raw, Mapping):
                return None
            # Training receipts have used both a top-level scorer_state and
            # teacher/model aliases.  Strip only wrappers that are common in
            # DataParallel or teacher checkpoints; do not silently reshape a
            # scorer with a different feature dimension.
            candidates: list[Mapping[str, Any]] = [raw]
            for nested_key in ("scorer", "state_dict", "model_state_dict", "weights"):
                nested = raw.get(nested_key)
                if isinstance(nested, Mapping):
                    candidates.append(nested)
            prefixes = ("module.scorer.", "scorer.", "module.")
            for candidate in candidates:
                for prefix in ("", *prefixes):
                    state = {
                        (key[len(prefix):] if prefix and str(key).startswith(prefix) else str(key)): value
                        for key, value in candidate.items()
                    }
                    if expected.issubset(state):
                        return {key: state[key] for key in expected}
            return None

        state = None
        state_keys = (
            "scorer_state",
            "teacher_state_dict",
            "teacher_state",
            "teacher_weights",
            "model_state_dict",
            "state_dict",
            "model",
        )
        for key in state_keys:
            state = _normalise_state(payload.get(key))
            if state is not None:
                break
        if state is None:
            state = _normalise_state(payload)
        if state is None:
            raise ValueError(
                "checkpoint does not contain a complete scorer state; "
                f"expected keys={sorted(expected)}, top-level keys={sorted(map(str, payload))}"
            )
        try:
            self.scorer.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"checkpoint scorer shape/config does not match feature_dim={self.feature_dim}") from exc
        saved_scope = payload.get("update_scope")
        if saved_scope == "scorer":
            self.update_scope = "scorer"

    def attach_planner_lora(
        self,
        target_suffixes: Sequence[str],
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> list[str]:
        """Inject LoRA into the explicitly supplied independent planner copy."""

        planner = self.planner_policy
        if planner is None:
            raise NativeInterfaceError(
                "planner LoRA requires native_server.system2_policy, an independently loaded planner copy"
            )
        if planner is self.policy:
            raise NativeInterfaceError("refusing to inject planner LoRA into the S1 policy object")
        for parameter in planner.parameters():
            parameter.requires_grad_(False)
        self.lora_modules = inject_lora(planner, target_suffixes, rank=rank, alpha=alpha, dropout=dropout)
        self.update_scope = "planner_vlm_lora"
        return list(self.lora_modules)

    def _request(
        self,
        images: Any,
        state: Any,
        task: str,
        candidate: SkillCandidate,
        memory: Sequence[MemoryRecord],
        history: Sequence[Any] | None,
        state_present: bool | None = None,
    ) -> dict[str, Any]:
        context = _format_context(task, candidate, memory, history)
        state_value = state
        images_empty = isinstance(images, Mapping) and not images
        if isinstance(images, (list, tuple)) and not images:
            images_empty = True
        resolved_state_present = not images_empty
        if isinstance(state, Mapping):
            state_value = state.get("values", state.get("observation.state", state.get("state", state)))
            if "present" in state or "state_present" in state:
                resolved_state_present = bool(state.get("present", state.get("state_present")))
        if state_present is not None:
            resolved_state_present = bool(state_present)
        return {
            "images": images,
            "observation.images": images,
            "state": state_value,
            "observation.state": state_value,
            "state_present": resolved_state_present,
            "task": context,
            "raw_task": task,
            "skill_candidate": asdict(candidate),
            "retrieved_memory": [asdict(item) for item in memory],
            "history": list(history or []),
        }

    @staticmethod
    def _call_adapter(adapter: Callable[..., Any], request: Mapping[str, Any]) -> Any:
        """Call a documented one-request adapter while tolerating keyword style."""

        try:
            signature = inspect.signature(adapter)
        except (TypeError, ValueError):
            return adapter(request)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return adapter(**dict(request))
        if len(signature.parameters) == 1:
            return adapter(request)
        kwargs = {name: request[name] for name in signature.parameters if name in request}
        if len(kwargs) == len(signature.parameters):
            return adapter(**kwargs)
        return adapter(request)

    def _coerce_feature_output(self, output: Any, *, expected_count: int) -> Tensor:
        if isinstance(output, Mapping):
            for key in ("features", "feature", "pooled_features", "hidden"):
                if key in output:
                    output = output[key]
                    break
        features = _as_tensor(output, device=self.device, dtype=torch.float32)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape != (expected_count, self.feature_dim):
            raise NativeInterfaceError(
                f"native feature adapter must return [{expected_count},{self.feature_dim}], got {tuple(features.shape)}"
            )
        return features

    def _encode_official_prefix(self, prepared: Mapping[str, Any], policy: Any) -> Tensor:
        required = ("images", "img_masks", "lang_tokens", "lang_masks")
        missing = [key for key in required if key not in prepared]
        if missing:
            raise NativeInterfaceError(
                "system2_preprocess output is missing prepared fields "
                + ", ".join(missing)
                + "; see the native_server contract"
            )
        images = [_as_tensor(item, device=self.device) for item in prepared["images"]]
        img_masks = [_as_tensor(item, device=self.device, dtype=torch.bool) for item in prepared["img_masks"]]
        lang_tokens = _as_tensor(prepared["lang_tokens"], device=self.device, dtype=torch.long)
        lang_masks = _as_tensor(prepared["lang_masks"], device=self.device, dtype=torch.bool)
        lang_att_masks = prepared.get("lang_att_masks")
        if lang_att_masks is not None:
            lang_att_masks = _as_tensor(lang_att_masks, device=self.device, dtype=torch.bool)
        proprioception = prepared.get("proprioception")
        if proprioception is not None:
            proprioception = _as_tensor(proprioception, device=self.device, dtype=torch.float32)
        agent_pos_mask = prepared.get("agent_pos_mask")
        if agent_pos_mask is not None:
            agent_pos_mask = _as_tensor(agent_pos_mask, device=self.device, dtype=torch.float32)
        proprioception_present = prepared.get("proprioception_present")
        if proprioception_present is not None:
            proprioception_present = _as_tensor(proprioception_present, device=self.device, dtype=torch.bool)

        has_valid_image = any(bool(mask.any().item()) for mask in img_masks)
        if has_valid_image:
            prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = policy.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                lang_att_masks=lang_att_masks,
                proprioception=proprioception,
                agent_pos_mask=agent_pos_mask,
                proprioception_present=proprioception_present,
            )
            self.last_observation_mode = "rgb_vlm"
        else:
            # Curated bootstrap rows intentionally have no RGB observation.  Do
            # not run the vision encoder on zero placeholders or retain its
            # masked 768-token prefix; use the native language embedding path.
            lang_emb = policy.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            lang_emb = policy._scatter_proprioception_anchor(
                lang_tokens,
                lang_emb,
                proprioception,
                agent_pos_mask,
                proprioception_present,
            )
            prefix_embs = lang_emb
            prefix_pad_masks = lang_masks
            prefix_att_masks = (
                torch.zeros_like(lang_masks, dtype=torch.bool)
                if lang_att_masks is None
                else lang_att_masks
            )
            self.last_observation_mode = "text_only_bootstrap"
        state_memory = prepared.get("state_memory")
        state_memory_masks = prepared.get("state_memory_masks")
        emb_ids = prepared.get("emb_ids")
        if state_memory is not None:
            state_memory = _as_tensor(state_memory, device=self.device, dtype=prefix_embs.dtype)
        if state_memory_masks is not None:
            state_memory_masks = _as_tensor(state_memory_masks, device=self.device, dtype=torch.bool)
        if emb_ids is not None:
            emb_ids = _as_tensor(emb_ids, device=self.device, dtype=torch.long)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = policy._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        # Importing this helper lazily keeps cached-feature training independent
        # of the official source package and its optional model dependencies.
        make_att_2d_masks = _load_make_att_2d_masks()
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = policy._compute_1d_position_ids(prefix_pad_masks)
        outputs, _ = policy.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, None],
        )
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if hidden is None or hidden.ndim != 3:
            raise NativeInterfaceError("native PaliGemma forward did not return a [B,L,H] hidden state")
        positions = torch.arange(prefix_pad_masks.shape[1], device=prefix_pad_masks.device)
        positions = positions.unsqueeze(0).expand_as(prefix_pad_masks)
        last_indices = positions.masked_fill(~prefix_pad_masks.to(dtype=torch.bool), -1).max(dim=1).values
        last_indices = last_indices.clamp_min(0)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_indices, last_indices]
        if pooled.shape[-1] != self.feature_dim:
            raise NativeInterfaceError(
                f"native pooled hidden size {pooled.shape[-1]} does not match scorer feature_dim {self.feature_dim}"
            )
        return pooled

    def encode_candidate_features(
        self,
        images: Any,
        state: Any,
        task: str,
        skill_candidates: Sequence[Any],
        retrieved_memory: Sequence[Any] | None,
        history: Sequence[Any] | None = None,
        *,
        require_grad: bool = False,
        state_present: bool | None = None,
    ) -> Tensor:
        candidates = _normalise_candidates(skill_candidates)
        memory = _normalise_memory(retrieved_memory)
        features: list[Tensor] = []
        adapter = self.feature_extractor
        for candidate in candidates:
            request = self._request(images, state, task, candidate, memory, history, state_present)
            grad_context = torch.enable_grad() if require_grad else torch.no_grad()
            with grad_context:
                if adapter is not None:
                    output = self._call_adapter(adapter, request)
                    feature = self._coerce_feature_output(output, expected_count=1)
                else:
                    if self.preprocess is None:
                        raise NativeInterfaceError(
                            "native_server must expose system2_feature_extractor or system2_preprocess; "
                            "official get_policy keeps preprocessing transforms in a closure"
                        )
                    prepared = self._call_adapter(self.preprocess, request)
                    if not isinstance(prepared, Mapping):
                        raise NativeInterfaceError("system2_preprocess must return a mapping of prepared tensors")
                    model_for_features = self.planner_policy if self.update_scope == "planner_vlm_lora" else self.policy
                    # _encode_official_prefix pools the single request batch and
                    # already returns [1,H].  Keep that batch dimension so the
                    # candidate list concatenates to [N,H], as rank_from_features
                    # requires.
                    feature = self._encode_official_prefix(prepared, model_for_features)
            features.append(feature if require_grad else feature.detach())
        return torch.cat(features, dim=0)

    def score_features(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.scorer(_as_tensor(features, device=self.device, dtype=torch.float32))

    def rank_from_features(self, features: Tensor, skill_candidates: Sequence[Any]) -> dict[str, Any]:
        candidates = _normalise_candidates(skill_candidates)
        features = _as_tensor(features, device=self.device, dtype=torch.float32)
        if features.shape != (len(candidates), self.feature_dim):
            raise ValueError(
                f"features must have shape [{len(candidates)},{self.feature_dim}], got {tuple(features.shape)}"
            )
        self.scorer.eval()
        with torch.no_grad():
            score_logits, status_logits = self.score_features(features)
            skill_probs = torch.softmax(score_logits, dim=0)
            selected = int(torch.argmax(score_logits).item())
            status_probs = torch.softmax(status_logits[selected], dim=-1)
            status_index = int(torch.argmax(status_probs).item())
        scores = [
            {"skill_id": candidates[index].id, "score": float(skill_probs[index]), "logit": float(score_logits[index])}
            for index in range(len(candidates))
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        confidence = float(skill_probs[selected] * status_probs[status_index])
        return {
            "skill_id": candidates[selected].id,
            "status": INDEX_TO_STATUS[status_index],
            "confidence": confidence,
            "scores": scores,
            "update_scope": self.update_scope,
            "mode": "cached_feature_scorer",
        }

    def rank(
        self,
        images: Any,
        state: Any,
        task: str,
        skill_candidates: Sequence[Any],
        retrieved_memory: Sequence[Any] | None,
        history: Sequence[Any] | None = None,
        *,
        state_present: bool | None = None,
    ) -> dict[str, Any]:
        """Select one legal skill from VLM-conditioned candidates."""

        features = self.encode_candidate_features(
            images,
            state,
            task,
            skill_candidates,
            retrieved_memory,
            history,
            require_grad=False,
            state_present=state_present,
        )
        result = self.rank_from_features(features, skill_candidates)
        self.last_candidate_features = features.detach().cpu()
        result["mode"] = "native_vlm_candidate_selector"
        if self.last_observation_mode is not None:
            result["observation_mode"] = self.last_observation_mode
        return result

    def trainable_parameters(self, update_scope: str | None = None) -> list[nn.Parameter]:
        scope = update_scope or self.update_scope
        if scope == "scorer":
            return [parameter for parameter in self.scorer.parameters() if parameter.requires_grad]
        if scope == "planner_vlm_lora":
            parameters = [parameter for parameter in self.scorer.parameters() if parameter.requires_grad]
            if self.planner_policy is None:
                raise NativeInterfaceError("planner_vlm_lora requested without system2_policy")
            parameters.extend(parameter for parameter in self.planner_policy.parameters() if parameter.requires_grad)
            return parameters
        raise ValueError(f"unsupported update_scope {scope!r}; use scorer or planner_vlm_lora")


def load_gigabrain07_policy(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    source_root: str | Path | None = None,
) -> Any:
    """Load one native GigaBrain policy directly on an explicit device.

    This is an entry point for a caller that has a complete local
    diffusers-style checkpoint.  It loads no weights and starts no process.
    The returned policy is intentionally separate from the official server
    closure; callers should attach official ``ImageTransform``,
    ``PromptTokenizerTransform``, ``Normalize`` and ``PadStatesAndActions``
    instances to a server adapter before calling :meth:`from_native`.
    """

    source_root = _ensure_giga_models_source(source_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import GigaBrain07Policy
    except ImportError as exc:
        raise NativeInterfaceError("cannot import GigaBrain07Policy from the official source tree") from exc

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "config.json").exists():
        for child in ("model_ema", "model"):
            candidate = checkpoint_dir / child
            if (candidate / "config.json").exists():
                checkpoint_dir = candidate
                break
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"GigaBrain checkpoint has no config.json: {config_path}")
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides: dict[str, Any] = {}
    if "has_action_expert" not in saved_config:
        weights_path = checkpoint_dir / "diffusion_pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(f"checkpoint is missing diffusion_pytorch_model.bin: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
        overrides["has_action_expert"] = any(
            str(name).startswith("paligemma_with_expert.") for name in state_dict.keys()
        )
    policy = GigaBrain07Policy.from_pretrained(str(checkpoint_dir), **overrides)
    if dtype is not None:
        policy = policy.to(dtype=dtype)
    policy = policy.to(device)
    policy.eval()
    return policy


def _ensure_giga_models_source(source_root: str | Path | None = None) -> Path:
    if source_root is None:
        source_root = Path(__file__).resolve().parents[1] / "gigabrain07_execution_comm_20261009" / "giga_models_src"
    source_root = Path(source_root).expanduser().resolve()
    if source_root.name == "giga_models":
        source_root = source_root.parent
    if not source_root.exists():
        raise NativeInterfaceError(f"official GigaBrain source tree not found: {source_root}")
    if not (source_root / "giga_models").exists():
        raise NativeInterfaceError(f"official GigaBrain source tree lacks giga_models/: {source_root}")
    return source_root


def _build_direct_official_preprocess(native_server: Any, policy: Any) -> Callable[[Mapping[str, Any]], Mapping[str, Any]] | None:
    """Build the official image/token/state path from explicitly exposed fields.

    This mirrors the data-only part of ``inference_agilex_server_unified`` and
    intentionally does not call its ``inference`` closure (which moves tensors
    to hard-coded CUDA).  The native loader can therefore put the policy and
    transforms on CPU for a small feature extraction pass.
    """

    field_names = (
        "image_transform",
        "prompt_transform",
        "state_normalize",
        "pad_states_and_actions",
        "embodiment_id",
        "present_img_keys",
    )
    values = {name: getattr(native_server, name, None) for name in field_names}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        return None
    image_transform = values["image_transform"]
    prompt_transform = values["prompt_transform"]
    state_normalize = values["state_normalize"]
    pad_states_and_actions = values["pad_states_and_actions"]
    embodiment_id = int(values["embodiment_id"])
    present_img_keys = list(values["present_img_keys"])
    state_input_mode = str(getattr(native_server, "state_input_mode", getattr(policy, "state_input_mode", "prompt")))
    observation_memory_size = int(
        getattr(native_server, "observation_memory_size", getattr(policy, "observation_memory_size", 1))
    )
    prepare_state_memory = getattr(native_server, "prepare_state_memory", None)
    device = torch.device(getattr(native_server, "device", None) or System2Backend._infer_device(policy) or "cpu")

    def _image_mapping(raw_images: Any) -> dict[str, Any]:
        """Map public camera aliases to the official observation keys."""
        aliases = {
            key.rsplit(".", 1)[-1]: key
            for key in present_img_keys
            if key.startswith("observation.images.")
        }

        def canonical_key(raw_key: Any) -> str:
            key = str(raw_key)
            if key in present_img_keys or key.endswith("_is_pad") and key[:-7] in present_img_keys:
                return key
            suffix = key
            if suffix.startswith("observation.images."):
                suffix = suffix[len("observation.images.") :]
            is_pad = suffix.endswith("_is_pad")
            base = suffix[:-7] if is_pad else suffix
            canonical = aliases.get(base)
            if canonical is None:
                return key
            return f"{canonical}_is_pad" if is_pad else canonical

        def normalise_image(value: Any) -> Tensor:
            image = _as_tensor(value, device=device)
            if image.ndim == 3:
                if image.shape[-1] == 3 and image.shape[0] != 3:
                    image = image.permute(2, 0, 1)
            elif image.ndim == 4:
                if image.shape[-1] == 3 and image.shape[1] != 3:
                    image = image.permute(0, 3, 1, 2)
            else:
                raise NativeInterfaceError(
                    f"image must have CHW/HWC or TCHW/THWC shape, got {tuple(image.shape)}"
                )
            if image.dtype == torch.uint8:
                image = image.to(dtype=torch.float32).div(255.0)
            elif not torch.is_floating_point(image):
                image = image.to(dtype=torch.float32)
            else:
                image = image.to(dtype=torch.float32)
                # Native inference supplies float_native in [0, 1].  This
                # also makes float arrays that still carry uint8 scale safe.
                if image.numel() and float(image.detach().amin()) >= 0.0 and float(image.detach().amax()) > 1.0:
                    image = image.div(255.0)
            return image.contiguous()

        if isinstance(raw_images, Mapping):
            nested = raw_images.get("observation.images")
            data = dict(nested) if isinstance(nested, Mapping) else dict(raw_images)
        else:
            if not isinstance(raw_images, (list, tuple)):
                raise NativeInterfaceError("images must be a mapping or a sequence matching present_img_keys")
            if not raw_images:
                return {}
            if len(raw_images) != len(present_img_keys):
                raise NativeInterfaceError(
                    f"images sequence has {len(raw_images)} items but present_img_keys has {len(present_img_keys)}"
                )
            data = dict(zip(present_img_keys, raw_images))
        canonical_data: dict[str, Any] = {}
        for raw_key, value in data.items():
            key = canonical_key(raw_key)
            if key.endswith("_is_pad"):
                canonical_data[key] = _as_tensor(value, device=device, dtype=torch.bool)
            elif isinstance(value, (Tensor, list, tuple)) or hasattr(value, "shape"):
                canonical_data[key] = normalise_image(value)
            else:
                canonical_data[key] = value
        return canonical_data

    def _direct_preprocess(request: Mapping[str, Any]) -> Mapping[str, Any]:
        data = _image_mapping(request.get("images", request.get("observation.images")))
        raw_state = request.get("state", request.get("observation.state"))
        if raw_state is None:
            raise NativeInterfaceError("native preprocessing request has no state")
        state = _as_tensor(raw_state, device=device, dtype=torch.float32)
        if state.ndim not in (1, 2):
            raise NativeInterfaceError(f"state must have shape [D] or [K,D], got {tuple(state.shape)}")
        state_present = bool(request.get("state_present", True))
        agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
        if not state_present:
            state = torch.zeros_like(state)
            agent_pos_mask.zero_()
        if state_input_mode == "proprio_anchor":
            state = torch.nan_to_num(state, nan=0.0)
        state_normed = state_normalize(state, embodiment_id=embodiment_id)
        current_state = state_normed[-1] if state_normed.ndim == 2 else state_normed

        state_memory = None
        state_memory_masks = None
        if state_input_mode == "proprio_memory":
            if callable(prepare_state_memory):
                current_state, state_memory, state_memory_masks = prepare_state_memory(state_normed, data)
            elif state_normed.ndim == 2:
                if state_normed.shape[0] != observation_memory_size:
                    raise NativeInterfaceError(
                        f"state memory has {state_normed.shape[0]} frames, expected {observation_memory_size}"
                    )
                current_state = state_normed[-1]
                state_memory = state_normed
                state_memory_masks = torch.ones(observation_memory_size, dtype=torch.bool, device=device)
            else:
                current_state = state_normed
                state_memory = state_normed.unsqueeze(0).repeat(observation_memory_size, 1)
                state_memory_masks = torch.ones(observation_memory_size, dtype=torch.bool, device=device)

        task = request.get("task")
        if not isinstance(task, str) or not task.strip():
            raise NativeInterfaceError("native preprocessing request task must be a non-empty string")
        prompt_data: dict[str, Any] = {"task": task, "embodiment_id": embodiment_id}
        if state_input_mode == "prompt" and state_present:
            prompt_data["observation.state"] = current_state
        prompt_output = prompt_transform(prompt_data)
        if isinstance(prompt_output, Mapping):
            lang_tokens = prompt_output["input_ids"]
            lang_masks = prompt_output.get("attention_mask", prompt_output.get("lang_masks"))
            lang_att_masks = prompt_output.get("lang_att_masks")
        else:
            if len(prompt_output) < 3:
                raise NativeInterfaceError("prompt_transform must return language tokens, masks and attention masks")
            lang_tokens, lang_masks, lang_att_masks = prompt_output[:3]
        if lang_masks is None:
            raise NativeInterfaceError("prompt_transform did not return language masks")
        # The official anchor tokenizer always emits <|propri|> in
        # ``proprio_anchor`` mode.  A curated text-only bootstrap row has no
        # observed state, so keeping that token would contradict
        # ``proprioception_present=False`` and trigger the native count check.
        # Remove it from the selector prompt only; the actor's transform and
        # inference closure remain unchanged.
        lang_tokens = _as_tensor(lang_tokens, device=device, dtype=torch.long)
        lang_masks = _as_tensor(lang_masks, device=device, dtype=torch.bool)
        lang_att_masks = (
            None
            if lang_att_masks is None
            else _as_tensor(lang_att_masks, device=device, dtype=torch.bool)
        )
        if state_input_mode == "proprio_anchor" and not state_present:
            propri_token_id = getattr(policy, "propri_token_id", None)
            if propri_token_id is None:
                propri_token_id = getattr(prompt_transform, "propri_token_id", None)
            if propri_token_id is not None:
                anchor_mask = lang_tokens == int(propri_token_id)
                tokenizer = getattr(prompt_transform, "paligemma_tokenizer", None)
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = 0
                lang_tokens = lang_tokens.masked_fill(anchor_mask, int(pad_token_id))
                lang_masks = lang_masks & ~anchor_mask
                if lang_att_masks is not None:
                    lang_att_masks = lang_att_masks & ~anchor_mask
        # The planner owns the longer prompt budget (512 by default), while
        # the official tokenizer right-pads to that budget.  Trim only the
        # trailing invalid region after all selector text has been tokenized;
        # this preserves candidate/memory/history tokens and avoids paying for
        # hundreds of masked language positions on CPU.
        if lang_masks.ndim not in (1, 2):
            raise NativeInterfaceError(f"language masks must have shape [L] or [B,L], got {tuple(lang_masks.shape)}")
        mask_for_trim = lang_masks if lang_masks.ndim == 2 else lang_masks.unsqueeze(0)
        positions = torch.arange(mask_for_trim.shape[-1], device=device).unsqueeze(0).expand_as(mask_for_trim)
        last_valid = positions.masked_fill(~mask_for_trim, -1).max(dim=-1).values
        trim_length = max(1, int(last_valid.max().item()) + 1)
        lang_tokens = lang_tokens[..., :trim_length]
        lang_masks = lang_masks[..., :trim_length]
        if lang_att_masks is not None:
            lang_att_masks = lang_att_masks[..., :trim_length]
        state_padded = pad_states_and_actions({"observation.state": current_state})["observation.state"]
        proprioception = None
        anchor_agent_pos_mask = None
        proprioception_present = None
        if state_input_mode == "proprio_anchor":
            current_agent_pos_mask = agent_pos_mask[-1] if agent_pos_mask.ndim == 2 else agent_pos_mask
            mask_padded = pad_states_and_actions({"observation.state": current_agent_pos_mask})["observation.state"]
            proprioception = state_padded.unsqueeze(0).unsqueeze(0)
            anchor_agent_pos_mask = mask_padded.unsqueeze(0).unsqueeze(0)
            proprioception_present = torch.tensor([state_present], dtype=torch.bool, device=device)

        imgs, img_masks, _ = image_transform(data)
        has_valid_image = any(bool(mask.any().item()) for mask in img_masks)
        return {
            "images": [img.unsqueeze(0).to(device) for img in imgs],
            "img_masks": [mask.unsqueeze(0).to(device) for mask in img_masks],
            "lang_tokens": lang_tokens.unsqueeze(0) if lang_tokens.ndim == 1 else lang_tokens,
            "lang_masks": lang_masks.unsqueeze(0) if lang_masks.ndim == 1 else lang_masks,
            "lang_att_masks": None
            if lang_att_masks is None
            else lang_att_masks.unsqueeze(0) if lang_att_masks.ndim == 1 else lang_att_masks,
            "proprioception": proprioception,
            "agent_pos_mask": anchor_agent_pos_mask,
            "proprioception_present": proprioception_present,
            "state_present": state_present,
            "state_memory": None if state_memory is None else _as_tensor(state_memory, device=device, dtype=torch.float32).unsqueeze(0)
            if state_memory is not None and _as_tensor(state_memory).ndim == 2
            else state_memory,
            "state_memory_masks": None
            if state_memory_masks is None
            else _as_tensor(state_memory_masks, device=device, dtype=torch.bool).unsqueeze(0)
            if _as_tensor(state_memory_masks).ndim == 1
            else _as_tensor(state_memory_masks, device=device, dtype=torch.bool),
            "emb_ids": torch.tensor([embodiment_id], dtype=torch.long, device=device),
            "observation_mode": "rgb_vlm" if has_valid_image else "text_only_bootstrap",
        }

    return _direct_preprocess


def _load_make_att_2d_masks() -> Callable[[Tensor, Tensor], Tensor]:
    """Load the official mask helper only when the native path is used."""

    try:
        from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import make_att_2d_masks

        return make_att_2d_masks
    except ImportError:
        source_root = Path(__file__).resolve().parents[1] / "gigabrain07_execution_comm_20261009" / "giga_models_src"
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import make_att_2d_masks

            return make_att_2d_masks
        except ImportError as exc:
            raise NativeInterfaceError(
                "cannot import official make_att_2d_masks; attach the GigaBrain source package before native inference"
            ) from exc


def load_feature_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, MutableMapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _load_external_factory(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("factory must be written as module:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {spec}")
    return factory


__all__ = [
    "CandidateScorer",
    "INDEX_TO_STATUS",
    "LoRALinear",
    "MemoryRecord",
    "NativeInterfaceError",
    "SkillCandidate",
    "STATUS_TO_INDEX",
    "System2Backend",
    "inject_lora",
    "load_gigabrain07_policy",
    "load_feature_rows",
]
