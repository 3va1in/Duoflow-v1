"""Public-observation transition evidence for bounded harness edits.

Only consecutive public RGB and qpos observations are compared.  The module
does not identify objects, contacts, grasping, or semantic completion.  Its
semantic state is therefore deliberately ``uncertain``; the small finite
editor may use only the conservative stagnation signal.  A missing final
frame always makes the overall transition ``unknown``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - the remote runtime supplies numpy.
    _np = None


_NON_RGB_KEYS = {
    "qpos", "state", "joint", "joint_state", "proprioception",
    "depth", "depth_image", "depth_map", "intrinsic", "intrinsics",
    "camera_matrix", "cam_intrinsics", "extrinsic", "extrinsics",
    "segmentation", "mask", "masks", "normal", "normals",
    "pointcloud", "point_cloud", "metadata", "info", "timestamp",
}


def _is_non_rgb_key(key: str) -> bool:
    """Reject auxiliary camera data while retaining camera/color containers."""

    return key in _NON_RGB_KEYS or any(
        token in key for token in ("depth", "intrinsic", "extrinsic", "segmentation", "pointcloud")
    )


def _numeric_array(value: Any):
    """Flatten finite numeric data, using numpy when it is available."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, Mapping):
        return None
    if _np is not None:
        try:
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = _np.asarray(value)
            if array.dtype.kind not in "biufc":
                return None
            array = array.reshape(-1)
            if not array.size or not _np.isfinite(array).all():
                return None
            return array.astype(_np.float32, copy=False)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            child = _numeric_array(item)
            if child is None:
                return None
            result.extend(child)
        return result or None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return [number] if number == number and abs(number) != float("inf") else None


def _collect_rgb(value: Any, prefix: str = "default") -> dict[str, Any]:
    """Collect only public RGB/color arrays from common observation layouts."""

    direct = _numeric_array(value)
    if direct is not None:
        return {prefix: direct}
    if not isinstance(value, Mapping):
        return {}
    explicit_color = next(
        ((key, child) for key, child in value.items()
         if str(key).lower() in {"color", "rgb"}),
        None,
    )
    if explicit_color is not None:
        # A frame mapping may also carry depth/intrinsics and arbitrary
        # metadata.  An explicit color/RGB field is the only accepted image.
        return _collect_rgb(explicit_color[1], prefix)
    result: dict[str, Any] = {}
    for key, child in value.items():
        key_text = str(key).lower()
        if _is_non_rgb_key(key_text):
            continue
        # A camera frame often contains color, depth and calibration fields;
        # once an explicit color/RGB field is present, recurse into that field
        # only so depth and intrinsics never become pseudo-cameras.
        child_prefix = key_text if prefix == "default" else f"{prefix}/{key_text}"
        result.update(_collect_rgb(child, child_prefix))
    return result


def _extract(observation: Any, *, rgb: Any = None, qpos: Any = None) -> tuple[dict[str, Any], Any]:
    if rgb is None and isinstance(observation, Mapping):
        for key in ("rgb", "image", "images", "vision", "color", "camera"):
            if key in observation:
                rgb = observation[key]
                break
    if qpos is None and isinstance(observation, Mapping):
        for key in ("qpos", "joint_state", "joint", "state", "proprioception"):
            if key in observation:
                qpos = observation[key]
                break
    return _collect_rgb(rgb), _numeric_array(qpos)


def _rgb_stats(before: dict[str, Any], after: dict[str, Any], eps: float) -> dict[str, Any]:
    keys = sorted(set(before) & set(after))
    deltas: list[float] = []
    changed: list[float] = []
    for key in keys:
        first, last = before[key], after[key]
        if len(first) != len(last):
            continue
        if not len(first):
            continue
        if _np is not None:
            first_scale = 255.0 if float(_np.max(_np.abs(first))) > 1.5 else 1.0
            last_scale = 255.0 if float(_np.max(_np.abs(last))) > 1.5 else 1.0
            delta = _np.abs(first / first_scale - last / last_scale)
            deltas.append(float(delta.mean()))
            changed.append(float(_np.mean(delta > eps)))
        else:
            first_scale = 255.0 if max(abs(item) for item in first) > 1.5 else 1.0
            last_scale = 255.0 if max(abs(item) for item in last) > 1.5 else 1.0
            delta = [abs(left / first_scale - right / last_scale) for left, right in zip(first, last)]
            deltas.append(sum(delta) / len(delta))
            changed.append(sum(item > eps for item in delta) / len(delta))
    if not deltas:
        return {"state": "unknown", "cameras_compared": 0}
    mean_delta = sum(deltas) / len(deltas)
    fraction = sum(changed) / len(changed)
    return {
        "state": "changed" if mean_delta > eps or fraction > 0.01 else "unchanged",
        "mean_abs_delta": mean_delta,
        "changed_fraction": fraction,
        "cameras_compared": len(deltas),
    }


def _qpos_stats(before: Any, after: Any, eps: float) -> dict[str, Any]:
    if before is None or after is None or len(before) != len(after):
        return {"state": "unknown", "dimensions_compared": 0}
    if not len(before):
        return {"state": "unknown", "dimensions_compared": 0}
    if _np is not None:
        delta = _np.abs(before - after)
        mean_delta = float(delta.mean())
        max_delta = float(delta.max())
    else:
        delta = [abs(left - right) for left, right in zip(before, after)]
        mean_delta = sum(delta) / len(delta)
        max_delta = max(delta)
    return {
        "state": "changed" if max_delta > eps else "unchanged",
        "mean_abs_delta": mean_delta,
        "max_abs_delta": max_delta,
        "dimensions_compared": len(delta),
    }


def build_transition_evidence(
    before: Mapping[str, Any] | Any = None,
    after: Mapping[str, Any] | Any = None,
    *,
    before_rgb: Any = None,
    after_rgb: Any = None,
    before_qpos: Any = None,
    after_qpos: Any = None,
    rgb_epsilon: float = 1.0 / 255.0,
    qpos_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Return JSON-safe, non-semantic evidence for a public transition.

    ``before`` and ``after`` may be full observations, while explicit keyword
    values are useful when the caller already split RGB and qpos.  A missing
    ``after`` observation or missing final frame marker yields an overall
    ``stagnation_state == 'unknown'`` regardless of the other modality.
    """

    final_missing = after is None or (
        isinstance(after, Mapping)
        and after.get("final_frame_missing") is True
    )
    before_images, before_joint = _extract(before, rgb=before_rgb, qpos=before_qpos)
    after_images, after_joint = _extract(after, rgb=after_rgb, qpos=after_qpos)
    if final_missing:
        after_images, after_joint = {}, None
    rgb = _rgb_stats(before_images, after_images, float(rgb_epsilon))
    qpos = _qpos_stats(before_joint, after_joint, float(qpos_epsilon))
    if final_missing or rgb["state"] == "unknown" or qpos["state"] == "unknown":
        stagnation_state = "unknown"
        stagnated = None
        stagnation_chunks = None
    elif rgb["state"] == "unchanged" and qpos["state"] == "unchanged":
        stagnation_state = "stagnated"
        stagnated = True
        stagnation_chunks = 1
    else:
        stagnation_state = "changed"
        stagnated = False
        stagnation_chunks = 0
    return {
        "schema_version": "0.1",
        "source": "public_rgb_qpos",
        "semantic_state": "uncertain",
        "final_frame_missing": bool(final_missing),
        "rgb": rgb,
        "qpos": qpos,
        "stagnation_state": stagnation_state,
        "stagnated": stagnated,
        "stagnation_chunks": stagnation_chunks,
    }


def compute_transition_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for callers using a compute-style name."""

    return build_transition_evidence(*args, **kwargs)


__all__ = ["build_transition_evidence", "compute_transition_evidence"]
