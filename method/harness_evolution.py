"""从既有训练证据提出规则，并用独立开发回合选择冻结版本。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


TASKS = ('place_object_basket', 'stack_bowls_three', 'beat_block_hammer', 'handover_block', 'open_microwave', 'rotate_qrcode')
CONDITION = "demo_randomized"
STATUSES = {"satisfied", "unsatisfied", "unknown"}
SCHEMA_NAME = "harness_patch_v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _norm(value: str) -> str:
    return " ".join(value.split())


def default_harness(task: str, condition: str = CONDITION) -> dict:
    if task not in TASKS or condition != CONDITION:
        raise ValueError("本试验仅支持预定任务的 Randomized 条件")
    return {
        "version": "h0",
        "rules": [
            {"rule_id": f"{task}:advance", "scope": {"task": task, "condition": condition},
             "operation": "advance_gate", "parameters": {"allowed_status": ["satisfied"], "min_attempts": 1},
             "training_refs": [], "hypothesis": "仅按公开观测估计子目标满足后推进；预测不等于真实阶段标签。"},
            {"rule_id": f"{task}:recovery", "scope": {"task": task, "condition": condition},
             "operation": "recovery", "parameters": {"trigger_status": ["unsatisfied", "unknown"],
                                                         "after_attempts": 2, "max_revisions": 1},
             "training_refs": [], "hypothesis": "公开观测未满足或不确定时，允许在原动作预算内修改一次剩余计划。"},
        ],
    }


def validate_harness(payload: dict, task: str, condition: str = CONDITION,
                     known_training_refs: set[str] | None = None) -> dict:
    default_harness(task, condition)
    if not isinstance(payload, dict) or set(payload) != {"version", "rules"}:
        raise ValueError("Harness 只允许 version 和 rules 字段")
    if not isinstance(payload["version"], str) or not payload["version"].strip():
        raise ValueError("Harness version 不能为空")
    if not isinstance(payload["rules"], list):
        raise ValueError("rules 必须为数组")
    rules = [r for r in payload["rules"] if isinstance(r, dict) and
             r.get("scope") == {"task": task, "condition": condition}]
    if len(rules) != 2 or {r.get("operation") for r in rules} != {"advance_gate", "recovery"}:
        raise ValueError("每任务必须恰有 advance_gate 与 recovery 两条规则")
    for rule in rules:
        if set(rule) != {"rule_id", "scope", "operation", "parameters", "training_refs", "hypothesis"}:
            raise ValueError("规则包含缺失或未知字段")
        if not isinstance(rule["rule_id"], str) or not rule["rule_id"]:
            raise ValueError("rule_id 必须是非空字符串")
        if not isinstance(rule["hypothesis"], str) or not rule["hypothesis"].strip():
            raise ValueError("hypothesis 必须是非空字符串")
        refs = rule["training_refs"]
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError("training_refs 必须为字符串数组")
        if known_training_refs is not None and not set(refs).issubset(known_training_refs):
            raise ValueError("提案引用了未提供的训练证据")
        params = rule["parameters"]
        if not isinstance(params, dict):
            raise ValueError("parameters 必须为对象")
        if rule["operation"] == "advance_gate":
            if set(params) != {"allowed_status", "min_attempts"}:
                raise ValueError("advance_gate 参数不合法")
            status_key, count_key = "allowed_status", "min_attempts"
            allowed = {"satisfied", "unknown"}
        else:
            if set(params) != {"trigger_status", "after_attempts", "max_revisions"}:
                raise ValueError("recovery 参数不合法")
            status_key, count_key = "trigger_status", "after_attempts"
            allowed = STATUSES
            if type(params["max_revisions"]) is not int or not 0 <= params["max_revisions"] <= 2:
                raise ValueError("max_revisions 必须在 0..2")
        statuses = params[status_key]
        if (not isinstance(statuses, list) or not statuses or any(not isinstance(s, str) for s in statuses)
                or len(set(statuses)) != len(statuses) or not set(statuses).issubset(allowed)):
            raise ValueError("状态条件不合法")
        if type(params[count_key]) is not int or not 1 <= params[count_key] <= 4:
            raise ValueError(f"{count_key} 必须在 1..4")
    if len({r["rule_id"] for r in rules}) != 2:
        raise ValueError("规则 ID 重复")
    return {"version": payload["version"], "rules": deepcopy(rules)}


def load_harness(path: str | Path, task: str, condition: str = CONDITION) -> dict:
    return validate_harness(_read(Path(path)), task, condition)


def _effective(harness: dict) -> dict:
    return {r["operation"]: r["parameters"] for r in harness["rules"]}


def _summarize_training_cell(source: Path, cell_id: str) -> dict:
    cell = source / "train_runs" / cell_id
    config = _read(cell / "config.json")
    if config["split"] != "train":
        raise ValueError("不能读取评测终局作为训练证据")
    events = _rows(cell / "decisions.jsonl")
    outcomes = [e for e in events if e.get("record_type") == "episode_outcome"]
    if len(outcomes) != 1 or outcomes[0].get("split") != "train":
        raise ValueError(f"训练格缺少唯一训练终局: {cell_id}")
    outcome = outcomes[0]
    decisions = [e for e in events if e.get("selected_skill_id") and not e.get("terminal")]
    transitions = [e for e in events if e.get("event") == "harness_transition"]
    steps = []
    for event in decisions:
        skill = event.get("selected_base_skill_id", event["selected_skill_id"].split("__step_")[0])
        if steps and steps[-1]["skill_id"] == skill:
            steps[-1]["attempted_chunks"] += 1
        else:
            steps.append({"skill_id": skill, "attempted_chunks": 1,
                          "execution_prompt": event.get("execution_prompt", "")})
    return {
        "cell_id": cell_id, "episode_id": outcome["episode_id"], "split": "train",
        "task": outcome["task"], "condition": outcome["condition"], "seed": int(outcome["seed"]),
        "instruction": _norm(outcome["instruction"]),
        "episode_success": outcome["execution_feedback"] == "success", "label_scope": "whole_episode",
        "attempted_skill_sequence": steps, "stage_completion_labels": "unavailable",
        "public_transition_counts": dict(Counter((e.get("transition_evidence") or {}).get(
            "stagnation_state", "unknown") for e in transitions)),
        "source_records": [str((cell / "config.json").resolve()), str((cell / "decisions.jsonl").resolve())],
    }


def prepare_training_assets(source: str | Path, output: str | Path) -> dict:
    source, output = Path(source), Path(output)
    feedback = _rows(source / "edit_feedback.jsonl")
    valid = [row for row in feedback if row.get("delta") is not None]
    if len(valid) != 205:
        raise ValueError(f"预定冻结训练源应有 205 个有效对照，实际为 {len(valid)}")
    cache: dict[str, dict] = {}
    evidence = {task: [] for task in TASKS}
    excluded = []
    for row in feedback:
        if row["task"] not in TASKS:
            continue
        if row.get("delta") is None:
            excluded.append({"task": row["task"], "candidate": Path(row["candidate_run"]).name,
                             "reason": row.get("update_skipped", "unmatched")})
            continue
        comparison = row["comparison"]
        if comparison["matched"] != 1 or comparison["unmatched_reference"] or comparison["unmatched_candidate"]:
            raise ValueError("有效训练对照的实际场景或指令不匹配")
        cells = []
        for field in ("reference_run", "candidate_run"):
            cell_id = Path(row[field]).name
            if cell_id not in cache:
                cache[cell_id] = _summarize_training_cell(source, cell_id)
            cells.append(cache[cell_id])
        reference, candidate = cells
        pair = comparison["pairs"][0]
        for item, success_key in ((reference, "reference_success"), (candidate, "candidate_success")):
            if (item["seed"] != pair["seed"] or item["instruction"] != _norm(pair["instruction"])
                    or item["episode_success"] != pair[success_key] or item["task"] != row["task"]):
                raise ValueError("训练原始轨迹与配对回执不一致")
        delta = int(candidate["episode_success"]) - int(reference["episode_success"])
        if delta != row["delta"]:
            raise ValueError("训练收益与原始终局不一致")
        evidence[row["task"]].append({
            "evidence_id": "train_pair:" + candidate["cell_id"], "split": "train", "round": row["round"],
            "condition": row["condition"], "candidate_group": row["group"], "tested_patch": row["patch"],
            "whole_episode_delta": delta, "reference": reference, "candidate": candidate,
            "credit_limit": "终局差异不能标成任何单步完成标签或补丁局部因果收益。",
        })
    memory = _rows(source / "execution_memory" / "events.jsonl")
    memory_cases = defaultdict(list)
    for item in memory:
        if item.get("task") in TASKS:
            if item.get("split") != "train":
                raise ValueError("训练记忆包含非训练记录")
            memory_cases[item["episode_id"]].append(item)
    for task in TASKS:
        task_dir = output / task
        cases = []
        for episode_id, events in memory_cases.items():
            terminal = [e for e in events if e.get("terminal")]
            if not terminal or terminal[0]["task"] != task:
                continue
            if len(terminal) != 1:
                raise ValueError("训练记忆的回合终局不唯一")
            outcome = terminal[0]
            sequence = [e.get("execution_prompt", e.get("selected_skill_id", ""))
                        for e in events if not e.get("terminal")]
            cases.append({"evidence_id": "train_memory:" + episode_id, "split": "train",
                          "condition": outcome["condition"], "instruction": outcome["instruction"],
                          "episode_success": outcome["execution_feedback"] == "success",
                          "attempted_sequence": list(dict.fromkeys(sequence)),
                          "label_scope": "whole_episode", "stage_completion_labels": "unavailable"})
        bundle = {"task": task, "source_split": "train", "paired_evidence": evidence[task],
                  "shared_training_memory": cases, "evaluation_data_used": False,
                  "positive_negative_zero_pairs_preserved": True}
        _write_new(task_dir / "TRAINING_EVIDENCE.json", bundle)
        _write_new(task_dir / "H0.json", default_harness(task))
        _write_new(task_dir / "REQUEST.json", build_proposal_request(bundle))
    manifest = {
        "source": str(source.resolve()), "source_valid_pairs": len(valid), "source_unmatched_pairs": len(feedback) - len(valid),
        "tasks": {task: {"valid_pairs": len(evidence[task]),
                          "reward_counts": dict(Counter(str(e["whole_episode_delta"]) for e in evidence[task]))}
                  for task in TASKS},
        "excluded_target_pairs": excluded, "evaluation_data_used": False,
        "shared_rag_for_H0_and_Hprime": True, "proposal_limit_per_task": 1,
    }
    _write_new(output / "TRAINING_MANIFEST.json", manifest)
    _write_new(output / "H0.json", {"version": "h0", "rules": [rule for task in TASKS
                                                                for rule in default_harness(task)["rules"]]})
    return manifest


def build_proposal_request(bundle: dict) -> dict:
    task = bundle["task"]
    return {
        "mode": "propose_harness_patch", "schema_name": SCHEMA_NAME,
        "context": {
            "task": task, "condition": CONDITION, "h0": default_harness(task), "training_evidence": bundle,
            "output_contract": {
                "top_level": ["version", "rules"], "rule_fields": ["rule_id", "scope", "operation", "parameters", "training_refs", "hypothesis"],
                "advance_gate": {"allowed_status": "nonempty subset of satisfied,unknown", "min_attempts": "integer 1..4"},
                "recovery": {"trigger_status": "nonempty subset of satisfied,unsatisfied,unknown", "after_attempts": "integer 1..4", "max_revisions": "integer 0..2"},
                "rules_per_task": 2,
            },
            "instructions": [
                "只输出符合 H0 同一结构的完整候选 JSON，不输出 Markdown。每任务只允许本次一项提案，不会继续追问优化。",
                "依据给定 TRAIN 正负与零收益证据，最多修改一条规则的一个参数；若证据不足可保持 H0 参数不变。",
                "将 version 设为 hprime-加任务名。每条规则引用实际提供的 evidence_id，说明假设而非声称已经提升。",
                "全回合成功不代表每个技能成功；尝试次数、运动变化和静止均不是阶段完成真值。所有运行时状态只是公开观测预测。",
                "低层 PaliGemma2/VLA、技能库、RAG训练语料与环境动作预算保持不变；只允许 advance_gate 和 recovery。",
                "开发和测试回合尚未运行；不能请求或使用其结果来生成此提案。",
            ],
        },
    }


def save_candidate_response(task_dir: str | Path, payload: dict, response_metadata: dict | None = None) -> dict:
    task_dir = Path(task_dir)
    bundle = _read(task_dir / "TRAINING_EVIDENCE.json")
    task = bundle["task"]
    _write_new(task_dir / "PROPOSAL_RESPONSE.json", {"payload": payload, "metadata": response_metadata or {}})
    if len(payload.get("rules", [])) != 2:
        raise ValueError("每次提案只能包含本任务两条规则")
    refs = {e["evidence_id"] for e in bundle["paired_evidence"] + bundle["shared_training_memory"]}
    candidate = validate_harness(payload, task, known_training_refs=refs)
    h0 = default_harness(task)
    differences = [(op, key) for op, params in _effective(candidate).items()
                   for key, value in params.items() if value != _effective(h0)[op][key]]
    if len(differences) > 1:
        raise ValueError("一次候选只允许改变一个规则参数")
    if candidate["version"] != "hprime-" + task:
        raise ValueError("候选版本名不符合独立提案约定")
    for rule in candidate["rules"]:
        if not rule["training_refs"]:
            raise ValueError("候选规则必须引用已提供的训练证据")
    _write_new(task_dir / "candidate.json", candidate)
    receipt = {"task": task, "origin": "astra_response", "changed_parameters": differences,
               "candidate_differs_from_h0": bool(differences), "proposal_attempts": 1,
               "development_data_used": False, "evaluation_data_used": False}
    _write_new(task_dir / "PROPOSAL_RECEIPT.json", receipt)
    return receipt


def select_development(episodes: list[dict], candidates: dict[str, dict],
                       output_dir: str | Path | None = None) -> dict:
    """调用方仅传入 accepted 完整开发格；2任务×2场景×2版本固定8回合。"""
    for task, value in candidates.items():
        if "h0_path" in value and _effective(load_harness(value["h0_path"], task)) != _effective(default_harness(task)):
            raise ValueError("开发 H0 与预先声明的初始规则不同")
    candidates = {task: (load_harness(value["candidate_path"], task) if "candidate_path" in value else value)
                  for task, value in candidates.items()}
    if set(candidates) != set(TASKS):
        raise ValueError("必须同时提供两个预定任务候选")
    grouped: dict[tuple, dict] = {}
    for row in episodes:
        if row.get("task") not in TASKS or row.get("condition") != CONDITION or row.get("method") not in {"H0", "Hprime"}:
            raise ValueError("开发记录任务、条件或版本错误")
        if row.get("split", "dev") != "dev" or row.get("accepted", True) is not True:
            raise ValueError("只能使用完整有效开发回合")
        if type(row["success"]) is not bool or type(row["seed"]) is not int:
            raise ValueError("开发 success/seed 类型错误")
        key = (row["task"], row["method"], row["seed"], _norm(row["instruction"]))
        if key in grouped:
            raise ValueError("开发回合实际场景重复")
        grouped[key] = row
    reports = {}
    complete = len(episodes) == 8
    total_net = 0
    for task in TASKS:
        validate_harness(candidates[task], task)
        arms = {arm: {(k[2], k[3]): value for k, value in grouped.items() if k[:2] == (task, arm)}
                for arm in ("H0", "Hprime")}
        complete = complete and all(len(arm) == 2 for arm in arms.values())
        matched = sorted(set(arms["H0"]) & set(arms["Hprime"]))
        wins = sum(not arms["H0"][k]["success"] and arms["Hprime"][k]["success"] for k in matched)
        losses = sum(arms["H0"][k]["success"] and not arms["Hprime"][k]["success"] for k in matched)
        total_net += wins - losses
        reports[task] = {"matched": len(matched), "H0_successes": sum(arms["H0"][k]["success"] for k in matched),
                         "Hprime_successes": sum(arms["Hprime"][k]["success"] for k in matched),
                         "new_win_old_lose": wins, "old_win_new_lose": losses, "net_wins": wins - losses,
                         "unmatched": {arm: [arms[arm][k] for k in sorted(set(arms[arm]) - set(matched))]
                                       for arm in arms},
                         "pairs": [{"seed": k[0], "instruction": k[1], "H0_success": arms["H0"][k]["success"],
                                    "Hprime_success": arms["Hprime"][k]["success"]} for k in matched]}
    differs = any(_effective(candidates[t]) != _effective(default_harness(t)) for t in TASKS)
    promoted = bool(complete and all(r["matched"] > 0 and r["net_wins"] >= 0 for r in reports.values())
                    and total_net > 0 and differs)
    result = {"complete_development_budget": complete, "development_episodes": len(episodes),
              "paired": reports, "net_wins": total_net, "promoted": promoted, "run_c": promoted,
              "selected_version": "Hprime" if promoted else "H0",
              "selection_scope": "exploratory_resource_selection_not_significance",
              "reason": "paired_net_positive_without_task_regression" if promoted else "retain_H0_and_skip_equivalent_C",
              "selected_harnesses": {t: candidates[t] if promoted else default_harness(t) for t in TASKS}}
    if output_dir is not None:
        output_dir = Path(output_dir)
        result["selected_paths"] = {}
        for task, harness in result["selected_harnesses"].items():
            path = output_dir / (task + ".json")
            _write_new(path, harness)
            result["selected_paths"][task] = str(path.resolve())
        _write_new(output_dir / "selection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare_training_assets(arguments.source, arguments.output), ensure_ascii=False, indent=2))
