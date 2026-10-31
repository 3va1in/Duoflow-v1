"""只导出已训练条件的公开技能卡，供上层规划器调用独立执行器。"""
import argparse
import json
import math
from pathlib import Path

from prepare_training_data import action_valid, condition_key


# 仅旧两任务的采集记录缺少这两项元数据；新任务必须使用自己的技能定义。
LEGACY_DEFAULTS = {
    'pick_and_lift': (250, 'The named object is on its support and reachable by the specified arm.'),
    'reorient_and_place': (350, 'The specified arm already holds the named object above the table.'),
    'place_and_release': (350, 'The specified arm already holds the named object; the named support is available.'),
    'retreat': (150, 'The named object has been placed and released; the specified gripper is open.'),
}


def skill_contract(segment):
    maximum, precondition = segment.get('max_policy_steps'), segment.get('precondition')
    if segment['task'] in ('rotate_qrcode', 'stack_bowls_three'):
        legacy = LEGACY_DEFAULTS.get(segment['skill_id'])
        if legacy:
            maximum = legacy[0] if maximum is None else maximum
            precondition = legacy[1] if precondition is None else precondition
    if type(maximum) is not int or maximum <= 0 or not isinstance(precondition, str) or not precondition.strip():
        raise ValueError(f"技能段缺少有效max_policy_steps/precondition: {condition_key(segment)}")
    return maximum, precondition


def build_catalog(data_root, actor, completion, checkpoint, completion_checkpoint):
    if actor.get('format') != 'atomic_skill_lora_v1' or actor.get('run_complete') is not True:
        raise ValueError('动作checkpoint不是已完成的技能训练结果')
    if completion.get('format') != 'atomic_skill_completion_repair_v1':
        raise ValueError('必须提供本轮重训完成头checkpoint')
    conditions = actor['condition_to_index']
    if not conditions or sorted(conditions.values()) != list(range(len(conditions))):
        raise ValueError('动作checkpoint的条件索引不连续或重复')
    if completion.get('condition_to_index') != conditions:
        raise ValueError('动作策略与完成头条件词表或索引不一致')
    stop = dict(completion['stop_config'])
    threshold, consecutive = stop.get('threshold'), stop.get('consecutive_observations')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('完成头停止阈值无效')
    if type(consecutive) is not int or consecutive < 1:
        raise ValueError('完成头连续观测数无效')
    records = {}
    for path in sorted(Path(data_root).glob('*/seed_*/segments.json')):
        for segment in json.loads(path.read_text()):
            condition = condition_key(segment)
            if (condition not in conditions or segment['split'] != 'train' or
                    not action_valid(segment) or segment['end_frame'] <= segment['start_frame']):
                continue
            maximum, precondition = skill_contract(segment)
            key = (condition, segment['task'], json.dumps(segment['bindings'], sort_keys=True))
            record = {'condition_index': conditions[condition], 'condition_key': condition,
                      'skill_id': segment['skill_id'], 'skill_text': condition.split('|', 1)[1],
                      'bindings': segment['bindings'], 'task': segment['task'],
                      'max_policy_steps': maximum, 'precondition': precondition}
            if key in records and records[key] != record:
                raise ValueError(f'同一技能绑定的执行约束不一致: {key}')
            records[key] = record
    if not records:
        raise ValueError('没有同时属于已训练词表且具有有效动作训练段的技能')
    cards, seen_indices = [], {}
    for key, record in sorted(records.items(), key=lambda item: (item[1]['condition_index'], item[0])):
        index = record['condition_index']
        occurrence = seen_indices.get(index, 0)
        seen_indices[index] = occurrence + 1
        identifier = f'atomic_{index:03d}' + (f'_binding{occurrence + 1}' if occurrence else '')
        cards.append({'id': identifier, **record, 'policy_checkpoint': str(checkpoint),
                      'completion_checkpoint': str(completion_checkpoint), 'stop_config': stop,
                      'executor': 'SkillExecutor.start/step',
                      'termination': f'learned completion probability >= {threshold:g} on {consecutive} consecutive observations; timeout is separate',
                      'automatic_next_skill': False,
                      'capability_status': 'trained; inspect held-out skill evaluation before making capability claims'})
    return {
        'format': 'duoflow_atomic_skill_catalog_v1',
        'policy_structure': 'shared skill-conditioned action LoRA; explicit skill and object/arm binding',
        'observation_inputs': ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'state14'],
        'planner_contract': 'Choose a listed card and call it without changing its skill_text or bindings. Wait for its execution result before requesting another skill.',
        'no_oracle_inputs': True, 'stop_config': stop,
        'trained_condition_count': len(conditions), 'exported_condition_count': len(seen_indices),
        'unexported_conditions_without_valid_action_segments': sorted(
            condition for condition, index in conditions.items() if index not in seen_indices),
        'skills': cards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--completion-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.training_root / 'TRAIN_RESULT.json').read_text())
    completion_result = json.loads((args.completion_root / 'TRAIN_RESULT.json').read_text())
    if result['status'] != 'complete' or completion_result['status'] != 'complete':
        raise ValueError('动作策略或完成头训练尚未完成，不能发布为可加载技能')
    checkpoint = args.training_root / 'skill_checkpoint.pt'
    completion_checkpoint = args.completion_root / 'completion_checkpoint.pt'
    import torch
    actor = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completion = torch.load(completion_checkpoint, map_location='cpu', weights_only=False)
    payload = build_catalog(args.data_root, actor, completion, checkpoint, completion_checkpoint)
    with args.output.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'exported_conditions': payload['exported_condition_count'],
                      'skill_cards': len(payload['skills'])}))


if __name__ == '__main__':
    main()
