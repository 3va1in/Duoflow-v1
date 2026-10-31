"""汇总实际采集目录；只链接源数据，不筛掉失败段或补造监督。"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def condition_key(segment):
    text = ' '.join(str(segment['skill_text']).split())
    if not text or 'next action:' in text.lower() or 'subtask:' in text.lower():
        raise ValueError('技能指令为空或包含旧规划分隔符')
    return str(segment['skill_id']) + '|' + text


def completion_label(segment, frame):
    labels = segment.get('termination_labels')
    if labels is not None:
        value = labels.get(str(frame), labels.get(frame))
    elif not segment.get('label_valid', False):
        value = None
    elif frame == segment['end_frame'] and segment.get('postcondition_success') is True:
        value = 1
    elif frame == segment['start_frame'] and segment.get('start_postcondition_success') is False:
        value = 0
    else:
        value = None
    if value is not None and value not in (0, 1):
        raise ValueError('完成标签必须为 0/1/null')
    return value


def action_valid(segment):
    return bool(segment.get('action_train_valid', segment.get('label_valid', False) and
                            segment.get('postcondition_success', False)))


def prepare(data_roots, expected_tasks, output):
    expected_tasks = list(expected_tasks)
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise ValueError('expected_tasks必须是非空且不重复的任务列表')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = {task: defaultdict(Counter) for task in expected_tasks}
    conditions, sources, errors, collisions, missing_roots = {}, [], [], [], []
    selected, seen_paths, split_sources, labelled_frames = {}, set(), defaultdict(set), set()
    unexpected_tasks, empty_attempts = [], []
    for supplied in data_roots:
        root = Path(supplied).resolve()
        if (root / 'data').is_dir():
            root /= 'data'
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        if root == output:
            raise ValueError('输入源不能是本次汇总输出目录')
        for directory in sorted(root.glob('*/seed_*')):
            if not directory.is_dir():
                continue
            task, source = directory.parent.name, directory.resolve()
            if source in seen_paths:
                continue
            seen_paths.add(source)
            if task not in stats:
                unexpected_tasks.append(str(source))
                continue
            key = (task, directory.name)
            if key in selected and selected[key] != source:
                collisions.append({'task': task, 'seed_directory': directory.name,
                                   'sources': [str(selected[key]), str(source)]})
                continue
            selected[key] = source
            sources.append({'task': task, 'source': str(source), 'link': str(output / task / directory.name)})
            segments_path = source / 'segments.json'
            if not segments_path.is_file():
                empty_attempts.append(str(source))
                continue
            try:
                segments = json.loads(segments_path.read_text())
                if not segments:
                    empty_attempts.append(str(source))
                    continue
                rows = [json.loads(line) for line in (source / 'observations.jsonl').read_text().splitlines() if line]
                frames = {int(row['frame_index']): row for row in rows}
                if len(frames) != len(rows):
                    raise ValueError('重复frame_index')
                missing_images = []
                for row in rows:
                    if len(row['state']) != 14 or not all(math.isfinite(x) for x in row['state']):
                        raise ValueError('state不是有限14维数据')
                    missing_images.extend(row['images'][camera] for camera in CAMERAS
                                          if not (source / row['images'][camera]).is_file())
                if missing_images:
                    errors.append({'source': str(source), 'error': '观测图像未收齐',
                                   'missing_image_files': len(missing_images), 'examples': missing_images[:3]})
                for segment in segments:
                    if segment['task'] != task:
                        raise ValueError('segment任务与目录不一致')
                    split = 'val' if segment['split'] == 'validation' else segment['split']
                    if split not in ('train', 'val'):
                        raise ValueError(f'非训练/验证split: {split}')
                    seed = int(segment['training_seed'])
                    split_sources[(task, seed)].add(split)
                    start, end = int(segment['start_frame']), int(segment['end_frame'])
                    if start > end or any(frame not in frames for frame in range(start, end + 1)):
                        raise ValueError(f'技能段观测不连续: {start}:{end}')
                    condition = condition_key(segment)
                    entry = conditions.setdefault(condition, {'tasks': set(), 'train': Counter(), 'val': Counter()})
                    entry['tasks'].add(task)
                    counts = Counter(segments=1, frames=end - start + 1,
                                     action_valid_segments=int(action_valid(segment) and end > start),
                                     action_valid_frames=(end - start) if action_valid(segment) else 0,
                                     failed_segments=int(segment.get('label_valid') is True and
                                                         segment.get('postcondition_success') is False),
                                     unknown_segments=int(not segment.get('label_valid', False)))
                    for frame in range(start, end + 1):
                        label = completion_label(segment, frame)
                        counts['unknown_labels' if label is None else f'label_{label}'] += 1
                        if label is not None:
                            labelled_frames.add((str(source), frame, split))
                    stats[task][split].update(counts)
                    entry[split].update(counts)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({'source': str(source), 'error': str(error)})
    for (task, name), source in selected.items():
        target = output / task / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                collisions.append({'link': str(target), 'requested_source': str(source),
                                   'error': '已有路径不属于同一源，未覆盖'})
        else:
            target.symlink_to(source, target_is_directory=True)
    for existing in output.glob('*/seed_*'):
        if (existing.parent.name, existing.name) not in selected:
            collisions.append({'link': str(existing), 'error': '输出含本次源清单以外的数据，未删除'})
    task_gaps = []
    for task in expected_tasks:
        for split in ('train', 'val'):
            absent = [key for key in ('action_valid_segments', 'label_0', 'label_1') if not stats[task][split][key]]
            if absent:
                task_gaps.append({'task': task, 'split': split, 'missing': absent})
    vocabulary_gaps, validation_unknown, condition_diagnostics = [], [], []
    for condition, entry in conditions.items():
        absent = [split for split in ('train', 'val') if not entry[split]['segments']]
        if absent:
            vocabulary_gaps.append({'condition': condition, 'missing_splits': absent,
                                   'blocks_training': 'train' in absent})
        if 'train' in absent:
            validation_unknown.append(condition)
        for split in ('train', 'val'):
            missing = [key for key in ('action_valid_segments', 'label_0', 'label_1') if not entry[split][key]]
            if missing:
                condition_diagnostics.append({'condition': condition, 'split': split, 'missing': missing})
        entry['tasks'] = sorted(entry['tasks'])
    split_overlap = [{'task': task, 'training_seed': seed} for (task, seed), splits in split_sources.items()
                     if len(splits) > 1]
    train_conditions = sorted(key for key, value in conditions.items() if value['train']['segments'])
    ready = not (missing_roots or errors or collisions or task_gaps or validation_unknown or split_overlap)
    report = {'status': 'ready' if ready else 'incomplete', 'data_ready_for_training': ready,
              'expected_tasks': expected_tasks, 'source_roots': [str(Path(p).resolve()) for p in data_roots],
              'source_attempts': sources, 'empty_or_precollection_attempts': empty_attempts,
              'missing_roots': missing_roots, 'unexpected_task_sources_not_linked': unexpected_tasks,
              'source_errors': errors, 'collisions': collisions, 'task_coverage_gaps': task_gaps,
              'condition_split_gaps': vocabulary_gaps, 'condition_label_diagnostics': condition_diagnostics,
              'validation_unknown_conditions': validation_unknown,
              'train_conditions_without_validation': [row['condition'] for row in vocabulary_gaps
                                                       if 'val' in row['missing_splits']],
              'train_val_seed_overlap': split_overlap, 'tasks': stats, 'conditions': conditions,
              'condition_to_index': {key: i for i, key in enumerate(train_conditions)},
              'completion_cache_uint8_bytes_estimate': len(labelled_frames) * 3 * 3 * 128 * 128,
              'cache_note': '仅估计现有完成头的标注图缓存；torch.stack峰值及其他对象另计。',
              'coverage_note': '段与标签统计来自原始元数据；存在源文件错误时不代表这些样本已可用于训练。',
              'preservation_note': '源数据只读；失败段、unknown标签和未完成attempt全部保留，不重新分配split。'}
    (output / 'DATA_COVERAGE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--expected-tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data_roots, json.loads(args.expected_tasks.read_text()), args.output)
    print(json.dumps({'status': report['status'], 'attempts': len(report['source_attempts']),
                      'conditions': len(report['conditions']), 'report': str(args.output / 'DATA_COVERAGE.json')}))
    return 0 if report['data_ready_for_training'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
