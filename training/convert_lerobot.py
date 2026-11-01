"""将本地 LeRobot v2.1/v3 数据及人工技能标注转换为 DuoFlow 训练目录。"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} 必须是整数: {value!r}')
    return int(value)


def local_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'数据文件必须位于数据集目录内: {name}')
    return path


class LocalLeRobot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = read_json(self.root / 'meta/info.json')
        self.version = self.info['codebase_version']
        if self.version not in ('v2.1', 'v3.0'):
            raise ValueError(f'支持 codebase_version v2.1/v3.0，实际为 {self.version}')
        self.fps = float(self.info['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('fps 必须为有限正数')
        if self.version == 'v2.1':
            records = [json.loads(line) for line in (self.root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        else:
            records = []
            for path in sorted((self.root / 'meta/episodes').rglob('*.parquet')):
                records.extend(pq.read_table(path).to_pylist())
        self.episodes = {}
        for row in records:
            index = integer(row['episode_index'], 'episode_index')
            if index in self.episodes:
                raise ValueError(f'重复 episode 元数据: {index}')
            self.episodes[index] = row
        if not self.episodes:
            raise ValueError('未找到 episode 元数据')
        paths = sorted((self.root / 'data').rglob('*.parquet'))
        if not paths:
            raise ValueError('未找到 data/**/*.parquet')
        self.data = ds.dataset([str(p) for p in paths], format='parquet')

    def rows(self, episode, mapping):
        keys = [mapping['state']['key'], mapping['action']['key'], 'frame_index', 'episode_index', 'timestamp']
        for key in mapping['camera_map'].values():
            if self.info['features'][key]['dtype'] == 'image':
                keys.append(key)
        rows = self.data.to_table(columns=list(dict.fromkeys(keys)),
                                  filter=ds.field('episode_index') == episode).to_pylist()
        rows.sort(key=lambda row: row['frame_index'])
        if not rows or len(rows) != self.episodes[episode]['length']:
            raise ValueError(f'episode {episode} 的记录数与元数据 length 不一致')
        if [integer(r['frame_index'], 'frame_index') for r in rows] != list(range(len(rows))):
            raise ValueError(f'episode {episode} 的 frame_index 必须从 0 连续且无重复')
        timestamps = np.asarray([r['timestamp'] for r in rows], dtype=float)
        if not np.isfinite(timestamps).all() or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f'episode {episode} 的 timestamp 非有限、负数或不递增')
        return rows

    def video(self, episode, key):
        if self.version == 'v2.1':
            path = self.info['video_path'].format(episode_chunk=episode // self.info['chunks_size'],
                                                  episode_index=episode, video_key=key)
            return local_path(self.root, path), 0.0, None
        meta = self.episodes[episode]
        prefix = f'videos/{key}/'
        path = self.info['video_path'].format(video_key=key, chunk_index=meta[prefix + 'chunk_index'],
                                              file_index=meta[prefix + 'file_index'])
        start, end = float(meta[prefix + 'from_timestamp']), float(meta[prefix + 'to_timestamp'])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f'episode {episode} 的 {key} 视频时间范围无效')
        return local_path(self.root, path), start, end


def mapped_vector(row, spec):
    raw = np.asarray(row[spec['key']], dtype=np.float64)
    indices = spec.get('indices', list(range(14)))
    if (raw.ndim != 1 or len(indices) != 14 or len(set(indices)) != 14
            or any(integer(i, 'indices') < 0 or i >= len(raw) for i in indices)):
        raise ValueError(f'{spec["key"]} 必须可映射到 14 个不同的状态/动作维度')
    scale = np.asarray(spec.get('scale', [1.0] * 14), dtype=float)
    offset = np.asarray(spec.get('offset', [0.0] * 14), dtype=float)
    if scale.shape != (14,) or offset.shape != (14,):
        raise ValueError('scale/offset 必须各有 14 个数')
    result = raw[indices] * scale + offset
    if not np.isfinite(result).all():
        raise ValueError(f'{spec["key"]} 映射后包含 NaN/Inf')
    return result.tolist()


def normalise_segments(episode, count):
    task, split = episode['task'], episode['split']
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('task 请使用字母、数字、下划线或连字符组成的目录名')
    if split not in ('train', 'val'):
        raise ValueError('每个 episode 必须明确指定 train 或 val，不能按帧拆分')
    result = []
    for index, original in enumerate(episode['segments']):
        segment = dict(original)
        start, end = (integer(segment[key], key) for key in ('start_frame', 'end_frame'))
        if not 0 <= start <= end < count:
            raise ValueError(f'技能段超出 episode 帧范围: {start}:{end}, length={count}')
        if not isinstance(segment['skill_text'], str):
            raise ValueError('skill_text 必须为非空字符串')
        text = ' '.join(segment['skill_text'].split())
        if (not text or not isinstance(segment['skill_id'], str) or not segment['skill_id']
                or 'next action:' in text.lower() or 'subtask:' in text.lower()):
            raise ValueError('请提供非空 skill_id 和独立 skill_text')
        if type(segment.get('action_train_valid')) is not bool:
            raise ValueError('每段必须用 action_train_valid 明确动作是否可供模仿')
        labels = segment.pop('termination_labels', {})
        spans = segment.pop('completion_spans', [])
        if labels and spans:
            raise ValueError('termination_labels 和 completion_spans 请选择一种')
        expanded = {}
        for span in spans:
            first, last = integer(span['start_frame'], 'start_frame'), integer(span['end_frame'], 'end_frame')
            if not start <= first <= last <= end:
                raise ValueError('完成标签区间必须位于本技能段内')
            for frame in range(first, last + 1):
                if str(frame) in expanded and expanded[str(frame)] != span['label']:
                    raise ValueError('同一帧存在矛盾完成标签')
                expanded[str(frame)] = span['label']
        for frame, label in (labels or expanded).items():
            if str(int(frame)) != str(frame) or not start <= int(frame) <= end:
                raise ValueError(f'完成标签帧超出技能范围: {frame}')
            if label is not None and (type(label) is not int or label not in (0, 1)):
                raise ValueError('完成标签只能为整数 0、1 或 null')
        labels = {str(k): v for k, v in (labels or expanded).items()}
        segment.update(task=task, split=split, training_seed=episode['episode_index'], skill_text=text,
                       skill_instance_id=f'{episode["episode_index"]}:{index}', action_source='recorded_action',
                       termination_labels=labels, label_valid=any(v is not None for v in labels.values()),
                       start_postcondition_success=None if labels.get(str(start)) is None else bool(labels[str(start)]),
                       postcondition_success=None if labels.get(str(end)) is None else bool(labels[str(end)]))
        result.append(segment)
    if not result:
        raise ValueError('episode 缺少技能分段；模板中的空 segments 必须先标注')
    return result


def image_from_value(value, root):
    if isinstance(value, dict):
        data = value.get('bytes')
        source = io.BytesIO(data) if data else local_path(root, value['path'])
    elif isinstance(value, bytes):
        source = io.BytesIO(value)
    elif isinstance(value, str):
        source = local_path(root, value)
    else:
        raise ValueError('image 特征必须包含图像 bytes 或本地相对路径')
    with Image.open(source) as image:
        return image.convert('RGB')


def video_frames(path, timestamps, tolerance):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((timestamps[0] - tolerance) / stream.time_base)),
                       stream=stream, backward=True)
        decoder = iter(container.decode(stream))

        def next_frame():
            frame = next(decoder, None)
            if frame is None:
                return None
            if frame.pts is None:
                raise ValueError(f'视频帧缺少 PTS: {path}')
            return float(frame.pts * frame.time_base), frame

        previous, current = None, next_frame()
        for target in timestamps:
            while current is not None and current[0] < target:
                previous, current = current, next_frame()
            choices = [f for f in (previous, current) if f is not None]
            if not choices:
                raise ValueError(f'视频无可用帧: {path}')
            stamp, frame = min(choices, key=lambda item: abs(item[0] - target))
            if abs(stamp - target) > tolerance:
                raise ValueError(f'视频对齐失败: {path.name}, target={target}, PTS={stamp}')
            yield frame.to_image().convert('RGB'), stamp


def convert(dataset, annotations, output):
    source = LocalLeRobot(dataset)
    config = read_json(annotations)
    if config.get('format') != 'duoflow_lerobot_annotations_v1':
        raise ValueError('标注文件 format 必须为 duoflow_lerobot_annotations_v1')
    if config['action'].get('mode') != 'absolute_joint' or config['action'].get('alignment') != 'same_row':
        raise ValueError('须确认 action 为与同一行观测对齐的绝对关节/夹爪目标；不接受速度、末端位姿或增量命令')
    cameras = config['camera_map']
    if set(cameras) != set(CAMERAS) or len(set(cameras.values())) != 3:
        raise ValueError('camera_map 必须明确映射全局、左腕、右腕三个不同视角')
    for key in cameras.values():
        if key not in source.info['features'] or source.info['features'][key]['dtype'] not in ('video', 'image'):
            raise ValueError(f'未找到 image/video 相机特征: {key}')
    entries, used, vocabulary = [], set(), {'train': set(), 'val': set()}
    for episode in config['episodes']:
        index = integer(episode['episode_index'], 'episode_index')
        if index in used or index not in source.episodes:
            raise ValueError(f'episode 重复或不存在: {index}')
        used.add(index)
        segments = normalise_segments(episode, source.episodes[index]['length'])
        vocabulary[episode['split']].update(s['skill_id'] + '|' + s['skill_text'] for s in segments)
        entries.append((episode, segments))
    if not entries or vocabulary['val'] - vocabulary['train']:
        raise ValueError('没有待转换 episode，或验证集包含训练集未出现的技能条件')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'输出已存在，未覆盖: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, reports = {'train': Counter(), 'val': Counter()}, []
    with tempfile.TemporaryDirectory(prefix=output.name + '.partial-', dir=output.parent) as temporary:
        destination = Path(temporary) / 'data'
        destination.mkdir()
        for episode, segments in entries:
            index, split = episode['episode_index'], episode['split']
            raw = source.rows(index, config)
            directory = destination / episode['task'] / f'seed_{index:06d}'
            directory.mkdir(parents=True)
            rows = [{'frame_index': r['frame_index'], 'timestamp': float(r['timestamp']),
                     'source_episode_index': index, 'state': mapped_vector(r, config['state']),
                     'action': mapped_vector(r, config['action']), 'images': {}, 'image_timestamps': {}}
                    for r in raw]
            for camera, key in cameras.items():
                (directory / 'images' / camera).mkdir(parents=True)
                if source.info['features'][key]['dtype'] == 'video':
                    path, start, end = source.video(index, key)
                    stamps = [start + r['timestamp'] for r in rows]
                    if end is not None and stamps[-1] >= end + 1e-4:
                        raise ValueError(f'episode {index} 视频查询超出本回合范围')
                    images = video_frames(path, stamps, 0.5 / source.fps + 1e-4)
                else:
                    images = ((image_from_value(r[key], source.root), r['timestamp']) for r in raw)
                for row, (image, stamp) in zip(rows, images):
                    name = f'images/{camera}/{row["frame_index"]:06d}.png'
                    image.save(directory / name)
                    row['images'][camera] = name
                    row['image_timestamps'][camera] = stamp
            with (directory / 'observations.jsonl').open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            write_json(directory / 'segments.json', segments)
            counts[split].update(episodes=1, frames=len(rows), segments=len(segments))
            for s in segments:
                counts[split]['action_frames'] += (s['end_frame'] - s['start_frame']) if s['action_train_valid'] else 0
                counts[split].update(f'completion_{label}' for label in s['termination_labels'].values() if label is not None)
            jitter = max((abs((b['timestamp'] - a['timestamp']) - 1 / source.fps)
                          for a, b in zip(rows, rows[1:])), default=0.0)
            reports.append({'episode_index': index, 'task': episode['task'], 'split': split,
                            'frames': len(rows), 'max_timestamp_interval_error_s': jitter})
            print(f'episode {index}: {len(rows)} frames, {len(segments)} segments', flush=True)
        report = {'format': 'duoflow_lerobot_conversion_v1', 'source_version': source.version,
                  'source_fps': source.fps, 'resampled': False, 'action_source': 'recorded_action',
                  'mapping': {k: config[k] for k in ('camera_map', 'state', 'action')},
                  'counts': counts, 'episodes': reports,
                  'unselected_episode_indices': sorted(set(source.episodes) - used),
                  'completion_labels_inferred': False, 'normalization_computed': False}
        write_json(destination / 'CONVERSION_REPORT.json', report)
        destination.rename(output)
    return report


def make_template(dataset, output):
    source = LocalLeRobot(dataset)
    features = source.info['features']
    cameras = {c: ('observation.images.' + c if 'observation.images.' + c in features else None) for c in CAMERAS}
    template = {'format': 'duoflow_lerobot_annotations_v1', 'camera_map': cameras,
                'state': {'key': 'observation.state', 'indices': list(range(14))},
                'action': {'key': 'action', 'indices': list(range(14)), 'mode': None, 'alignment': None},
                'source_features': features,
                'episodes': [{'episode_index': index, 'task': None, 'split': None, 'segments': []}
                             for index in sorted(source.episodes)]}
    with Path(output).open('x') as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--annotations', type=Path)
    group.add_argument('--init-annotations', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.init_annotations:
        make_template(args.dataset, args.init_annotations)
        print('待填写标注模板: ' + str(args.init_annotations))
    elif args.output:
        report = convert(args.dataset, args.annotations, args.output)
        print(json.dumps(report['counts'], ensure_ascii=False))
    else:
        parser.error('转换需要 --output')


if __name__ == '__main__':
    main()
