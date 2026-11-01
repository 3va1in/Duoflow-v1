"""从专家分段构造技能动作与显式完成监督，保留原动作归一化。"""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


@lru_cache(maxsize=None)
def legacy_module(policy_root, name):
    import sys
    path = Path(policy_root).resolve() / (name + '.py')
    key = '_atomic_legacy_' + name + '_' + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def skill_instruction(text):
    text = ' '.join(str(text).split())
    if not text or 'next action:' in text.lower() or 'subtask:' in text.lower():
        raise ValueError('技能输入必须是非空独立指令，不能包含旧规划分隔符')
    return text


def condition_key(skill_id, text):
    return str(skill_id) + '|' + skill_instruction(text)


def load_norm(path):
    payload = json.loads(Path(path).read_text())
    stats = payload.get('norm_stats', payload)
    return {key: stats[key] for key in ('observation.state', 'action')}


def completion_images(images):
    return torch.stack([torch.from_numpy(np.asarray(
        Image.fromarray(np.asarray(images[key], dtype=np.uint8)).resize((128, 128)),
        dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0 for key in CAMERAS])


def completion_label(segment, frame):
    labels = segment.get('termination_labels')
    if labels is not None:
        value = labels.get(str(frame), labels.get(frame))
    elif not segment.get('label_valid', False):
        return None
    elif frame == segment['end_frame'] and segment.get('postcondition_success') is True:
        value = 1
    elif frame == segment['start_frame'] and segment.get('start_postcondition_success') is False:
        value = 0
    else:
        value = None
    if value is not None and value not in (0, 1):
        raise ValueError('termination_labels 只能包含 0/1/null')
    return value


def bounded_targets(rows, index, segment, horizon=50):
    """末帧是技能终止观测，不把其后的动作归到本技能。"""
    end = int(segment['end_frame'])
    source = segment.get('action_source', 'next_state')
    if source == 'recorded_action':
        eligible = [row for row in rows[index:index + horizon]
                    if int(row['frame_index']) < end]
        values = [row['action'] for row in eligible]
    elif source == 'next_state':
        eligible = [row for row in rows[index + 1:index + 1 + horizon]
                    if int(row['frame_index']) <= end]
        values = [row['state'] for row in eligible]
    else:
        raise ValueError(f'未知动作监督来源: {source}')
    valid_action = segment.get('action_train_valid',
                               segment.get('label_valid', False) and
                               segment.get('postcondition_success', False))
    mask = [bool(valid_action)] * len(values)
    fill = values[-1] if values else rows[index]['state']
    values += [fill] * (horizon - len(values))
    mask += [False] * (horizon - len(mask))
    return values, mask


class SkillDataset(Dataset):
    def __init__(self, data_root, split, policy_root, norm_stats, tokenizer,
                 image_transform, condition_to_index=None):
        self.legacy = legacy_module(str(Path(policy_root).resolve()), 'dataset')
        self.norm_stats = norm_stats
        self.norm = self.legacy._QuantileNorm(norm_stats['observation.state'], norm_stats['action'])
        self.tokenizer, self.image_transform = tokenizer, image_transform
        self.samples, self.segments, self.sources = [], [], set()
        for path in sorted(Path(data_root).glob('*/seed_*/segments.json')):
            segments = json.loads(path.read_text())
            accepted_splits = {'val', 'validation'} if split in {'val', 'validation'} else {split}
            selected = [s for s in segments if s['split'] in accepted_splits]
            if not selected:
                continue
            rows = [json.loads(line) for line in (path.parent / 'observations.jsonl').read_text().splitlines() if line]
            frame_to_index = {int(row['frame_index']): i for i, row in enumerate(rows)}
            if len(frame_to_index) != len(rows):
                raise ValueError(f'重复 frame_index: {path}')
            for segment in selected:
                segment = dict(segment, skill_text=skill_instruction(segment['skill_text']))
                start, end = int(segment['start_frame']), int(segment['end_frame'])
                if start > end or any(frame not in frame_to_index for frame in range(start, end + 1)):
                    raise ValueError(f'技能段缺少连续观测: {path} {start}:{end}')
                self.sources.add((segment['task'], int(segment['training_seed'])))
                self.segments.append(segment)
                for frame in range(start, end + 1):
                    self.samples.append((path.parent, rows, frame_to_index[frame], segment))
        ids = sorted({s['skill_id'] for s in self.segments})
        self.skill_to_index = {s: i for i, s in enumerate(ids)}
        conditions = sorted({condition_key(s['skill_id'], s['skill_text']) for s in self.segments})
        self.condition_to_index = (dict(condition_to_index) if condition_to_index is not None
                                   else {s: i for i, s in enumerate(conditions)})
        if not self.samples or any(s not in self.condition_to_index for s in conditions):
            raise ValueError(f'{split} 无样本或出现训练词表外技能绑定条件')
        self.label_counts = {label: sum(completion_label(s, rows[i]['frame_index']) == label
                                       for _, rows, i, s in self.samples) for label in (0, 1)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        directory, rows, row_index, segment = self.samples[index]
        row = rows[row_index]
        raw_state = self.legacy._as_vector(row['state'], name='state')
        targets, mask = bounded_targets(rows, row_index, segment)
        actions = np.asarray(targets, dtype=np.float32)
        delta = np.asarray(self.legacy.DEFAULT_DELTA_MASK, dtype=bool)
        actions[:, delta] -= raw_state[delta]
        state14 = self.norm.apply(torch.as_tensor(raw_state), kind='state')
        action14 = self.norm.apply(torch.as_tensor(actions), kind='action')
        state32, actions32 = torch.zeros(32), torch.zeros(50, 32)
        state32[:14], actions32[:, :14] = state14, action14
        raw_images = {key: np.asarray(Image.open(directory / row['images'][key]).convert('RGB')) for key in CAMERAS}
        images, img_masks, image_params = self.image_transform(
            {'observation.images.' + key: torch.from_numpy(value.copy()).permute(2, 0, 1).contiguous().float() / 255.0
             for key, value in raw_images.items()})
        text = skill_instruction(segment['skill_text'])
        tokens = self.legacy._normalize_tokenizer_output(self.tokenizer(
            {'task': text, 'observation.state': state14, 'action': action14,
             'embodiment_id': 0, 'robot_type': 'agilex_cobot_magic'}))
        label = completion_label(segment, int(row['frame_index']))
        sample = dict(images=list(images), img_masks=list(img_masks), image_transform_params=image_params,
                      emb_ids=torch.tensor(0), proprioception=state32.unsqueeze(0),
                      agent_pos_mask=torch.cat((torch.ones(14), torch.zeros(18))).unsqueeze(0),
                      proprioception_present=torch.tensor(True), actions32=actions32,
                      action_valid_mask=torch.tensor(mask),
                      stableexampleindex=f"{segment['task']}:{segment['training_seed']}:{segment['skill_id']}:{row['frame_index']}",
                      task=text, instruction=text, task_index=torch.tensor(self.skill_to_index[segment['skill_id']]),
                      episode_index=torch.tensor(segment['training_seed']), frame_index=torch.tensor(row['frame_index']),
                      completion_images=completion_images(raw_images), completion_state=state14,
                      condition_index=torch.tensor(self.condition_to_index[condition_key(segment['skill_id'], text)]),
                      completion_target=torch.tensor(float(label or 0)), completion_valid=torch.tensor(label is not None))
        sample.update(tokens)
        return sample

    def collate(self, samples):
        result = self.legacy.collate_official(samples)
        for key in ('completion_images', 'completion_state', 'condition_index', 'completion_target', 'completion_valid'):
            result[key] = torch.stack([s[key] for s in samples])
        return result
