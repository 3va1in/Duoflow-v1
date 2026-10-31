"""固定动作策略，只用既有训练轨迹重训和校准技能完成头。"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from skill_dataset import CAMERAS, SkillDataset, completion_images, completion_label, condition_key
from skill_model import CompletionHead


THRESHOLDS = tuple(round(i / 100, 2) for i in range(10, 96, 5)) + (0.98, 0.99)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def group_weights(conditions, labels):
    counts = Counter((int(c), int(y)) for c, y in zip(conditions, labels))
    if not counts or set(int(y) for y in labels) != {0, 1}:
        raise ValueError('完成头训练需要已标注的正例和负例')
    total, groups = len(labels), len(counts)
    weights = torch.tensor([total / (groups * counts[(int(c), int(y))])
                            for c, y in zip(conditions, labels)], dtype=torch.float32)
    return weights, [{'condition_index': c, 'label': y, 'count': count,
                      'loss_weight': total / (groups * count)}
                     for (c, y), count in sorted(counts.items())]


class CompletionDataset(Dataset):
    def __init__(self, root, split, policy_root, stats, condition_to_index):
        # 只复用段索引和归一化，不调用原动作数据集的图像增强、tokenizer或动作目标。
        source = SkillDataset(root, split, policy_root, stats, None, None, condition_to_index)
        self.sources, self.metadata = set(source.sources), []
        states, labels, conditions, image_ids = [], [], [], []
        image_cache, images = {}, []
        for directory, rows, index, segment in source.samples:
            row = rows[index]
            label = completion_label(segment, int(row['frame_index']))
            if label is None:
                continue
            image_key = (str(directory), int(row['frame_index']))
            if image_key not in image_cache:
                raw = {key: np.asarray(Image.open(directory / row['images'][key]).convert('RGB'))
                       for key in CAMERAS}
                # 使用与部署相同的 resize；uint8缓存不改变其像素值。
                resized = ((completion_images(raw) + 1) * 127.5).round().to(torch.uint8)
                image_cache[image_key] = len(images)
                images.append(resized)
            image_ids.append(image_cache[image_key])
            state = torch.as_tensor(row['state'], dtype=torch.float32)
            if state.shape != (14,) or not bool(torch.isfinite(state).all()):
                raise ValueError('完成头状态必须为有限14维数据')
            states.append(source.norm.apply(state, kind='state'))
            labels.append(int(label))
            conditions.append(condition_to_index[condition_key(segment['skill_id'], segment['skill_text'])])
            self.metadata.append({'task': segment['task'], 'training_seed': int(segment['training_seed']),
                'skill_id': segment['skill_id'], 'skill_instance_id': segment['skill_instance_id'],
                'condition_index': conditions[-1], 'frame_index': int(row['frame_index']),
                'segment_start_frame': int(segment['start_frame']),
                'segment_end_frame': int(segment['end_frame']), 'label': int(label)})
        if not labels:
            raise ValueError(f'{split}没有显式标注的完成监督')
        self.images = torch.stack(images)
        self.image_ids = torch.tensor(image_ids, dtype=torch.long)
        self.states = torch.stack(states)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.conditions = torch.tensor(conditions, dtype=torch.long)
        self.weights = torch.ones(len(labels))
        self.label_counts = dict(Counter(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.images[self.image_ids[index]], self.states[index], self.conditions[index],
                self.labels[index], self.weights[index])


def sequence_metrics(probabilities, metadata, threshold):
    groups = defaultdict(list)
    for probability, row in zip(probabilities, metadata):
        key = (row['task'], row['training_seed'], row['skill_instance_id'])
        groups[key].append((row['frame_index'], float(probability), row['label']))
    correct = false = no_stop = eligible = 0
    for rows in groups.values():
        rows.sort()
        eligible += int(any(row[2] == 1 for row in rows))
        previous = None
        for frame, probability, label in rows:
            if (previous is not None and frame == previous[0] + 1 and
                    previous[1] >= threshold and probability >= threshold):
                correct += int(label == 1)
                false += int(label == 0)
                break
            previous = (frame, probability)
        else:
            no_stop += 1
    return {'segments': len(groups), 'segments_with_positive_frame': eligible,
            'first_two_frame_stop_correct': correct, 'first_two_frame_stop_false': false,
            'no_two_frame_stop': no_stop,
            'scope': 'expert_sequence_diagnostic_not_policy_stop_or_skill_SR'}


def threshold_metrics(probabilities, labels, threshold):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    high, positive = probabilities >= threshold, labels == 1
    tp, fp = int((high & positive).sum()), int((high & ~positive).sum())
    positives, negatives = int(positive.sum()), int((~positive).sum())
    fn = positives - tp
    denominator = 1.25 * tp + 0.25 * fn + fp
    return {'threshold': float(threshold), 'true_positive': tp, 'false_positive': fp,
            'positive_frames': positives, 'negative_frames': negatives,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / positives if positives else 0.0,
            'false_positive_rate': fp / negatives if negatives else 0.0,
            'f0_5': 1.25 * tp / denominator if denominator else 0.0}


def selection_key(metrics):
    return (metrics['f0_5'], -metrics['false_positive_rate'], metrics['threshold'])


def select_threshold(probabilities, labels, metadata=None):
    metrics = [threshold_metrics(probabilities, labels, threshold) for threshold in THRESHOLDS]
    if metadata is not None:
        for row in metrics:
            row['sequence_diagnostic'] = sequence_metrics(probabilities, metadata, row['threshold'])
    return max(metrics, key=selection_key), metrics


@torch.inference_mode()
def validate(head, loader, device):
    head.eval()
    probabilities, labels = [], []
    for images, state, condition, target, _ in loader:
        logits = head(images.to(device).float() / 127.5 - 1.0,
                      state.to(device), condition.to(device))
        probabilities.extend(logits.sigmoid().cpu().tolist())
        labels.extend(target.tolist())
    if not np.isfinite(probabilities).all():
        raise RuntimeError('验证完成概率非有限值')
    return probabilities, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('policy-root', 'data-root', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(epochs=24, batch_size=64, learning_rate=3e-4, weight_decay=1e-4, seed=2731,
                  actor_loaded=False, actor_updated=False, head_architecture_unchanged=True,
                  sample_schedule='shuffle_full_dataset_without_replacement_every_epoch',
                  weighting='inverse_nonempty_condition_label_group_mean_one_no_pos_weight',
                  selection='validation_max_f0.5_then_lower_FPR_then_higher_threshold_then_earlier_epoch',
                  thresholds=THRESHOLDS, held_out_data_used=False)
    with (args.output / 'TRAIN_CLAIM.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'started_at': time.time(), 'config': config}, stream, indent=2)
    if (args.output / 'completion_checkpoint.pt').exists():
        raise FileExistsError('拒绝覆盖已有完成头checkpoint')
    started = time.time()
    status = {'status': 'loading_training_data', 'pid': os.getpid(), 'started_at': started}
    write_json(args.output / 'STATUS.json', status)
    try:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.set_num_threads(4)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if payload.get('format') != 'atomic_skill_lora_v1':
            raise ValueError('输入必须为原技能checkpoint')
        mapping, stats = payload['condition_to_index'], payload['norm_stats']
        head_state = payload['completion_state']
        del payload
        train = CompletionDataset(args.data_root, 'train', args.policy_root, stats, mapping)
        val = CompletionDataset(args.data_root, 'val', args.policy_root, stats, mapping)
        if train.sources & val.sources:
            raise ValueError('训练和验证包含同任务同seed')
        train.weights, counts = group_weights(train.conditions, train.labels)
        config.update(train_samples=len(train), validation_samples=len(val),
                      train_label_counts=train.label_counts, validation_label_counts=val.label_counts,
                      train_sources=sorted(train.sources), validation_sources=sorted(val.sources),
                      condition_to_index=mapping, condition_label_groups=counts)
        write_json(args.output / 'TRAIN_CONFIG.json', config)
        device = torch.device(args.device)
        head = CompletionHead(len(mapping)).to(device)
        head.load_state_dict(head_state, strict=True)
        del head_state
        optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])
        loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        validation = DataLoader(val, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        probabilities, labels = validate(head, validation, device)
        baseline_selected, baseline_grid = select_threshold(probabilities, labels, val.metadata)
        write_json(args.output / 'BASELINE_VALIDATION.json', {
            'original_deployed_threshold': threshold_metrics(probabilities, labels, 0.9),
            'calibrated_diagnostic_only': baseline_selected, 'threshold_grid': baseline_grid})
        best, best_epoch, updates = None, None, 0
        for epoch in range(1, config['epochs'] + 1):
            head.train()
            total_loss, sample_count = 0.0, 0
            for images, state, condition, target, weight in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = head(images.to(device).float() / 127.5 - 1.0,
                              state.to(device), condition.to(device))
                loss = (F.binary_cross_entropy_with_logits(logits, target.to(device), reduction='none') *
                        weight.to(device)).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('完成头训练loss非有限值')
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                sample_count += len(target)
                updates += 1
            probabilities, labels = validate(head, validation, device)
            selected, grid = select_threshold(probabilities, labels, val.metadata)
            with (args.output / f'validation_epoch_{epoch:02d}.jsonl').open('x') as stream:
                for probability, row in zip(probabilities, val.metadata):
                    stream.write(json.dumps({**row, 'probability': probability}) + '\n')
            row = {'epoch': epoch, 'updates': updates, 'samples_consumed': epoch * len(train),
                   'weighted_training_bce': total_loss / sample_count,
                   'validation_selected': selected, 'validation_threshold_grid': grid,
                   'elapsed_seconds': time.time() - started}
            with (args.output / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if best is None or selection_key(selected) > selection_key(best):
                best, best_epoch = selected, epoch
                checkpoint = {'format': 'atomic_skill_completion_repair_v1',
                    'completion_state': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    'condition_to_index': mapping, 'norm_stats': stats,
                    'stop_config': {'threshold': best['threshold'], 'consecutive_observations': 2},
                    'metadata': {'source_actor_checkpoint': str(args.checkpoint.resolve()),
                                 'actor_updated': False, 'selected_epoch': epoch,
                                 'validation_selection': best, 'config': config}}
                temporary = args.output / 'completion_checkpoint.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(args.output / 'completion_checkpoint.pt')
            status.update(status='training', epoch=epoch, updates=updates, selected_epoch=best_epoch,
                          validation_selected=best, elapsed_seconds=time.time() - started)
            write_json(args.output / 'STATUS.json', status)
            print(json.dumps({'epoch': epoch, 'loss': row['weighted_training_bce'],
                              'selected_epoch': best_epoch, 'validation': best}), flush=True)
        status.update(status='complete', ended_at=time.time(), run_complete=True,
                      checkpoint=str(args.output / 'completion_checkpoint.pt'),
                      samples_consumed=config['epochs'] * len(train), skill_execution_sr=None)
        write_json(args.output / 'TRAIN_RESULT.json', status)
    except BaseException as error:
        status.update(status='needs_attention', ended_at=time.time(), error=repr(error),
                      traceback=traceback.format_exc(), run_complete=False)
        write_json(args.output / 'TRAIN_RESULT.json', status)
        raise
    finally:
        write_json(args.output / 'STATUS.json', status)


if __name__ == '__main__':
    main()
