"""独立技能动作微调；不修改原 DuoFlow 训练或评测目录。"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from skill_dataset import SkillDataset, condition_key, legacy_module, load_norm
from skill_model import AtomicSkillModel


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(x, device) for x in value]
    if isinstance(value, dict):
        return {key: to_device(x, device) for key, x in value.items()}
    return value


def balanced_action_indices(dataset, sample_count, seed):
    """任务、条件逐轮均衡；同条件有效帧洗牌轮巡，不用验证数据安排采样。"""
    groups, available = defaultdict(lambda: defaultdict(list)), Counter()
    for index, (_, rows, row_index, segment) in enumerate(dataset.samples):
        task = segment['task']
        condition = dataset.condition_to_index[condition_key(segment['skill_id'], segment['skill_text'])]
        available[(task, condition)] += 0
        valid = segment.get('action_train_valid', segment.get('label_valid', False) and
                            segment.get('postcondition_success', False))
        if (valid and row_index + 1 < len(rows) and
                int(rows[row_index + 1]['frame_index']) <= int(segment['end_frame'])):
            groups[task][condition].append(index)
            available[(task, condition)] += 1
    if not groups:
        raise ValueError('没有有效动作训练帧，不能执行技能动作微调')
    rng = random.Random(seed)
    task_deck, condition_decks, frame_decks, indices = [], {}, {}, []
    for _ in range(sample_count):
        if not task_deck:
            task_deck = sorted(groups)
            rng.shuffle(task_deck)
        task = task_deck.pop()
        if not condition_decks.get(task):
            condition_decks[task] = sorted(groups[task])
            rng.shuffle(condition_decks[task])
        condition = condition_decks[task].pop()
        key = (task, condition)
        if not frame_decks.get(key):
            frame_decks[key] = list(groups[task][condition])
            rng.shuffle(frame_decks[key])
        indices.append(frame_decks[key].pop())
    return indices, available


def coverage_snapshot(available, visits, targets, unique, mapping, update):
    names = {index: condition for condition, index in mapping.items()}
    rows = [{'task': task, 'condition_index': index, 'condition': names[index],
             'available_action_samples': count, 'consumed_action_samples': visits[(task, index)],
             'unique_action_samples_consumed': len(unique[(task, index)]),
             'valid_target_action_frames': targets[(task, index)]}
            for (task, index), count in sorted(available.items())]
    return {'optimizer_updates': update, 'samples_consumed': sum(visits.values()),
            'task_conditions_with_valid_action_data': sum(count > 0 for count in available.values()),
            'task_conditions_consumed': sum(count > 0 for count in visits.values()),
            'unconsumed_task_conditions': [row for row in rows if row['consumed_action_samples'] == 0],
            'by_task_condition': rows,
            'scope': '实际optimizer.step所用有效动作样本，不代表技能执行成功率'}


def action_loss(actor, batch, device):
    valid = batch['action_valid_mask']
    if not bool(valid.any()):
        return None
    actions = batch['actions32'].float()
    noise = torch.randn_like(actions)
    noise[..., 14:] = 0
    timestep = torch.rand(actions.shape[0], device=device).pow(2 / 3) * 0.999 + 0.001
    noisy = timestep[:, None, None] * noise + (1 - timestep[:, None, None]) * actions
    target = (noise - actions)[..., :14]
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with precision:
        prediction = actor.training_prediction(batch, noisy, timestep)
    return (prediction.float() - target).square()[valid.unsqueeze(-1).expand_as(target)].mean()


@torch.no_grad()
def validate(model, loader):
    model.completion.eval()
    targets, probabilities = [], []
    for raw in loader:
        b = to_device(raw, model.device)
        valid = b['completion_valid']
        logits = model.completion(b['completion_images'], b['completion_state'], b['condition_index'])
        targets.extend(b['completion_target'][valid].cpu().tolist())
        probabilities.extend(logits.sigmoid()[valid].cpu().tolist())
    rows = {}
    for threshold in (0.5, 0.9):
        tp = sum(y == 1 and p >= threshold for y, p in zip(targets, probabilities))
        fp = sum(y == 0 and p >= threshold for y, p in zip(targets, probabilities))
        positives, negatives = targets.count(1.0), targets.count(0.0)
        rows[str(threshold)] = {'true_positive': tp, 'false_positive': fp,
                               'positive_frames': positives, 'negative_frames': negatives,
                               'recall': tp / positives if positives else None,
                               'precision': tp / (tp + fp) if tp + fp else None,
                               'false_positive_rate': fp / negatives if negatives else None}
    model.completion.train()
    return {'thresholds': rows, 'labelled_frames': len(targets),
            'note': '独立seed的逐帧完成预测指标，不是独立技能执行成功率'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('policy-root', 'data-root', 'output', 'init-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path)
    parser.add_argument('--updates', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.gradient_accumulation) < 1:
        raise ValueError('更新数和batch参数须为正')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'skill_checkpoint.pt'
    if checkpoint.exists():
        raise FileExistsError('拒绝覆盖已有技能checkpoint；本入口不隐式续训')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    runtime = legacy_module(str(args.policy_root.resolve()), 'runtime')
    stats = load_norm(args.norm_stats or (runtime.MODEL_ROOT / 'norm_stats.json'))
    # 先读取技能词表；真实图像和token只在getitem时生成。
    train = SkillDataset(args.data_root, 'train', args.policy_root, stats, None, None)
    model = AtomicSkillModel(args.policy_root, train.condition_to_index, stats,
                             args.device, args.init_checkpoint)
    image_transform, tokenizer = runtime.build_demo_transforms(model.model)
    train.image_transform, train.tokenizer = image_transform, tokenizer
    val = SkillDataset(args.data_root, 'val', args.policy_root, stats, tokenizer,
                       image_transform, train.condition_to_index)
    if train.sources & val.sources:
        raise ValueError('训练和验证出现同任务同seed')
    if not train.label_counts[0] or not train.label_counts[1]:
        raise ValueError('完成头训练需要显式正例和负例')
    sample_count = args.updates * args.gradient_accumulation * args.batch_size
    indices, available = balanced_action_indices(train, sample_count, args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=indices, num_workers=0, collate_fn=train.collate)
    validation = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=val.collate)
    parameters = [p for p in model.actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': 2.5e-5},
                                  {'params': model.completion.parameters(), 'lr': 1e-3}],
                                 betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    pos_weight = torch.tensor(train.label_counts[0] / train.label_counts[1], device=model.device)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'train_samples': len(train), 'val_samples': len(val), 'skill_to_index': train.skill_to_index,
              'condition_to_index': train.condition_to_index,
              'train_label_counts': train.label_counts, 'val_label_counts': val.label_counts,
              'train_sources': sorted(train.sources), 'val_sources': sorted(val.sources),
              'action_sampling': 'shuffled_round_robin_task_then_valid_action_condition_then_frame',
              'action_sampling_budget': sample_count,
              'sampling_uses_validation': False,
              'action_sampling_sources': [dict(task=task, condition_index=index, valid_action_samples=count)
                                          for (task, index), count in sorted(available.items())],
              'completion_architecture': 'three_view_small_cnn_qpos_skill_binding_embedding',
              'completion_pos_weight': pos_weight.item(), 'terminal_labels': 'explicit_predicates_only'}
    (args.output / 'TRAIN_CONFIG.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    iterator, started = iter(loader), time.time()
    visits, targets, unique = Counter(), Counter(), defaultdict(set)
    model.completion.train()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        values = {'action_loss': [], 'completion_loss': []}
        consumed = []
        for _ in range(args.gradient_accumulation):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = to_device(raw, model.device)
            actor_loss = action_loss(model.actor, batch, model.device)
            stop_valid = batch['completion_valid']
            stop_loss = None
            if bool(stop_valid.any()):
                logits = model.completion(batch['completion_images'], batch['completion_state'], batch['condition_index'])
                stop_loss = F.binary_cross_entropy_with_logits(logits[stop_valid], batch['completion_target'][stop_valid], pos_weight=pos_weight)
            terms = []
            for name, loss in (('action_loss', actor_loss), ('completion_loss', stop_loss)):
                if loss is not None:
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f'非有限loss: {name}')
                    terms.append(loss)
                    values[name].append(loss.item())
            if terms:
                (sum(terms) / args.gradient_accumulation).backward()
            if actor_loss is not None:
                valid_counts = raw['action_valid_mask'].sum(dim=1).tolist()
                for i, count in enumerate(valid_counts):
                    if count:
                        example = raw['stableexampleindex'][i]
                        consumed.append((example.split(':', 1)[0], int(raw['condition_index'][i]), example, count))
        optimizer.step()
        for task, index, example, count in consumed:
            visits[(task, index)] += 1
            targets[(task, index)] += count
            unique[(task, index)].add(example)
        row = {'update': update, 'elapsed_seconds': time.time() - started,
               'actor_samples_this_update': len(consumed), 'actor_samples_consumed': sum(visits.values()),
               'actor_task_conditions_consumed': len(visits),
               **{k: sum(v) / len(v) if v else None for k, v in values.items()}}
        with (args.output / 'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if update % 100 == 0 or update == args.updates:
            coverage = coverage_snapshot(available, visits, targets, unique, train.condition_to_index, update)
            payload = {'format': 'atomic_skill_lora_v1', 'adapter_state': model.actor.trainable_state_dict(),
                       'completion_state': {k: v.detach().cpu() for k, v in model.completion.state_dict().items()},
                       'skill_to_index': train.skill_to_index, 'condition_to_index': train.condition_to_index,
                       'norm_stats': stats, 'update': update,
                       'run_complete': update == args.updates, 'config': config,
                       'actor_sample_coverage': coverage}
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
            coverage_path = args.output / 'ACTOR_SAMPLE_COVERAGE.json'
            temporary = coverage_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(coverage_path)
    result = {'status': 'complete', 'update': args.updates, 'checkpoint': str(checkpoint),
              'elapsed_seconds': time.time() - started, 'validation': validate(model, validation),
              'actor_sample_coverage': coverage,
              'skill_execution_sr': None}
    (args.output / 'TRAIN_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
