import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from convert_lerobot import convert, make_template, mapped_vector, write_json
from prepare_training_data import prepare
from skill_dataset import SkillDataset, bounded_targets
from skill_model import CompletionHead
from train_completion import CompletionDataset
from train_skills import balanced_action_indices

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'runtime/XPolicyLab/policy/TwinTimeFM'


def write_video(path, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode='w') as output:
        stream = output.add_stream('libx264', rate=5)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for color in colors:
            array = np.full((32, 32, 3), color, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def fixture(root, version='v3.0', visual='video'):
    root.mkdir()
    (root / 'meta').mkdir()
    cameras = dict(zip(('cam_high', 'cam_left_wrist', 'cam_right_wrist'),
                      ('observation.images.top', 'observation.images.left', 'observation.images.right')))
    features = {k: {'dtype': 'image' if visual == 'image_path' else visual, 'shape': [32, 32, 3]}
                for k in cameras.values()}
    features.update({k: {'dtype': 'float32', 'shape': [14]} for k in ('observation.state', 'action')})
    video_path = ('videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
                  if version == 'v2.1' else 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    info = {'codebase_version': version, 'fps': 5, 'chunks_size': 1000,
            'features': features, 'video_path': video_path}
    write_json(root / 'meta/info.json', info)
    annotations = {'format': 'duoflow_lerobot_annotations_v1', 'camera_map': cameras,
                   'state': {'key': 'observation.state', 'indices': list(reversed(range(14)))},
                   'action': {'key': 'action', 'mode': 'absolute_joint', 'alignment': 'same_row'},
                   'episodes': []}
    metadata, all_rows = [], []
    colors = [(220, 20, 20), (20, 220, 20)]
    for episode in range(2):
        meta = {'episode_index': episode, 'length': 6, 'tasks': ['Handover block']}
        for key in cameras.values():
            meta.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                         f'videos/{key}/from_timestamp': episode * 1.2,
                         f'videos/{key}/to_timestamp': (episode + 1) * 1.2})
            if visual == 'video' and version == 'v2.1':
                write_video(root / video_path.format(episode_chunk=0, episode_index=episode, video_key=key),
                            [colors[episode]] * 6)
        metadata.append(meta)
        rows = []
        for frame in range(6):
            state = [episode + frame * 0.1 + i * 0.01 for i in range(14)]
            row = {'episode_index': episode, 'frame_index': frame, 'timestamp': frame / 5,
                   'observation.state': state, 'action': [v + 0.25 for v in state]}
            if visual in ('image', 'image_path'):
                data = io.BytesIO()
                Image.new('RGB', (32, 32), colors[episode]).save(data, format='PNG')
                relative = f'frames/{episode}_{frame}.png'
                if visual == 'image_path':
                    (root / 'frames').mkdir(exist_ok=True)
                    (root / relative).write_bytes(data.getvalue())
                for key in cameras.values():
                    row[key] = ({'bytes': data.getvalue(), 'path': None} if visual == 'image'
                                else {'bytes': None, 'path': relative})
            rows.append(row)
        if version == 'v2.1':
            path = root / f'data/chunk-000/episode_{episode:06d}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
        all_rows.extend(rows)
        annotations['episodes'].append({'episode_index': episode, 'task': 'handover_block',
            'split': 'train' if episode == 0 else 'val', 'segments': [{
                'skill_id': 'grasp_block', 'skill_text': 'Grasp the block with the left arm.',
                'bindings': {'arm': 'left', 'object': 'the block', 'destination': ''},
                'precondition': 'The block is on the table.', 'max_policy_steps': 100,
                'start_frame': 0, 'end_frame': 5, 'action_train_valid': True,
                'completion_spans': [{'start_frame': 0, 'end_frame': 1, 'label': 0},
                                     {'start_frame': 4, 'end_frame': 5, 'label': 1}]}]})
    if version == 'v2.1':
        (root / 'meta/episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    else:
        for name, rows in [('data/chunk-000/file-000.parquet', all_rows),
                           ('meta/episodes/chunk-000/file-000.parquet', metadata)]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
        if visual == 'video':
            for key in cameras.values():
                write_video(root / video_path.format(video_key=key, chunk_index=0, file_index=0),
                            [colors[0]] * 6 + [colors[1]] * 6)
    path = root.parent / (root.name + '_annotations.json')
    write_json(path, annotations)
    return path, annotations


class ConversionTests(unittest.TestCase):
    def test_formats_images_video_and_training_readers(self):
        for version in ('v2.1', 'v3.0'):
            for visual in ('image', 'image_path', 'video'):
                with self.subTest(version=version, visual=visual), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    annotations, _ = fixture(root / 'source', version, visual)
                    report = convert(root / 'source', annotations, root / 'converted')
                    self.assertEqual(report['counts']['train']['action_frames'], 5)
                    self.assertEqual(report['counts']['val']['completion_1'], 2)
                    for episode in (0, 1):
                        directory = root / f'converted/handover_block/seed_{episode:06d}'
                        rows = [json.loads(r) for r in (directory / 'observations.jsonl').read_text().splitlines()]
                        pixels = np.asarray(Image.open(directory / rows[0]['images']['cam_high']))
                        self.assertEqual(int(pixels.mean(axis=(0, 1)).argmax()), episode)
                        self.assertAlmostEqual(rows[0]['state'][0], episode + 0.13)
                        self.assertAlmostEqual(rows[0]['action'][0], episode + 0.25)
                        if version == 'v3.0' and visual == 'video':
                            self.assertAlmostEqual(rows[0]['image_timestamps']['cam_high'], episode * 1.2)
                    stats = {k: {'q01': [-2.0] * 14, 'q99': [2.0] * 14} for k in ('observation.state', 'action')}
                    transform = lambda images: (list(images.values()), [True] * 3, {})
                    data = SkillDataset(root / 'converted', 'train', POLICY, stats, lambda value: {}, transform)
                    sample = data[0]
                    self.assertEqual(int(sample['action_valid_mask'].sum()), 5)
                    # 真实 action[0]-state[0]=0.12；若误用后续 state 会得到另一个值。
                    expected = (0.12 + 2.0) / (4.0 + 1e-6) * 2 - 1
                    self.assertAlmostEqual(float(sample['actions32'][0, 0]), expected, places=5)
                    self.assertFalse(data[5]['action_valid_mask'].any())
                    self.assertFalse(data[2]['completion_valid'])
                    indices, _ = balanced_action_indices(data, 10, 0)
                    self.assertNotIn(5, indices)
                    completion = CompletionDataset(root / 'converted', 'train', POLICY, stats, data.condition_to_index)
                    self.assertEqual(completion.labels.tolist(), [0.0, 0.0, 1.0, 1.0])
                    head = CompletionHead(1)
                    logits = head(completion.images[completion.image_ids].float() / 127.5 - 1,
                                  completion.states, completion.conditions)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, completion.labels)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    coverage = prepare([root / 'converted'], ['handover_block'], root / 'prepared')
                    self.assertEqual(coverage['status'], 'ready')

    def test_action_boundary_and_legacy_unchanged(self):
        mapped = mapped_vector({'state': list(range(14))},
                               {'key': 'state', 'indices': list(reversed(range(14))),
                                'scale': [2.0] * 14, 'offset': [-1.0] * 14})
        self.assertEqual(mapped, list(range(25, -2, -2)))
        rows = [{'frame_index': i, 'state': [i] * 14, 'action': [10 + i] * 14} for i in range(5)]
        segment = {'end_frame': 3, 'action_train_valid': True}
        old, mask = bounded_targets(rows, 1, segment, horizon=4)
        self.assertEqual([r[0] for r in old], [2, 3, 3, 3])
        self.assertEqual(mask, [True, True, False, False])
        segment['action_source'] = 'recorded_action'
        new, mask = bounded_targets(rows, 1, segment, horizon=4)
        self.assertEqual([r[0] for r in new], [11, 12, 12, 12])
        self.assertEqual(mask, [True, True, False, False])
        segment['action_train_valid'] = False
        self.assertFalse(any(bounded_targets(rows, 1, segment)[1]))

    def test_invalid_annotations_leave_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, original = fixture(root / 'source', visual='image')
            cases = []
            config = copy.deepcopy(original); config['episodes'].append(copy.deepcopy(config['episodes'][0])); cases.append(config)
            config = copy.deepcopy(original); config['episodes'][1]['segments'][0]['skill_text'] = 'Unseen binding'; cases.append(config)
            config = copy.deepcopy(original); config['action']['mode'] = 'velocity'; cases.append(config)
            config = copy.deepcopy(original); config['episodes'][0]['segments'][0]['end_frame'] = 7; cases.append(config)
            config = copy.deepcopy(original); config['episodes'][0]['segments'][0]['completion_spans'][0]['label'] = 2; cases.append(config)
            config = copy.deepcopy(original); config['state']['indices'] = [0] * 14; cases.append(config)
            for config in cases:
                write_json(path, config)
                with self.assertRaises(ValueError):
                    convert(root / 'source', path, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_template_does_not_guess_labels_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = fixture(root / 'source', visual='image')
            make_template(root / 'source', root / 'template.json')
            template = json.loads((root / 'template.json').read_text())
            self.assertIsNone(template['action']['mode'])
            self.assertEqual(template['episodes'][0]['segments'], [])
            self.assertIsNone(template['camera_map']['cam_high'])
            (root / 'out').mkdir()
            (root / 'out/keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                convert(root / 'source', path, root / 'out')
            self.assertEqual((root / 'out/keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
