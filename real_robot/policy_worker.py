"""一条 SSH 会话对应一个策略回合；只有设备端推进下一次真实观测。"""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
import uuid

from PIL import Image

from real_robot.airbot import vector
from real_robot.policy import RobotPolicy
from real_robot.remote import decode_images, receive, transmit, validate_actions
from real_robot.runner import Journal


def _compute(function, disconnected, timeout):
    result = queue.Queue(maxsize=1)

    def work():
        try:
            result.put((True, function()))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=work, daemon=True).start()
    deadline = time.monotonic() + timeout
    while not disconnected.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('远端模型超时')
        try:
            ok, value = result.get(timeout=min(0.1, remaining))
        except queue.Empty:
            continue
        if not ok:
            raise value
        if disconnected.is_set():
            break
        return value
    raise ConnectionError('设备端已断开；终止本次推理')


def serve(reader, writer, repo, policy_factory=RobotPolicy):
    incoming, disconnected = queue.Queue(maxsize=2), threading.Event()
    read_errors = []

    def read_loop():
        try:
            while True:
                incoming.put(receive(reader))
        except Exception as exc:
            read_errors.append(str(exc))
            disconnected.set()

    def next_message():
        while not disconnected.is_set():
            try:
                return incoming.get(timeout=0.1)
            except queue.Empty:
                continue
        raise ConnectionError('; '.join(read_errors) or '设备端已断开')

    threading.Thread(target=read_loop, daemon=True).start()
    journal = None
    result = {'status': 'error', 'predictions': 0, 'error': None}
    try:
        initial = next_message()
        if initial.get('type') != 'init' or initial.get('version') != 1:
            raise ValueError('首条消息需为 init/version=1')
        config = dict(initial['config'])
        if initial['route'] not in {'skill', 'native', 'duoflow'}:
            raise ValueError('未知策略')
        root = (Path(repo) / 'runs/remote_policy' /
                (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:12]))
        journal = Journal(root)
        (root / 'CONFIG.json').write_text(json.dumps(initial, ensure_ascii=False, indent=2) + '\n')
        config['bundle'] = str((Path(repo) / Path(config['bundle']).expanduser()).resolve())
        policy = _compute(lambda: policy_factory(config, initial['route'], initial.get('skill_id'),
                                                root, Path(repo), journal), disconnected, 600)
        transmit(writer, {'type': 'ready', 'version': 1, 'config': initial['config'],
                          'route': initial['route'], 'run_dir': str(root)})
        while True:
            request = next_message()
            if request.get('type') == 'close':
                result['status'] = 'closed_by_client'
                break
            seq = result['predictions'] + 1
            if request.get('type') != 'predict' or type(request.get('seq')) is not int or request['seq'] != seq:
                raise ValueError('请求顺序非法；不重放历史或跳过观测')
            state = vector(request['state'], 'model_state').astype('float32')
            images = decode_images(request['images'])
            folder = root / f'observation_{seq:04d}'
            folder.mkdir()
            for name, image in images.items():
                Image.fromarray(image).save(folder / f'{name}.png')
            journal({'event': 'remote_observation', 'seq': seq, 'model_state': state.tolist(),
                     'images': folder.name})
            actions, status = _compute(lambda: policy.predict(state, images), disconnected,
                                       config['inference_timeout_s'])
            actions = validate_actions(actions)
            if not isinstance(status, str) or not status:
                raise ValueError('模型缺少执行状态')
            if disconnected.is_set():
                raise ConnectionError('设备端已断开；丢弃预测')
            transmit(writer, {'type': 'prediction', 'seq': seq, 'actions': actions.tolist(), 'status': status})
            result['predictions'] = seq
            journal({'event': 'prediction_returned', 'seq': seq, 'status': status,
                     'action_count': len(actions), 'physical_execution_confirmed': False})
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if not disconnected.is_set():
            try:
                transmit(writer, {'type': 'error', 'error': result['error']})
            except (OSError, ConnectionError):
                pass
    finally:
        if journal is not None:
            journal({'event': 'worker_end', **result})
            (journal.root / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 0 if result['error'] is None else 1
