"""每次运行一个真实回合；影子模式只预测一次，不虚构已执行历史。"""
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from PIL import Image

from real_robot.airbot import JointMapping, MotionLimits


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()

    def __call__(self, event):
        row = {'time_unix': time.time(), 'monotonic': time.monotonic(), **event}
        with self.lock, (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def observation(self, index, physical, model, images, cameras):
        folder = self.root / f'observation_{index:04d}'
        folder.mkdir()
        for name, image in images.items():
            Image.fromarray(image).save(folder / f'{name}.png')
        self({'event': 'observation', 'index': index, 'robot_state': physical.tolist(),
              'model_state': model.tolist(), 'camera_receipts': cameras,
              'images': str(folder.relative_to(self.root))})


def validate_config(config, mode):
    for name in ('frequency_hz', 'inference_timeout_s', 'max_cycle_overrun_s'):
        value = config[name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须为正数')
    for name in ('max_steps', 'planner_call_limit'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f'{name} 必须为正整数')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed 必须为非负整数')
    if not str(config['instruction']).strip() or not str(config['task']).strip():
        raise ValueError('必须提供任务名和指令')
    if (type(config['robot']['rpc_timeout_ms']) is not int or config['robot']['rpc_timeout_ms'] < 1 or
            not np.isfinite(config['robot']['max_command_skew_s']) or
            config['robot']['max_command_skew_s'] <= 0):
        raise ValueError('RPC 超时和双臂下发间隔必须为正数')
    if not np.isfinite(config['robot']['max_state_age_s']) or config['robot']['max_state_age_s'] <= 0:
        raise ValueError('robot.max_state_age_s 必须为正数')
    for key in ('max_age_s', 'max_skew_s', 'fps', 'width', 'height'):
        if not np.isfinite(config['cameras'][key]) or config['cameras'][key] <= 0:
            raise ValueError(f'cameras.{key} 必须为正数')
    JointMapping(config['calibration'])
    if mode == 'execute':
        if config['calibration'].get('verified') is not True:
            raise ValueError('执行前须填写实测标定并设置 calibration.verified=true')
        MotionLimits(config['limits'], config['frequency_hz'])


def wait_prediction(policy, state, images, timeout, health):
    result = queue.Queue(maxsize=1)

    def work():
        try:
            result.put((True, policy.predict(state, images)))
        except BaseException as exc:
            result.put((False, exc))

    # 该线程只能访问模型；超时后即使迟到返回，也不能下发硬件动作。
    threading.Thread(target=work, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        health()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('模型/Astra 推理超时，终止本回合；不重试')
        try:
            ok, value = result.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if not ok:
            raise value
        return value


def run_trial(config, mode, robot, cameras, policy, journal):
    mapping = JointMapping(config['calibration'])
    limits = MotionLimits(config['limits'], config['frequency_hz']) if mode == 'execute' else None
    result = {'status': 'error', 'mode': mode, 'submitted_action_pairs': 0, 'task_success': None,
              'success_source': 'not_labelled', 'error': None, 'cleanup_errors': []}
    emergency, step = False, 0
    try:
        cameras.wait_ready()
        if mode == 'execute':
            limits.check_state(robot.read())
            robot.acquire(limits)
        observation = 0
        while True:
            if (journal.root / 'STOP').exists():
                raise KeyboardInterrupt('收到 STOP 文件')
            physical = robot.read()
            images, camera_meta = cameras.read()
            state = mapping.to_model(physical)
            journal.observation(observation, physical, state, images, camera_meta)
            observation += 1
            if mode == 'observe':
                result['status'] = 'observed'
                break

            def health():
                if (journal.root / 'STOP').exists():
                    raise KeyboardInterrupt('收到 STOP 文件')
                cameras.read()
                if hasattr(policy, 'check_connection'):
                    policy.check_connection()
                current = robot.read()
                if limits is not None:
                    limits.check_tracking(current, physical)

            predicted, status = wait_prediction(policy, state, images,
                                                config['inference_timeout_s'], health)
            model_actions = np.asarray(predicted, dtype=np.float64)
            robot_actions = mapping.to_robot(model_actions)
            journal({'event': 'prediction', 'observation': observation - 1, 'status': status,
                     'model_actions': model_actions.tolist(), 'robot_actions': robot_actions.tolist(),
                     'calibration_verified': config['calibration']['verified']})
            if mode == 'shadow':
                result['status'] = 'shadow_complete'
                break
            if not len(robot_actions):
                if status == 'running':
                    raise RuntimeError('控制器报告运行中却未返回动作')
                result['status'] = status
                break
            health()
            current = robot.read()
            limits.check_chunk(robot_actions, current)
            remaining = config['max_steps'] - step
            # 仅回合总预算可截断最后一块；之后直接终止，不向控制器伪报整块已执行。
            selected = robot_actions[:remaining]
            if not len(selected):
                result['status'] = 'budget_exhausted'
                break
            robot.begin_chunk()
            period = 1.0 / config['frequency_hz']
            previous = current
            for action in selected:
                started = time.monotonic()
                if (journal.root / 'STOP').exists():
                    raise KeyboardInterrupt('收到 STOP 文件')
                if hasattr(policy, 'check_connection'):
                    policy.check_connection()
                cameras.read()
                measured = robot.read()
                limits.check_tracking(measured, previous)
                # 防止控制器严重滞后时继续向远处发送目标。
                limits.check_tracking(measured, action)
                skew = robot.send(action)
                step += 1
                result['submitted_action_pairs'] = step
                journal({'event': 'action', 'step': step, 'target': action.tolist(),
                         'feedback_before': measured.tolist(), 'command_skew_s': skew})
                elapsed = time.monotonic() - started
                if elapsed > period + config['max_cycle_overrun_s']:
                    raise TimeoutError('动作下发周期超时；不追赶补发')
                time.sleep(max(0, period - elapsed))
                previous = action
            feedback = robot.read()
            limits.check_tracking(feedback, selected[-1])
            journal({'event': 'chunk_feedback', 'step': step, 'state': feedback.tolist()})
            # 推理和联网期间保持当前位置，不继续播放上一段预测。
            robot.hold()
            if step >= config['max_steps']:
                result['status'] = 'budget_exhausted'
                break
    except KeyboardInterrupt as exc:
        result.update(status='interrupted', error=str(exc) or '用户中断')
        emergency = True
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        emergency = True
    finally:
        result['cleanup_errors'].extend(robot.stop(emergency))
        result['cleanup_errors'].extend(robot.close())
        result['cleanup_errors'].extend(cameras.close())
        if policy is not None and hasattr(policy, 'close'):
            try:
                policy.close()
            except Exception as exc:
                result['cleanup_errors'].append(f'policy.close: {exc}')
        if result['cleanup_errors']:
            result['status'] = 'cleanup_error'
        journal({'event': 'trial_end', **result})
        (journal.root / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result
