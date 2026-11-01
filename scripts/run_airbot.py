#!/usr/bin/env python3
"""AIRBOT 双臂入口；默认 observe，不申请控制权，不下发动作。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys

REPO = Path(__file__).resolve().parents[1]
for relative in ('', 'method', 'training', 'runtime',
                 'runtime/XPolicyLab/policy/TwinTimeFM/vendor/backbone_lib'):
    sys.path.insert(0, str(REPO / relative))
os.environ.setdefault('TWINTIMEFM_ASSETS', str(REPO / 'models/base'))

from real_robot.airbot import DualAirbot, load_sdk
from real_robot.cameras import CameraRig, list_cameras
from real_robot.policy import RobotPolicy, bundle_paths
from real_robot.remote import RemotePolicy, ssh_command
from real_robot.runner import Journal, run_trial, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--list-cameras', action='store_true', help='仅列出 RealSense 设备和序列号')
    parser.add_argument('--mode', choices=('observe', 'shadow', 'execute'), default='observe')
    parser.add_argument('--policy', choices=('skill', 'native', 'duoflow'), default='native')
    parser.add_argument('--skill-id')
    parser.add_argument('--list-skills', action='store_true')
    parser.add_argument('--remote-host', help='5090 的 SSH 别名或 user@host；不填则本机推理')
    parser.add_argument('--remote-repo', help='5090 上 DuoFlow 仓库的绝对路径')
    parser.add_argument('--remote-python', help='5090 策略环境 Python 的绝对路径')
    parser.add_argument('--run-dir', help='新建回合目录，不能复用已有目录')
    args = parser.parse_args()
    if args.list_cameras:
        print(json.dumps(list_cameras(), ensure_ascii=False, indent=2))
        return 0
    if not args.config:
        parser.error('请提供 --config，或用 --list-cameras 列出设备')
    if args.remote_host and (not args.remote_repo or not args.remote_python):
        parser.error('--remote-host 需要同时提供 --remote-repo 和 --remote-python')
    if not args.remote_host and (args.remote_repo or args.remote_python):
        parser.error('远端路径需与 --remote-host 一起使用')
    config = json.loads(Path(args.config).expanduser().read_text())
    bundle = Path(config['bundle']).expanduser()
    if not args.remote_host or args.list_skills:
        config['bundle'] = str((REPO / bundle).resolve())
    if args.list_skills:
        paths = bundle_paths(config['bundle'], config['task'])
        cards = json.loads(paths['catalog'].read_text())['skills']
        print(json.dumps([{'id': c['id'], 'skill_id': c['skill_id'], 'skill_text': c['skill_text'],
                           'max_policy_steps': c['max_policy_steps']}
                          for c in cards if c['task'] == config['task']], ensure_ascii=False, indent=2))
        return 0
    validate_config(config, args.mode)
    directory = args.run_dir or str(REPO / 'runs/airbot' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    journal = Journal(directory)
    (journal.root / 'CONFIG.json').write_text(json.dumps(
        {'mode': args.mode, 'policy': args.policy, 'skill_id': args.skill_id, 'config': config,
         'remote_host': args.remote_host, 'remote_repo': args.remote_repo, 'remote_python': args.remote_python},
        ensure_ascii=False, indent=2) + '\n')
    print(f'回合目录: {journal.root.resolve()}\n模式: {args.mode} / {args.policy}', flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'收到信号 {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    robot = cameras = policy = None
    handed_to_runner = False
    try:
        sdk = load_sdk()
        # 大模型先加载；模型或路径报错时尚未连接机器人。
        if args.mode != 'observe':
            if args.remote_host:
                command = ssh_command(args.remote_host, args.remote_repo, args.remote_python)
                print('正在通过 SSH 加载远端策略；日志见回合目录 remote_stderr.log', flush=True)
                policy = RemotePolicy(command, config, args.policy, args.skill_id, journal)
            else:
                policy = RobotPolicy(config, args.policy, args.skill_id, journal.root, REPO, journal)
        cameras = CameraRig(config['cameras'])
        robot = DualAirbot(config['robot'], journal, sdk=sdk)
        handed_to_runner = True
        result = run_trial(config, args.mode, robot, cameras, policy, journal)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result['error'] or result['cleanup_errors'] else 0
    except (Exception, KeyboardInterrupt) as exc:
        result = {'status': 'startup_failed', 'error': f'{type(exc).__name__}: {exc}',
                  'task_success': None, 'submitted_action_pairs': 0}
        (journal.root / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    finally:
        if not handed_to_runner:
            errors = []
            if robot is not None:
                errors.extend(robot.stop(True))
                errors.extend(robot.close())
            if cameras is not None:
                errors.extend(cameras.close())
            if policy is not None and hasattr(policy, 'close'):
                try:
                    policy.close()
                except Exception as exc:
                    errors.append(f'policy.close: {exc}')
            journal({'event': 'startup_failed', 'cleanup_errors': errors})


if __name__ == '__main__':
    raise SystemExit(main())
