#!/usr/bin/env python3
"""由设备端通过 SSH 启动的单回合推理进程；不连接相机或机器人。"""
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
for relative in ('', 'method', 'training', 'runtime',
                 'runtime/XPolicyLab/policy/TwinTimeFM/vendor/backbone_lib'):
    sys.path.insert(0, str(REPO / relative))
os.environ.setdefault('TWINTIMEFM_ASSETS', str(REPO / 'models/base'))


def main():
    # 含原生库输出在内的模型日志走 stderr，stdout 专供二进制协议。
    wire = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from real_robot.policy_worker import serve
    code = serve(sys.stdin.buffer, wire, REPO)
    wire.close()
    sys.stdout.flush()
    sys.stderr.flush()
    # 断连时也退出本次独占推理进程，避免迟到的模型线程继续占用 GPU。
    os._exit(code)


if __name__ == '__main__':
    main()
