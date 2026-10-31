"""发送一次合成观测，检查登录桥与结构化决策；不连接机器人。"""
import argparse
import json
from pathlib import Path

import numpy as np

from astra_reentry import AstraNativeFallbackBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-root', required=True)
    args = parser.parse_args()
    root = Path(args.request_root).resolve()
    planner = AstraNativeFallbackBackend(transport='siwc_files', request_root=str(root),
                                        audit_dir=str(root / 'connection_check'))
    result = planner.rank(
        images={name: np.zeros((64, 64, 3), dtype=np.uint8)
                for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
        state=np.zeros(14),
        task='Connection check with synthetic black images; no robot is connected. Choose whole_task and report progress as unknown.',
        candidates=[{'skill_id': 'whole_task', 'execution_prompt': 'Connection check only; no action will execute.'}],
        memory=[],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
