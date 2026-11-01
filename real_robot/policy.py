"""将实测 RGB/关节状态接到已有技能和完整 DuoFlow 控制器。"""
import json
from pathlib import Path

import numpy as np

from real_robot.cameras import CAMERAS


def bundle_paths(root, task):
    root = Path(root)
    bundle = json.loads((root / 'BUNDLE.json').read_text())
    if task not in bundle['task_to_bank']:
        raise ValueError(f'权重包没有任务 {task}')
    bank = bundle['banks'][bundle['task_to_bank'][task]]
    return {key: root / bank[key] for key in ('actor', 'completion', 'catalog')} | {
        key: root / value for key, value in bundle['native'].items()}


def model_observation(state, images, instruction):
    return {'env_idx': 0, 'instruction': instruction,
            'state': {'left_arm_joint_state': state[:6], 'left_ee_joint_state': state[6:7],
                      'right_arm_joint_state': state[7:13], 'right_ee_joint_state': state[13:14]},
            'vision': {name: {'color': images[name]} for name in CAMERAS}}


def model_config(config, route, paths, run_dir, repo):
    result = {
        'run_dir': str(run_dir / 'controller'), 'split': 'eval', 'condition': 'real_robot',
        'variant': 'combined', 'env_cfg_type': 'aloha_agilex', 'action_type': 'joint',
        'camera_map': {key: key for key in CAMERAS}, 'model_path': str(paths['adapter'].parent),
        'memory_dir': str(run_dir / 'empty_memory'), 'planner_checkpoint': str(paths['selector']),
        'runtime_editor_checkpoint': str(paths['runtime_editor']), 'harness_variant': 'rsi',
        'planner_backend': 'native', 'execution_contract': 'legacy', 'native_fallback': False,
    }
    if route == 'duoflow':
        if config.get('allow_cloud_observations') is not True:
            raise ValueError('DuoFlow 会向 Astra 发送真机画面；请显式设置 allow_cloud_observations=true')
        request_root = Path(config['astra_request_root']).expanduser()
        if not request_root.is_absolute():
            raise ValueError('astra_request_root 必须为绝对路径，且与登录桥一致')
        result.update(
            planner_backend='astra_guided', execution_contract='reentry_atomic_v1', native_fallback=True,
            atomic_policy_root=str(repo / 'runtime/XPolicyLab/policy/TwinTimeFM'),
            atomic_actor_checkpoint=str(paths['actor']), atomic_completion_checkpoint=str(paths['completion']),
            atomic_skill_catalog=str(paths['catalog']), reentry_max_steps=config['max_steps'],
            reentry_planner_call_limit=config['planner_call_limit'],
            astra_transport='siwc_files', astra_request_root=str(request_root),
        )
    return result


class RobotPolicy:
    def __init__(self, config, route, skill_id, run_dir, repo, log):
        self.config, self.route = config, route
        self.log, self.index = log, 0
        paths = bundle_paths(config['bundle'], config['task'])
        import torch
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        if route == 'skill':
            from XPolicyLab.policy.TwinTimeFM.inference import TwinTimeInference
            from astra_reentry import load_atomic_cards
            from reentry_actors import SharedAtomicActors
            from atomic_skill_executor import SkillExecutor
            cards = load_atomic_cards(paths['catalog'], config['task'])
            chosen = next((c for c in cards if c['skill_id'] == skill_id), None)
            if chosen is None:
                raise ValueError('请用 --list-skills 查看该任务的卡片，再通过 --skill-id 选择')
            engine = TwinTimeInference(paths['adapter'])
            actors = SharedAtomicActors(engine, repo / 'runtime/XPolicyLab/policy/TwinTimeFM',
                                        paths['actor'], paths['completion'],
                                        lambda text, mode: log({'event': 'actor_text', 'text': text, 'mode': mode}))
            self.executor = SkillExecutor(actors, actors.threshold)
            self.executor.start({'skill_id': chosen['atomic_skill_id'], 'skill_text': chosen['execution_prompt'],
                                 'max_policy_steps': min(chosen['max_policy_steps'], config['max_steps'])})
            log({'event': 'selected_skill', 'card': chosen})
        else:
            from skill_policy import Model
            cfg = model_config(config, route, paths, run_dir, repo)
            self.model = Model(cfg)
            self.model.prepare_case({'task_name': config['task'], 'seed': config['seed']})
            log({'event': 'model_config', 'config': cfg, 'seed_scope': 'model_sampling_only'})

    def predict(self, state, images):
        self.index += 1
        if self.route == 'skill':
            result = self.executor.step({'state': state, 'images': images, 'observation_id': self.index})
            self.log({'event': 'skill_step', **{k: v for k, v in result.items() if k != 'actions'}})
            return result['actions'], result['status']
        from XPolicyLab.utils.process_data import pack_robot_state
        self.model.update_obs(model_observation(state, images, self.config['instruction']))
        actions = self.model.get_action()
        packed = np.asarray([pack_robot_state({'state': action}, 'joint', self.model.dim_info)
                             for action in actions], dtype=np.float32).reshape(-1, 14)
        return packed, 'running' if len(packed) else 'controller_stopped'
