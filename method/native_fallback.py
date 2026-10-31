"""原 native 控制器独立维护状态，只消费自己实际发出的动作。"""
from copy import copy

import numpy as np


def missing_execution_contracts(cards, task):
    # 这些要求来自任务技能合同，不读取本回合的成功标签或仿真物理状态。
    expected = []
    if task == 'handover_block':
        expected = [('place_on_pad', 'right', None)]
    elif task == 'stack_blocks_two':
        expected = [('place_and_release_block', arm, obj)
                    for arm in ('left', 'right') for obj in ('the red block', 'the green block')]
    missing = []
    for skill, arm, obj in expected:
        if not any(c.get('atomic_skill_id') == skill and
                   c.get('bindings', {}).get('arm') == arm and
                   (obj is None or c.get('bindings', {}).get('object') == obj) for c in cards):
            missing.append({'skill_id': skill, 'arm': arm, 'object': obj})
    return missing


class NativeFallback:
    def __init__(self, owner, planner, runtime_editor, checkpoint):
        self.owner, self.planner, self.runtime_editor = owner, planner, runtime_editor
        self.checkpoint = checkpoint
        self.reset()

    def reset(self):
        self.model = None
        self.active = False

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        owner = self.owner
        model = copy(owner)
        for name in ('_atomic_actors', '_native_fallback'):
            model.__dict__.pop(name, None)
        model.execution_contract, model.planner_backend = 'legacy', 'native'
        model.native_fallback_enabled = False
        model.planner, model.runtime_editor = self.planner, self.runtime_editor
        model.model_version = str(self.checkpoint)
        model.run_dir = owner.run_dir / 'native_controller'
        model.run_dir.mkdir(exist_ok=True)
        model._log = lambda event: owner._log({**event, 'control_owner': 'full_native_fallback'})
        model.reset()
        model.case, model.episode_id = dict(owner.case), owner.episode_id
        self.model = model
        return model

    def step(self, state, images, instruction):
        from XPolicyLab.utils.process_data import pack_robot_state
        owner, model = self.owner, self._ensure_model()
        # selector 与 actor 都必须看到原始 LoRA，而不只是动作生成前才恢复。
        owner._atomic_actors._activate('whole')
        if not self.active:
            model._reset_pending = True
        env_id, obs = owner._native_current_observation
        model._observations = {env_id: obs}
        owner._active_actor_policy = model
        try:
            actions = model._predict(env_id)
        finally:
            owner._active_actor_policy = owner
        self.active = True
        # 原 native 已执行其标准的 50 步预测、30 步下发规则。
        return np.asarray([pack_robot_state({'state': action}, 'joint', owner.dim_info)
                           for action in actions], dtype=np.float32)

    def suspend(self, state, images):
        if not self.active:
            return
        from transition_evidence import build_transition_evidence
        model = self.model
        _, obs = self.owner._native_current_observation
        transition = None
        if model._last_public_observation is not None:
            transition = build_transition_evidence(before=model._last_public_observation, after=obs,
                before_qpos=model._last_public_state, after_qpos=state)
        if (model._last_issued_plan_index is not None and model.plan_state is not None and
                model._last_issued_plan_index == model.plan_state.current_index):
            model.plan_state.mark_executed(model._last_issued_plan_index,
                                          transition_evidence=transition)
        model._log_harness_post_observation(obs)
        # 暂停期间 Astra 的动作不能增加 native 的尝试数或成为 native 转移证据。
        model._last_issued_plan_index = None
        model._last_public_observation = model._last_public_state = None
        self.active = False
