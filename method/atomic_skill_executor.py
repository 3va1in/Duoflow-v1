"""单次技能调用；停止后只能由显式 start 开始下一次调用。"""
import copy

import numpy as np


class SkillExecutor:
    def __init__(self, model, completion_threshold=0.9):
        if (isinstance(completion_threshold, bool) or
                not np.isfinite(completion_threshold) or not 0 < completion_threshold < 1):
            raise ValueError('完成阈值必须是 (0, 1) 内的有限数值')
        self.model = model
        self.threshold = completion_threshold
        self.spec = None
        self.status = 'idle'
        self.steps = 0
        self.high_count = 0
        self.last_observation_id = None
        self.first_prediction = True
        self.completion_reference = None

    def start(self, skill_spec):
        if self.status == 'running':
            raise RuntimeError('当前技能尚未终止')
        required = ('skill_id', 'skill_text', 'max_policy_steps')
        if any(key not in skill_spec for key in required):
            raise ValueError('技能入口必须包含 skill_id、skill_text 和 max_policy_steps')
        text = str(skill_spec['skill_text']).strip()
        if not text or 'next action' in text.lower():
            raise ValueError('只允许当前技能文本')
        condition = str(skill_spec['skill_id']) + '|' + ' '.join(text.split())
        mapping = getattr(self.model, 'condition_to_index', None)
        if skill_spec['skill_id'] == 'whole_task' or not isinstance(mapping, dict) or condition not in mapping:
            raise ValueError('技能及固定绑定必须属于实际加载的训练条件词表')
        maximum = skill_spec['max_policy_steps']
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise ValueError('max_policy_steps 必须是正整数')
        self.spec = copy.deepcopy(skill_spec)
        self.status, self.steps, self.high_count = 'running', 0, 0
        self.last_observation_id, self.first_prediction = None, True
        self.completion_reference = None

    def _result(self, actions=None, **fields):
        return {'status': self.status, 'actions': np.empty((0, 14), dtype=np.float32)
                if actions is None else actions,
                'policy_steps': self.steps, 'completion_streak': self.high_count, **fields}

    def step(self, public_observation, action_limit=10):
        if not isinstance(action_limit, int) or isinstance(action_limit, bool) or action_limit < 1:
            raise ValueError('本次动作上限必须为正整数')
        if self.spec is None:
            raise RuntimeError('必须先显式 start(skill_spec)')
        if self.status != 'running':
            return self._result()
        if self.steps >= self.spec['max_policy_steps'] and self.high_count != 1:
            self.status = 'timeout'
            return self._result()
        observation_id = public_observation['observation_id']
        if observation_id == self.last_observation_id:
            raise ValueError('重复观测不能重复消费或作为第二次完成确认')
        state = np.asarray(public_observation['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('state 必须是有限的 14 维 drive-target 状态')
        kwargs = {'reset': self.first_prediction}
        if getattr(self.model, 'completion_pair', False):
            if self.completion_reference is None:
                self.completion_reference = {'state': state.copy(), 'images':
                    {name: image.copy() for name, image in public_observation['images'].items()}}
            kwargs['completion_reference'] = self.completion_reference
        prediction = self.model.predict(
            state, public_observation['images'], self.spec['skill_text'],
            self.spec['skill_id'], **kwargs)
        probability = float(prediction['completion_prob'])
        actions = np.asarray(prediction['actions'], dtype=np.float32)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('completion_prob 必须在 [0, 1]')
        if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError('动作必须是非空、有限的 [H, 14] 数组')
        self.first_prediction, self.last_observation_id = False, observation_id
        self.high_count = self.high_count + 1 if probability >= self.threshold else 0
        if self.high_count >= 2:
            self.status = 'predicted_complete'
            return self._result(completion_prob=probability, decision='stop')
        if self.steps >= self.spec['max_policy_steps']:
            self.status = 'timeout'
            return self._result(completion_prob=probability, decision='budget_exhausted')
        if self.high_count == 1:
            # 两次确认之间必须经过真实的新观测；保持动作也计入原技能预算。
            actions, decision = state[None].copy(), 'confirmation_hold'
        else:
            remaining = self.spec['max_policy_steps'] - self.steps
            # 只消费预测的前10步，再用新观测判断技能完成，避免跨过技能监督边界。
            actions, decision = actions[:min(10, remaining, action_limit)].copy(), 'actor_chunk'
        self.steps += len(actions)
        return self._result(actions, completion_prob=probability, decision=decision)
