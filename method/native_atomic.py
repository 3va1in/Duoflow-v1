"""冻结本地选择器配合已训练技能阶段；末段委托单独记账。"""
import copy

import numpy as np

from atomic_skill_executor import SkillExecutor
from plan_state import PlanState
from runtime_editor import safe_context
from transition_evidence import build_transition_evidence


STAGES = {
    'handover_mic': ('pick_and_lift', 'handover_object'),
    'lift_pot': ('bimanual_grasp', 'bimanual_lift'),
    'pick_dual_bottles': ('grasp_and_lift_bottles', 'bring_bottles_to_center'),
    'press_stapler': ('prepare_stapler_press', 'press_stapler_top'),
    'adjust_bottle': ('pick_and_lift', 'orient_and_hold'),
    'beat_block_hammer': ('pick_and_lift', 'strike_block'),
    'click_bell': ('approach_bell', 'press_bell', 'withdraw_from_bell'),
    'rotate_qrcode': ('pick_and_lift', 'reorient_and_place'),
}


class NativeAtomicController:
    def __init__(self, actors, planner, cards, instruction, task, condition,
                 max_steps, log, patch=None, runtime_editor=None, stage_spec=None):
        if task not in STAGES and stage_spec is None:
            raise ValueError('该任务尚未声明 native 原子阶段链')
        self.actors, self.planner, self.log = actors, planner, log
        self.instruction, self.task, self.condition = instruction, task, condition
        self.max_steps, self.runtime_editor = int(max_steps), runtime_editor
        if self.max_steps <= 0:
            raise ValueError('动作预算必须为正数')
        self.stage_defs = (copy.deepcopy(stage_spec['stages']) if stage_spec is not None
                           else [{'skill_id': skill} for skill in STAGES[task]])
        if not self.stage_defs:
            raise ValueError('技能阶段链不能为空')
        self.stages = []
        for definition in self.stage_defs:
            skill = definition['skill_id']
            bindings = definition.get('bindings', {})
            stage = [copy.deepcopy(c) for c in cards if c['atomic_skill_id'] == skill
                     and all(c['bindings'].get(k) == v for k, v in bindings.items())]
            if not stage or not all(actors.supports(c) for c in stage):
                raise ValueError(f'缺少可执行的训练条件: {task}/{skill}')
            self.stages.append(stage)
        plan = [dict(skill_id=f'native_atomic_stage_{i}', base_skill_id=skill,
                     execution_prompt=skill.replace('_', ' '), plan_index=i)
                for i, skill in enumerate(d['skill_id'] for d in self.stage_defs)]
        self.plan_state = PlanState(plan, patch=patch)
        self.total_steps = self.observation_id = self.calls = 0
        self.active = self.executor = self.last_phase = None
        self.completed = set()
        self.arm = None
        self.object_arms = {}
        self.whole_reason = None
        self.previous_observation = self.previous_state = None
        self.history = []
        self.log({'event': 'native_atomic_plan_created', 'plan': plan,
                  'stage_source': 'declared_trained_skill_chain', 'api_calls': 0,
                  'completion_is_prediction': True})

    def export(self):
        return dict(execution_contract='native_atomic_v1', issued_policy_steps=self.total_steps,
                    max_policy_steps=self.max_steps, native_selector_calls=self.calls,
                    api_calls=0, completed_stage_predictions=sorted(self.completed),
                    whole_task_reason=self.whole_reason, plan_state=self.plan_state.export(),
                    verified_complete=False, environment_step_budget_delta=0)

    def _delegate(self, reason):
        self.whole_reason = reason
        self.log({'event': 'native_atomic_handoff', 'reason': reason,
                  'executor': 'original_whole_task_actor', 'environment_reset': False,
                  'environment_step_budget_delta': 0})

    def _issue(self, actions, state, observation, *, mode, result=None):
        actions = np.asarray(actions, dtype=np.float32)[:self.max_steps - self.total_steps]
        if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError('执行器必须下发非空有限 [H,14] 动作')
        self.total_steps += len(actions)
        self.last_phase = self.plan_state.current_index if mode != 'whole_task' else None
        self.previous_observation = copy.deepcopy(observation)
        self.previous_state = np.asarray(state, dtype=np.float32).copy()
        self.log({'event': 'native_atomic_execution', 'execution_mode': mode,
                  'selected_skill_id': self.active['skill_id'] if mode == 'atomic' else mode,
                  'issued_actions': len(actions), 'total_policy_steps': self.total_steps,
                  **{k: v for k, v in (result or {}).items() if k != 'actions'}})
        return actions

    def _stage_candidates(self, phase):
        candidates = [dict(c, plan_index=phase) for c in self.stages[phase]]
        relation = self.stage_defs[phase].get('arm_relation')
        if relation is not None:
            if relation == 'same_object':
                candidates = [c for c in candidates if self.object_arms.get(c['bindings'].get('object'))
                              in (None, c['bindings']['arm'])]
            elif relation == 'same' and self.arm is not None:
                candidates = [c for c in candidates if c['bindings']['arm'] == self.arm]
            elif relation == 'opposite' and self.arm is not None:
                target = 'right' if self.arm == 'left' else 'left'
                candidates = [c for c in candidates if c['bindings']['arm'] == target]
            elif relation in {'left', 'right', 'both'}:
                candidates = [c for c in candidates if c['bindings']['arm'] == relation]
        if phase > 0 and self.arm is not None and self.task in {
                'adjust_bottle', 'beat_block_hammer', 'click_bell', 'rotate_qrcode'}:
            candidates = [c for c in candidates if c['bindings']['arm'] == self.arm]
        if phase == 1 and self.arm is not None:
            target = ('right' if self.arm == 'left' else 'left') if self.task == 'handover_mic' else self.arm
            if self.task in {'handover_mic', 'press_stapler'}:
                candidates = [c for c in candidates if c['bindings']['arm'] == target]
        if not candidates:
            raise ValueError('已选择的手臂没有对应后续训练技能')
        return candidates

    def step(self, state, images, memories):
        if self.total_steps >= self.max_steps:
            raise RuntimeError('环境在原任务动作预算耗尽后仍请求动作')
        self.observation_id += 1
        observation = {'vision': {key: {'color': value} for key, value in images.items()}}
        transition = None
        if self.previous_observation is not None:
            transition = build_transition_evidence(before=self.previous_observation, after=observation,
                before_qpos=self.previous_state, after_qpos=state)
        if self.last_phase == self.plan_state.current_index:
            self.plan_state.mark_executed(self.last_phase, transition_evidence=transition)
        self.last_phase = None
        while True:
            if self.whole_reason is not None:
                actions = self.actors.whole(state, images, self.instruction)[:30]
                return self._issue(actions, state, observation, mode='whole_task')
            if self.executor is not None:
                result = self.executor.step({'state': state, 'images': images,
                    'observation_id': self.observation_id}, action_limit=min(10, self.max_steps - self.total_steps))
                if len(result['actions']):
                    return self._issue(result['actions'], state, observation, mode='atomic', result=result)
                phase = self.plan_state.current_index
                self.log({'event': 'atomic_skill_return', 'selected_skill_id': self.active['skill_id'],
                          'plan_index': phase, **{k: v for k, v in result.items() if k != 'actions'}})
                self.executor = None
                if result['status'] == 'timeout':
                    self._delegate('atomic_timeout')
                    continue
                if result['status'] != 'predicted_complete':
                    raise ValueError('未知技能终止状态')
                self.completed.add(phase)
                if phase == len(self.stages) - 1:
                    self._delegate('all_stages_predicted_complete')
                    continue
            if self.runtime_editor is not None:
                context = safe_context(self.task, self.condition, self.plan_state.export(), transition)
                edit = self.runtime_editor.select(context)
                self.plan_state.apply_harness_patch(edit['patch'])
                self.log({'event': 'runtime_edit', 'context': context, **edit,
                          'evaluation_updates_blocked': True})
            eligible = self.plan_state.eligible()
            phases = [s['plan_index'] for s in eligible if s['plan_index'] not in self.completed]
            if not phases:
                # 预测完成不能绕过原 Harness 最小尝试数；保持动作仍占原预算。
                return self._issue(np.asarray(state)[None], state, observation, mode='guard_hold')
            phase = min(phases)
            candidates = self._stage_candidates(phase)
            # 共享骨干刚执行过技能 LoRA；本地评分必须恢复其训练时的原 LoRA。
            self.actors._activate('whole')
            selected = self.planner.rank(images, state, self.instruction, candidates, memories,
                history=[{'ordered_plan': self.plan_state.plan,
                          'eligible_plan_indices': [phase], 'progress_is_prediction': True}] + self.history[-4:])
            if hasattr(selected, '__dict__'):
                selected = vars(selected)
            chosen = next((c for c in candidates if c['skill_id'] == selected['skill_id']), None)
            if chosen is None:
                raise ValueError('本地选择器返回候选外技能')
            self.plan_state.choose(phase)
            self.active = chosen
            if phase == 0 and chosen.get('bindings', {}).get('arm') in {'left', 'right'}:
                self.arm = chosen['bindings']['arm']
            if chosen.get('bindings', {}).get('arm') in {'left', 'right'}:
                self.object_arms[chosen['bindings'].get('object')] = chosen['bindings']['arm']
            secondary = chosen.get('bindings', {}).get('secondary_object')
            secondary_arm = chosen.get('bindings', {}).get('secondary_arm')
            if secondary and secondary_arm in {'left', 'right'}:
                self.object_arms[secondary] = secondary_arm
            self.executor = SkillExecutor(self.actors, self.actors.threshold)
            self.executor.start({'skill_id': chosen['atomic_skill_id'],
                'skill_text': chosen['execution_prompt'], 'max_policy_steps': chosen['max_policy_steps']})
            self.calls += 1
            self.history.append({'skill_id': chosen['skill_id'], 'execution_prompt': chosen['execution_prompt'],
                                 'plan_index': phase, 'outcome': 'unconfirmed'})
            self.log({'event': 'native_atomic_decision', 'selected_skill_id': chosen['skill_id'],
                      'atomic_skill_id': chosen['atomic_skill_id'], 'execution_prompt': chosen['execution_prompt'],
                      'plan_index': phase, 'planner_output': selected, 'api_calls': 0})
            self.log({'event': 'atomic_skill_start', 'selected_skill_id': chosen['skill_id'],
                      'atomic_skill_id': chosen['atomic_skill_id'], 'plan_index': phase,
                      'max_policy_steps': chosen['max_policy_steps']})
