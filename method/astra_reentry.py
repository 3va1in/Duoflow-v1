"""短段委托后回到 Astra；技能完成预测只结束当前技能调用。"""
import copy
import json
from pathlib import Path

import numpy as np

from astra_guided import AstraGuidedBackend
from astra_planner import AstraPlannerError
from execution_outcomes import AstraAPIError
from atomic_skill_executor import SkillExecutor


class AstraReentryBackend(AstraGuidedBackend):
    execution_guidance = (
        'Control the bimanual task using only the original instruction, public RGB/qpos, supplied '
        'trained skill contracts, execution history and frozen training memory. Select one eligible '
        'skill_id. Atomic skills use a trained skill-conditioned action adapter and their exact isolated '
        'skill text; never rewrite their text, objects or arm bindings. Inspect each precondition before '
        'selection. A completion-head report is a prediction, not ground truth or terminal task success. '
        'You remain the high-level controller: skill execution returns after at most 60 action steps, '
        'predicted completion or timeout. Selecting the same unfinished skill resumes its cumulative '
        'budget. whole_task uses the original whole-task action adapter and unchanged task instruction '
        'for at most 90 action steps, then always returns fresh observations to you. It is not a permanent '
        'handoff and it never resets the environment or expands the original task action budget. '
        'You may continue whole_task or return to an appropriate trained skill after reviewing progress. '
        'The frontmost/middle/backmost bowl names identify the objects in the INITIAL images; track '
        'those identities after movement, do not rename them by current position. Left and right arm '
        'bindings must remain literal. Use previous-decision images and current images to assess the '
        'last segment as unknown, unsatisfied or satisfied, supported by visible evidence. An attempt, '
        'motion or gripper closure does not prove success. No simulator state, future outcomes, joint '
        'commands, new code or terminal-success claims. API errors interrupt execution and never select '
        'a fallback. Return only the required decision JSON.'
    )


class AstraNativeFallbackBackend(AstraReentryBackend):
    execution_guidance = AstraReentryBackend.execution_guidance.replace(
        'whole_task uses the original whole-task action adapter and unchanged task instruction ',
        'whole_task runs the full native local skill selector, runtime Harness, and original action '
        'adapter with the unchanged task instruction ')


def load_atomic_cards(path, task):
    source = json.loads(Path(path).read_text())
    cards, seen = [], set()
    for row in source['skills']:
        if row['task'] != task or row['id'] in seen:
            continue
        seen.add(row['id'])
        # 规划器只接收公开执行语义；checkpoint及其他本地存储元数据不进入请求。
        public_fields = ('id', 'condition_index', 'condition_key', 'skill_id', 'skill_text',
                         'bindings', 'task', 'max_policy_steps', 'precondition', 'stop_config',
                         'termination', 'automatic_next_skill', 'capability_status')
        card = {key: copy.deepcopy(row[key]) for key in public_fields if key in row}
        card.update(skill_id=row['id'], atomic_skill_id=row['skill_id'],
                    execution_prompt=row['skill_text'], executor='trained_atomic_actor',
                    completion_is_prediction=True, binding_updates_allowed=False)
        cards.append(card)
    if not cards:
        raise ValueError(f'该任务没有训练过的技能: {task}')
    return cards


class ReentryController:
    def __init__(self, actors, planner, cards, instruction, max_steps, log,
                 whole_window=90, skill_window=60, call_limit=120, task_name=None,
                 native_fallback=None):
        self.actors, self.planner = actors, planner
        self.cards = copy.deepcopy(cards)
        self.instruction, self.max_steps, self.log = instruction, int(max_steps), log
        self.task_name = task_name or self.cards[0].get('task')
        self.whole_window, self.skill_window = int(whole_window), int(skill_window)
        self.call_limit = int(call_limit)
        if min(self.max_steps, self.whole_window, self.skill_window, self.call_limit) <= 0:
            raise ValueError('执行窗口、回合预算和调用上限必须为正数')
        self.whole_card = {'skill_id': 'whole_task', 'execution_prompt': instruction,
                           'executor': 'original_whole_task_actor', 'segment_steps': self.whole_window,
                           'returns_to_astra': True, 'environment_step_budget_delta': 0}
        self.total_steps = self.segment_steps = self.calls = self.observation_index = 0
        self.active = None
        self.executors, self.used_steps = {}, {}
        self.history, self.last_result = [], None
        self.initial_images = self.previous_images = None
        self.previous_qpos = None
        self.review_due = True
        self.native_fallback = native_fallback
        self.blocked_skills = set()
        self.missing_contracts = []
        if native_fallback is not None:
            from native_fallback import missing_execution_contracts
            self.missing_contracts = missing_execution_contracts(self.cards, self.task_name)
            self.whole_card['executor'] = 'full_native_controller'
            self.whole_card['native_components'] = ['local_skill_selector', 'runtime_rsi',
                                                   'independent_plan_state', 'original_action_lora']

    def export(self):
        return {'execution_contract': 'reentry_atomic_v1', 'issued_policy_steps': self.total_steps,
                'max_policy_steps': self.max_steps, 'planner_calls': self.calls,
                'active_skill_id': self.active, 'segment_steps': self.segment_steps,
                'skill_used_steps': dict(self.used_steps), 'last_execution': self.last_result,
                'verified_complete': False, 'environment_step_budget_delta': 0,
                'native_fallback': self.native_fallback is not None,
                'missing_execution_contracts': self.missing_contracts,
                'blocked_atomic_skills': sorted(self.blocked_skills)}

    def _select(self, state, images, memories, excluded):
        if self.calls >= self.call_limit:
            raise AstraAPIError('upstream_call_budget_exhausted')
        candidates = [dict(c, consumed_policy_steps=self.used_steps.get(c['skill_id'], 0))
                      for c in self.cards if c['skill_id'] not in excluded | self.blocked_skills and
                      self.used_steps.get(c['skill_id'], 0) < c['max_policy_steps']]
        candidates.append(self.whole_card)
        planning_images = {'current_' + k: v for k, v in images.items()}
        planning_images.update({'initial_' + k: v for k, v in self.initial_images.items()})
        planning_images.update({'previous_decision_' + k: v
                                for k, v in (self.previous_images or {}).items()})
        control = self.export()
        control.update(current_observation_id=self.observation_index,
                       completion_feedback_scope='predicted_current_skill_only')
        context = {'mode': 'guided_execution', 'task_name': self.task_name,
                   'progress_is_prediction': True, 'terminal_feedback_available': False,
                   'previous_qpos': self.previous_qpos, 'execution_control': control,
                   'execution_catalog': self.cards + [self.whole_card]}
        self.calls += 1
        # API 异常直接传播；没有重试或原 actor 兜底。
        decision = self.planner.rank(planning_images, state, self.instruction, candidates,
                                     memories, history=[context] + self.history[-6:])
        chosen = next((c for c in candidates if c['skill_id'] == decision['skill_id']), None)
        if chosen is None:
            raise AstraPlannerError('Astra 返回了训练技能库之外的绑定')
        if self.native_fallback is not None and chosen['skill_id'] != 'whole_task':
            self.native_fallback.suspend(state, images)
        previous = self.active
        self.active, self.segment_steps, self.review_due = chosen['skill_id'], 0, False
        if self.active != 'whole_task':
            executor = self.executors.get(self.active)
            if executor is None or executor.status != 'running':
                executor = SkillExecutor(self.actors, self.actors.threshold)
                executor.start({'skill_id': chosen['atomic_skill_id'],
                                'skill_text': chosen['execution_prompt'],
                                'max_policy_steps': chosen['max_policy_steps'] -
                                                    self.used_steps.get(self.active, 0)})
                self.executors[self.active] = executor
                self.log({'event': 'atomic_skill_start', 'selected_skill_id': self.active,
                          'atomic_skill_id': chosen['atomic_skill_id'],
                          'remaining_policy_steps': executor.spec['max_policy_steps'],
                          'previously_consumed_policy_steps': self.used_steps.get(self.active, 0)})
            if previous != self.active:
                executor.first_prediction = True
        self.previous_images = {k: v.copy() for k, v in images.items()}
        self.previous_qpos = np.asarray(state, dtype=np.float32).tolist()
        record = {'event': 'astra_reentry_decision', 'decision_index': self.calls - 1,
                  'selected_skill_id': self.active, 'execution_prompt': chosen['execution_prompt'],
                  'decision': decision, 'execution_control': control}
        self.history.append({'skill_id': self.active, 'execution_prompt': chosen['execution_prompt'],
                             'plan_index': None, 'outcome': 'unconfirmed'})
        self.log(record)

    def _delegate_native(self, reason):
        self.active, self.segment_steps, self.review_due = 'whole_task', 0, False
        self.log({'event': 'native_handoff', 'reason': reason,
                  'missing_execution_contracts': self.missing_contracts,
                  'before_first_action': self.total_steps == 0,
                  'returns_to_astra': not bool(self.missing_contracts),
                  'environment_reset': False, 'environment_step_budget_delta': 0})

    def step(self, state, images, memories):
        if self.total_steps >= self.max_steps:
            raise RuntimeError('环境在原动作预算耗尽后继续请求动作')
        self.observation_index += 1
        if self.initial_images is None:
            self.initial_images = {k: v.copy() for k, v in images.items()}
        excluded = set()
        while True:
            if self.review_due or self.active is None:
                if self.native_fallback is not None and self.missing_contracts:
                    self._delegate_native('incomplete_task_skill_chain')
                else:
                    self._select(state, images, memories, excluded)
            remaining = self.max_steps - self.total_steps
            if self.active == 'whole_task':
                actions = (self.native_fallback.step(state, images, self.instruction)
                           if self.native_fallback is not None else
                           self.actors.whole(state, images, self.instruction))
                actions = actions[:min(30, remaining, self.whole_window - self.segment_steps)]
                result = {'status': 'running', 'decision': ('native_controller_chunk'
                          if self.native_fallback is not None else 'whole_task_chunk'),
                          'completion_prob': None}
                window = self.whole_window
            else:
                executor = self.executors[self.active]
                result = executor.step({'state': state, 'images': images,
                                        'observation_id': self.observation_index},
                                       action_limit=min(10, remaining,
                                                        self.skill_window - self.segment_steps))
                actions = result['actions'][:remaining]
                window = self.skill_window
                if not len(actions):
                    self.last_result = {k: v for k, v in result.items() if k != 'actions'}
                    self.last_result.update(skill_id=self.active,
                                            cumulative_policy_steps=self.used_steps.get(self.active, 0))
                    self.log({'event': 'atomic_skill_return', **self.last_result})
                    if self.native_fallback is not None and (
                            result['status'] == 'timeout' or
                            (result['status'] == 'predicted_complete' and
                             self.used_steps.get(self.active, 0) <= 30)):
                        self.blocked_skills.add(self.active)
                        self._delegate_native('atomic_timeout' if result['status'] == 'timeout'
                                              else 'early_completion_prediction')
                        continue
                    excluded.add(self.active)
                    self.review_due = True
                    continue
                self.used_steps[self.active] = self.used_steps.get(self.active, 0) + len(actions)
            if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                raise ValueError('执行器必须返回有限的 [H,14] 动作')
            self.total_steps += len(actions)
            self.segment_steps += len(actions)
            self.last_result = {k: v for k, v in result.items() if k != 'actions'}
            self.last_result.update(skill_id=self.active, issued_actions=len(actions),
                                    cumulative_policy_steps=self.used_steps.get(self.active),
                                    segment_steps=self.segment_steps)
            self.review_due = self.segment_steps >= window
            self.log({'event': 'reentry_execution', **self.last_result,
                      'execution_control': self.export(),
                      'returns_to_astra': self.review_due and not bool(self.missing_contracts)})
            return actions
