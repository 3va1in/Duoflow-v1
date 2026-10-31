"""Astra 的技能计划、公开观测反馈及回合内后缀修订。"""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time

from astra_planner import AstraPlannerBackend, AstraPlannerError, AstraAPIError
from skill_library import render_skill_prompt


def decision_schema(cards):
    binding = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'value': {'type': 'string'}},
               'required': ['name', 'value'], 'additionalProperties': False}
    node = {'type': 'object', 'properties': {
        'skill_id': {'type': 'string', 'enum': [card['skill_id'] for card in cards]},
        'bindings': {'type': 'array', 'items': binding},
        'expected_postcondition': {'type': 'string'}},
        'required': ['skill_id', 'bindings', 'expected_postcondition'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'subgoal_status': {'type': 'string', 'enum': ['satisfied', 'unsatisfied', 'unknown']},
        'supporting_evidence': {'type': 'array', 'items': {'type': 'string'}},
        'next_action': {'type': 'string', 'enum': ['initialize', 'continue', 'advance', 'revise']},
        'remaining_plan': {'type': 'array', 'items': node},
        'reason': {'type': 'string'}},
        'required': ['subgoal_status', 'supporting_evidence', 'next_action', 'remaining_plan', 'reason'],
        'additionalProperties': False}


def parse_decision(response, cards):
    if not isinstance(response, dict) or response.get('status') != 'completed':
        raise AstraPlannerError('Astra 规划响应未完成')
    texts = []
    for message in response.get('output', []):
        if message.get('status') not in (None, 'completed'):
            raise AstraPlannerError('Astra 规划消息未完成')
        for part in message.get('content', []):
            if part.get('type') == 'refusal':
                raise AstraPlannerError('Astra 拒绝本次规划')
            if part.get('type') == 'output_text':
                texts.append(part['text'])
    if len(texts) != 1:
        raise AstraPlannerError('Astra 必须返回唯一规划 JSON')
    value = json.loads(texts[0])
    if not isinstance(value, dict) or set(value) != set(decision_schema(cards)['required']):
        raise AstraPlannerError('Astra 规划字段不符合约定')
    if value['subgoal_status'] not in {'satisfied', 'unsatisfied', 'unknown'}:
        raise AstraPlannerError('非法子目标预测状态')
    if value['next_action'] not in {'initialize', 'continue', 'advance', 'revise'}:
        raise AstraPlannerError('非法计划操作')
    if not isinstance(value['supporting_evidence'], list) or not all(
            isinstance(item, str) and item.strip() for item in value['supporting_evidence']):
        raise AstraPlannerError('观测依据必须是文本列表')
    if value['subgoal_status'] != 'unknown' and not value['supporting_evidence']:
        raise AstraPlannerError('已判断的子目标状态必须提供公开观测依据')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise AstraPlannerError('规划必须给出原因')
    if not isinstance(value['remaining_plan'], list) or len(value['remaining_plan']) > 16:
        raise AstraPlannerError('有序计划最多16个节点')
    if value['next_action'] in {'initialize', 'revise'} and not value['remaining_plan']:
        raise AstraPlannerError('初始规划和修订必须提供非空后续计划')
    if value['next_action'] in {'continue', 'advance'} and value['remaining_plan']:
        raise AstraPlannerError('继续或推进不能暗中替换计划')
    by_id = {card['skill_id']: card for card in cards}
    for node in value['remaining_plan']:
        if not isinstance(node, dict) or set(node) != {'skill_id', 'bindings', 'expected_postcondition'}:
            raise AstraPlannerError('非法计划节点')
        if node['skill_id'] not in by_id:
            raise AstraPlannerError('计划引用检索集合外的技能')
        if not isinstance(node['bindings'], list):
            raise AstraPlannerError('技能绑定必须为名称值列表')
        names = []
        for item in node['bindings']:
            if not isinstance(item, dict) or set(item) != {'name', 'value'} or not all(
                    isinstance(item[key], str) and item[key].strip() for key in ('name', 'value')):
                raise AstraPlannerError('非法技能参数绑定')
            names.append(item['name'])
            parameter = by_id[node['skill_id']]['parameters'].get(item['name'])
            if parameter is None:
                raise AstraPlannerError('技能绑定包含未知参数')
            if parameter.startswith('enum[') and item['value'] not in parameter[5:-1].split(','):
                raise AstraPlannerError('技能枚举参数不在允许范围内')
        if len(names) != len(set(names)):
            raise AstraPlannerError('技能参数不能重复')
        required = set(re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', by_id[node['skill_id']]['prompt']))
        if not required.issubset(names):
            raise AstraPlannerError('技能提示所需参数不完整')
        if not isinstance(node['expected_postcondition'], str) or not node['expected_postcondition'].strip():
            raise AstraPlannerError('计划节点缺少可观察的预期后条件')
    return value


class AstraHierarchyBackend(AstraPlannerBackend):
    def plan_and_monitor(self, images, previous_images, state, instruction, cards, memories, context):
        if self._transport != 'siwc_files':
            raise AstraPlannerError('本轮层级规划仅使用已授权的本机文件桥')
        directory = self._next_audit()
        started = time.monotonic()
        try:
            frames = {'current_' + key: value for key, value in images.items()}
            frames.update({'previous_' + key: value for key, value in (previous_images or {}).items()})
            payload, audit = self._request(frames, state, instruction, cards, memories, [context], True)
            instructions = (
                'Plan and monitor a bimanual task using only the supplied public RGB, qpos, skill cards, '
                'frozen training examples and persistent harness rules. Current and previous camera labels '
                'identify observations before and after the last issued action chunk. Treat all scene progress '
                'as a prediction, never as simulator ground truth. Training outcomes label whole past episodes '
                'only. Skill cards are language contracts, not independently validated action policies. '
                'At initialization build a concise ordered plan of catalog skills with explicit parameter '
                'bindings and observable expected postconditions. You may choose, reorder or repeat retrieved '
                'skills; no reference recipe fixes their sequence. Preserve the original task and arm constraints. '
                'On later calls estimate the current subgoal as satisfied, unsatisfied or unknown, cite specific '
                'current/previous visual evidence, and choose continue, advance or revise. Gripper closure or '
                'motion alone is not evidence that an object is held. Do not advance merely because an action '
                'was attempted. initialize/revise must return a complete remaining plan starting from the next '
                'skill to execute; continue/advance must return remaining_plan=[]. A revision may replace the '
                'unresolved current node and future nodes, never executed history. Obey advance_gate and recovery '
                'rules; limits constrain this episode and do not authorize persistent edits. Bind enum values '
                'exactly as specified (for example left/right/auto, not descriptive sentences). Use unknown when '
                'occluded or ambiguous. Do not generate joint actions, Python, new skill IDs, simulator facts '
                'or terminal success. At the final node continue within the unchanged environment step budget; '
                'official evaluation alone ends the episode. Return the required JSON.'
            )
            for body in (payload, audit):
                body['instructions'] = instructions
                body['text']['format'] = {'type': 'json_schema', 'name': 'hierarchy_decision',
                                          'schema': decision_schema(cards), 'strict': True}
            self._audit(directory, 'request.json', {'endpoint': self._url, 'body': audit})
            status, raw = self._file_response(payload, directory)
            response = json.loads(raw)
            latency = time.monotonic() - started
            self._audit(directory, 'response.json', {'http_status': status, 'latency_seconds': latency,
                        'usage': response.get('usage'), 'raw_response': raw})
            if status != 200:
                raise AstraPlannerError(f'Astra HTTP {status}；不重试或回退')
            decision = parse_decision(response, cards)
            return {**decision, 'response_id': response.get('id'), 'model': response.get('model'),
                    'usage': response.get('usage'), 'latency_seconds': latency,
                    'mode': 'astra_hierarchy', 'status_is_prediction': True}
        except Exception as error:
            self._audit(directory, 'error.json', {'type': type(error).__name__, 'message': str(error),
                        'latency_seconds': time.monotonic() - started})
            if isinstance(error, AstraAPIError):
                raise
            raise AstraPlannerError(str(error)) from None


class HierarchyState:
    def __init__(self, library, harness):
        self.library = library
        self.harness = harness
        self.plan = []
        self.current_index = 0
        self.attempts = 0
        self.revisions = 0
        self.history = []
        self.node_sequence = 0

    def _nodes(self, items):
        nodes = []
        for item in items:
            bindings = {entry['name']: entry['value'] for entry in item['bindings']}
            prompt = render_skill_prompt(item['skill_id'], bindings, library=self.library)
            self.node_sequence += 1
            nodes.append({'skill_id': f"{item['skill_id']}__node_{self.node_sequence}",
                          'base_skill_id': item['skill_id'], 'bindings': bindings,
                          'execution_prompt': prompt, 'expected_postcondition': item['expected_postcondition']})
        return nodes

    def apply(self, decision):
        action = decision['next_action']
        status = decision['subgoal_status']
        before = self.export()
        blocked = None
        if not self.plan:
            if action != 'initialize':
                raise AstraPlannerError('第一条决定必须初始化计划')
            self.plan = self._nodes(decision['remaining_plan'])
        elif action == 'initialize':
            raise AstraPlannerError('不能重新初始化已有回合')
        elif action == 'advance':
            gate = next(rule['parameters'] for rule in self.harness['rules'] if rule['operation'] == 'advance_gate')
            if status not in gate['allowed_status'] or self.attempts < gate['min_attempts']:
                blocked = 'advance_gate'
            elif self.current_index + 1 >= len(self.plan):
                blocked = 'final_node_official_termination_only'
            else:
                self.current_index += 1
                self.attempts = 0
        elif action == 'revise':
            rule = next(rule['parameters'] for rule in self.harness['rules'] if rule['operation'] == 'recovery')
            if (status not in rule['trigger_status'] or self.attempts < rule['after_attempts']
                    or self.revisions >= rule['max_revisions']):
                blocked = 'recovery_budget_or_trigger'
            else:
                self.plan = self.plan[:self.current_index] + self._nodes(decision['remaining_plan'])
                self.attempts = 0
                self.revisions += 1
        self.history.append({'requested_action': action, 'effective_action': 'continue' if blocked else action,
                             'blocked_by': blocked, 'subgoal_status': status, 'status_is_prediction': True,
                             'supporting_evidence': decision['supporting_evidence'],
                             'plan_before': before['plan'], 'index_before': before['current_index']})
        chosen = dict(self.plan[self.current_index], plan_index=self.current_index)
        return chosen, self.history[-1]

    def mark_issued(self):
        self.attempts += 1

    def export(self):
        return {'plan': deepcopy(self.plan), 'current_index': self.current_index,
                'issued_chunks_current_node': self.attempts, 'revisions': self.revisions,
                'harness_version': self.harness['version'], 'progress_is_prediction': True,
                'history': deepcopy(self.history[-4:])}
