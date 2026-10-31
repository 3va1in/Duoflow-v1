"""Astra 在固定技能合同上规划；执行文本与阶段约束由本地控制器保留。"""
import json
import time

from astra_planner import AstraPlannerBackend, AstraPlannerError


def guided_schema(candidates):
    return {'type': 'object', 'properties': {
        'skill_id': {'type': 'string', 'enum': [c['skill_id'] for c in candidates]},
        'progress_status': {'type': 'string', 'enum': ['unknown', 'unsatisfied', 'satisfied']},
        'supporting_evidence': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'}},
        'required': ['skill_id', 'progress_status', 'supporting_evidence', 'reason'],
        'additionalProperties': False}


def parse_guided(response, candidates):
    if not isinstance(response, dict) or response.get('status') != 'completed':
        raise AstraPlannerError('Astra 决策没有完成')
    texts = []
    for message in response.get('output', []):
        if message.get('status') not in (None, 'completed'):
            raise AstraPlannerError('Astra 输出消息没有完成')
        for item in message.get('content', []):
            if item.get('type') == 'refusal':
                raise AstraPlannerError('Astra 拒绝本次规划')
            if item.get('type') == 'output_text':
                texts.append(item.get('text'))
    if len(texts) != 1:
        raise AstraPlannerError('Astra 必须返回唯一结构化决定')
    try:
        value = json.loads(texts[0])
    except (ValueError, TypeError) as error:
        raise AstraPlannerError('Astra 决策 JSON 无效') from error
    required = set(guided_schema(candidates)['required'])
    if not isinstance(value, dict) or set(value) != required:
        raise AstraPlannerError('Astra 决策字段不符合固定执行合同')
    if value['skill_id'] not in {c['skill_id'] for c in candidates}:
        raise AstraPlannerError('Astra 选择了当前不可执行的技能')
    if value['progress_status'] not in {'unknown', 'unsatisfied', 'satisfied'}:
        raise AstraPlannerError('非法阶段预测')
    evidence = value['supporting_evidence']
    if not isinstance(evidence, list) or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise AstraPlannerError('阶段证据必须是文本列表')
    if value['progress_status'] != 'unknown' and not evidence:
        raise AstraPlannerError('明确的阶段预测必须有公开观测证据')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise AstraPlannerError('决策原因不能为空')
    return value


class AstraGuidedBackend(AstraPlannerBackend):
    def rank(self, images, state, task, candidates, memory, history=None, state_present=True):
        if self._transport != 'siwc_files':
            raise AstraPlannerError('修复路线沿用本机 SIWC 文件桥')
        directory = self._next_audit()
        started = time.monotonic()
        payload, audit = self._request(images, state, task, candidates, memory, history, state_present)
        instructions = getattr(self, 'execution_guidance', None) or (
            'Guide a frozen bimanual VLA using the original task, public camera observations and qpos, '
            'the supplied ordered execution scaffold, eligible skill contracts, and frozen training memory. '
            'Select one eligible skill_id. Execution prompts and arm/object bindings are locked: do not '
            'invent, rewrite, insert or reorder micro-actions. The scaffold gives the intended multi-step '
            'task; eligible choices are its current step, legal successor or permitted recovery. '
            'A skill label is a language cue for a shared whole-task VLA, not a separately trained skill. '
            'That VLA may already have progressed beyond the named cue. Do not insist on executing each '
            'micro-action literally. Estimate the previous action result from public evidence as unknown, '
            'unsatisfied or satisfied; these are predictions, never ground truth. Motion or gripper '
            'closure alone does not prove grasp. Preserve all original task and arm constraints. '
            'When detailed subgoals are unreliable or conflict with observed progress, choose the eligible '
            'whole_task contract: it delegates remaining control to the same VLA with the unchanged original '
            'instruction, without resetting the environment or increasing the action budget. This is a '
            'normal execution choice, never a response to an API error. Do not repeatedly choose lift, '
            'align or release merely because completion is uncertain. Current/previous camera labels '
            'refer to real observations. Training outcomes describe past whole episodes, not skill labels '
            'or current success. No joint actions, executable code or terminal-success claim. Return JSON.'
        )
        for body in (payload, audit):
            body['instructions'] = instructions
            body['text']['format'] = {'type': 'json_schema', 'name': 'guided_skill_decision',
                                      'schema': guided_schema(candidates), 'strict': True}
        self._audit(directory, 'request.json', {'endpoint': self._url, 'body': audit})
        try:
            status, raw = self._file_response(payload, directory)
            response = json.loads(raw)
            self._audit(directory, 'response.json', {'http_status': status, 'raw_response': raw,
                        'usage': response.get('usage'), 'latency_seconds': time.monotonic() - started})
            if status != 200:
                raise AstraPlannerError(f'Astra 请求失败 HTTP {status}')
            return {**parse_guided(response, candidates), 'response_id': response.get('id'),
                    'model': response.get('model'), 'usage': response.get('usage'),
                    'latency_seconds': time.monotonic() - started,
                    'selection_distribution': 'api_response', 'selection_logprob': None,
                    'status_is_prediction': True}
        except Exception as error:
            self._audit(directory, 'error.json', {'type': type(error).__name__, 'message': str(error),
                        'latency_seconds': time.monotonic() - started})
            raise
