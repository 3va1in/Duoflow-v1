"""将候选技能选择器接到冻结的 DuoFlow；仅接收公共观测。"""
from pathlib import Path
import json
import os
import time
from typing import Any, Mapping

import numpy as np
import torch

from XPolicyLab.policy.TwinTimeFM.model import Model as DuoFlowModel
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state
from experience_memory import MemoryStore
from skill_library import load_library, build_execution_plan, retrieve_skill_cards
from plan_state import PlanState, normalize_patch
from transition_evidence import build_transition_evidence
from system2_backend import System2Backend
from native_factory import attach_native_preprocess
from actor_conditioning import compose_instruction, verify_actor_text
from runtime_editor import RuntimeEditPolicy, safe_context


class Model(DuoFlowModel):
    def __init__(self, model_cfg, engine=None):
        self.run_dir = Path(model_cfg['run_dir'])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.split = model_cfg['split']
        self.variant = model_cfg.get('variant', 'combined')
        self.planner_backend = model_cfg.get('planner_backend', 'native')
        if self.planner_backend not in {'native', 'gpt_6_astra', 'astra_hierarchy', 'astra_guided'}:
            raise ValueError(f'unsupported planner_backend: {self.planner_backend!r}')
        if self.planner_backend in {'gpt_6_astra', 'astra_guided'} and (
            self.split != 'eval' or self.variant == 'baseline'
        ):
            raise ValueError('GPT-6 Astra 编排器仅支持独立 combined eval，不进入原特征训练流程')
        self.execution_contract = model_cfg.get('execution_contract',
            'bounded_v2' if self.planner_backend == 'astra_guided' else 'legacy')
        self.native_fallback_enabled = bool(model_cfg.get('native_fallback', False))
        self.native_stage_spec_path = model_cfg.get('native_stage_spec_path')
        if self.native_fallback_enabled and self.execution_contract != 'reentry_atomic_v1':
            raise ValueError('完整 native 回退只用于独立 reentry_atomic_v1 路线')
        if self.execution_contract not in {'legacy', 'bounded_v2', 'reentry_atomic_v1', 'native_atomic_v1'}:
            raise ValueError('未知执行合同')
        if self.execution_contract in {'bounded_v2', 'reentry_atomic_v1', 'native_atomic_v1'} and self.split != 'eval':
            raise ValueError('修复版执行合同只用于冻结评测')
        if self.planner_backend == 'astra_guided' and self.execution_contract not in {'bounded_v2', 'reentry_atomic_v1'}:
            raise ValueError('修复后的 Astra 必须使用 bounded_v2 执行合同')
        if self.execution_contract == 'reentry_atomic_v1' and self.planner_backend != 'astra_guided':
            raise ValueError('可返回规划的技能执行只接受 astra_guided')
        if self.execution_contract == 'native_atomic_v1' and self.planner_backend != 'native':
            raise ValueError('native 原子技能路线只使用本地选择器')
        if self.planner_backend == 'astra_hierarchy':
            if self.split != 'eval' or self.variant == 'baseline':
                raise ValueError('Astra 层级规划只接受独立冻结 eval；开发阶段用 experiment_phase 区分')
            self.harness_rules_path = Path(model_cfg['harness_rules_path'])
            self.hierarchy_max_steps = int(model_cfg['hierarchy_max_steps'])
            self.hierarchy_chunk_steps = int(model_cfg.get('hierarchy_chunk_steps', 30))
            if self.hierarchy_max_steps <= 0 or self.hierarchy_chunk_steps != 30:
                raise ValueError('层级规划须沿用任务步数上限和每30步动作块')
            self.hierarchy_call_limit = (self.hierarchy_max_steps + 29) // 30 + 1
            self.hierarchy_retrieval_limit = int(model_cfg.get('hierarchy_retrieval_limit', 12))
            self.experiment_phase = model_cfg.get('experiment_phase', 'heldout_test')
        self.condition = model_cfg['condition']
        self.harness_variant = str(model_cfg.get('harness_variant', 'fixed'))
        if self.harness_variant not in {'fixed', 'adaptive', 'rsi'}:
            raise ValueError(f'unsupported harness_variant: {self.harness_variant!r}')
        patch = model_cfg.get('harness_patch', {'op': 'NO_CHANGE'})
        if patch is None:
            patch = {'op': 'NO_CHANGE'}
        if not isinstance(patch, Mapping):
            raise ValueError('harness_patch must be a JSON object')
        self.harness_patch = self._canonical_patch(dict(patch))
        self.editor_version = str(model_cfg.get('editor_version', 'bootstrap'))
        editor_path = model_cfg.get('runtime_editor_checkpoint')
        self.runtime_editor = (RuntimeEditPolicy.load(editor_path)
                               if editor_path and self.planner_backend != 'astra_hierarchy' and
                               self.execution_contract != 'reentry_atomic_v1' else None)
        self.memory = MemoryStore(model_cfg['memory_dir'], allowed_splits=('train',))
        self.library = load_library()
        self.model_version = ('gpt-6-astra' if self.planner_backend in {'gpt_6_astra', 'astra_hierarchy', 'astra_guided'}
                              else str(model_cfg.get('planner_checkpoint', 'baseline')))
        super().__init__(model_cfg, engine=engine)
        self.planner = None
        self._pending_harness_event = None
        self._last_public_observation = None
        self._last_public_state = None
        self._last_issued_plan_index = None
        self._rsi_context_written = False
        if self.variant != 'baseline':
            native = attach_native_preprocess(self.engine.model)
            if self.planner_backend in {'gpt_6_astra', 'astra_hierarchy', 'astra_guided'}:
                from astra_planner import AstraPlannerBackend
                from astra_hierarchy import AstraHierarchyBackend
                from astra_guided import AstraGuidedBackend
                planner_class = {'astra_hierarchy': AstraHierarchyBackend,
                                 'astra_guided': AstraGuidedBackend}.get(self.planner_backend, AstraPlannerBackend)
                if self.execution_contract == 'reentry_atomic_v1':
                    from astra_reentry import AstraReentryBackend
                    planner_class = AstraReentryBackend
                    if self.native_fallback_enabled:
                        from astra_reentry import AstraNativeFallbackBackend
                        planner_class = AstraNativeFallbackBackend
                self.planner = planner_class(
                    credential_file=model_cfg.get('astra_credential_file'),
                    base_url=model_cfg.get('astra_base_url'),
                    audit_dir=self.run_dir / 'astra_requests',
                    transport=model_cfg.get('astra_transport', 'api'),
                    request_root=model_cfg.get('astra_request_root'),
                )
            else:
                checkpoint = Path(model_cfg['planner_checkpoint'])
                if not checkpoint.is_file():
                    raise FileNotFoundError(f'技能选择器尚未训练：{checkpoint}')
                self.planner = System2Backend.from_native(native, checkpoint=str(checkpoint))
            self._install_actor_audit(native.actor_prompt_transform)
        if self.execution_contract in {'reentry_atomic_v1', 'native_atomic_v1'}:
            from reentry_actors import SharedAtomicActors
            self._atomic_actors = SharedAtomicActors(
                self.engine, model_cfg['atomic_policy_root'], model_cfg['atomic_actor_checkpoint'],
                model_cfg['atomic_completion_checkpoint'], self._reentry_actor_text)
            self._atomic_catalog_path = model_cfg['atomic_skill_catalog']
            self._reentry_max_steps = int(model_cfg['reentry_max_steps'])
            self._reentry_call_limit = int(model_cfg.get('reentry_planner_call_limit', 120))
            if self.native_fallback_enabled:
                from native_fallback import NativeFallback
                checkpoint = Path(model_cfg['planner_checkpoint'])
                if not checkpoint.is_file() or not editor_path:
                    raise ValueError('完整 native 回退要求原 selector 与 runtime editor')
                self._native_fallback = NativeFallback(self,
                    System2Backend.from_native(native, checkpoint=str(checkpoint)),
                    RuntimeEditPolicy.load(editor_path), checkpoint)

    @staticmethod
    def _observation_evidence(obs: Mapping[str, Any] | None, *, source: str) -> dict[str, Any]:
        """Keep a compact, public-observation audit record without copying RGB."""

        if not isinstance(obs, Mapping):
            return {'available': False, 'source': source, 'reason': 'observation_missing'}

        def array_meta(value: Any) -> dict[str, Any]:
            try:
                array = np.asarray(value)
                meta: dict[str, Any] = {
                    'shape': list(array.shape),
                    'dtype': str(array.dtype),
                    'finite': bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
                }
                return meta
            except Exception as error:
                return {'available': False, 'error': f'{type(error).__name__}: {error}'}

        vision = obs.get('vision')
        vision_meta = {}
        if isinstance(vision, Mapping):
            for name, frame in vision.items():
                if isinstance(frame, Mapping) and 'color' in frame:
                    frame = frame['color']
                vision_meta[str(name)] = array_meta(frame)
        state = obs.get('state', obs.get('robot_state', obs.get('qpos')))
        return {
            'available': True,
            'source': source,
            'vision': vision_meta,
            'state': array_meta(state) if state is not None else None,
            'instruction_present': isinstance(obs.get('instruction', obs.get('instructions')), (str, list)),
        }

    def _log_harness_post_observation(self, obs: Mapping[str, Any] | None) -> None:
        pending = self._pending_harness_event
        if pending is None:
            return
        self._log({
            'event': 'harness_post_observation',
            'episode_id': self.episode_id,
            'patch_id': pending['patch_id'],
            'decision_index': pending['decision_index'],
            # This is the next real policy observation.  It is not labelled
            # as a simulator success or a verified skill completion.
            'post_observation': self._observation_evidence(
                obs, source='next_policy_observation'
            ),
            'post_observation_status': 'available' if isinstance(obs, Mapping) else 'missing',
        })
        self._pending_harness_event = None

    @staticmethod
    def _canonical_patch(patch: Mapping[str, Any]) -> dict[str, Any]:
        """Accept controller aliases while constructing the fixed PlanState patch."""

        if 'patch_id' in patch:
            return normalize_patch(patch)
        operation = str(patch.get('operation', patch.get('op', 'NO_CHANGE'))).upper()
        if operation == 'NO_CHANGE':
            return normalize_patch({'patch_id': 'no_change', 'operation': operation})
        if operation == 'SET_GUARD':
            attempts = int(patch.get('min_attempts', 2))
            patch_id = 'guard_3' if attempts >= 3 else 'guard_2'
            return normalize_patch({
                'patch_id': patch_id,
                'operation': operation,
                'min_attempts': attempts,
                'stagnation_chunks': 3 if patch_id == 'guard_3' else 2,
            })
        if operation in {'RETRY_OR_BACKTRACK', 'BACKTRACK'}:
            return normalize_patch({'patch_id': 'backtrack_1', 'operation': 'RETRY_OR_BACKTRACK'})
        raise ValueError(f'unsupported harness patch operation: {operation!r}')

    def _log_harness_transition(
        self,
        obs: Mapping[str, Any],
        before_plan: Mapping[str, Any],
        after_plan: Mapping[str, Any],
        evidence: Mapping[str, Any] | None,
    ) -> str:
        patch_id = str(self.plan_state.patch['patch_id'])
        patch_id = f'{self.episode_id}:patch:{len(self.decisions)}:{patch_id}'
        self._log({
            'event': 'harness_transition',
            'episode_id': self.episode_id,
            'patch_id': patch_id,
            'decision_index': len(self.decisions),
            'harness_variant': self.harness_variant,
            'editor_version': self.editor_version,
            'harness_patch': self.plan_state.patch,
            'patch_applied': self.plan_state.patch['patch_id'] != 'no_change',
            'observation_before': self._observation_evidence(obs, source='current_policy_observation'),
            'transition_evidence': dict(evidence) if isinstance(evidence, Mapping) else None,
            'plan_before': before_plan,
            'plan_after': after_plan,
            'post_observation_status': 'pending_next_policy_observation',
        })
        self._pending_harness_event = {
            'patch_id': patch_id,
            'decision_index': len(self.decisions),
        }
        return patch_id

    def _write_rsi_context(self, *, candidates: list[dict[str, Any]], features: Any, patch_id: str) -> str | None:
        if self._rsi_context_written:
            return None
        if self.planner is None:
            raise RuntimeError('无法写入 RSI 上下文：当前回合没有 System2 planner')
        if features is None:
            raise RuntimeError(
                '无法写入 RSI 上下文：System2Backend.last_candidate_features 为空'
            )
        try:
            tensor = torch.as_tensor(features).detach().cpu().float()
        except Exception as error:
            raise RuntimeError(
                '无法读取 System2Backend.last_candidate_features：'
                f'{type(error).__name__}: {error}'
            ) from error
        expected_dim = int(getattr(self.planner, 'feature_dim', 2304))
        if tensor.ndim != 2:
            raise RuntimeError(
                'System2Backend.last_candidate_features 必须是二维张量 '
                f'[num_candidates, {expected_dim}]，实际维度={tuple(tensor.shape)}'
            )
        if tensor.shape[0] != len(candidates):
            raise RuntimeError(
                'System2Backend.last_candidate_features 的候选数不匹配：'
                f'features={tensor.shape[0]}, candidates={len(candidates)}'
            )
        if tensor.shape[1] != expected_dim:
            raise RuntimeError(
                'System2Backend.last_candidate_features 的特征维度不匹配：'
                f'expected={expected_dim}, actual={tensor.shape[1]}'
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError('System2Backend.last_candidate_features 含有非有限值')
        shared = tensor.mean(dim=0)
        path = self.run_dir / 'rsi_context.json'
        try:
            path.write_text(json.dumps({
                'schema_version': 'harness_rsi_context_v1',
                'source': 'first_decision_frozen_vlm',
                'shared_feature': shared.tolist(),
                'shared_feature_dim': int(shared.numel()),
                'candidate_ids': [str(item['skill_id']) for item in candidates],
                'feature_shape': list(tensor.shape),
                'task': self.case.get('task_name'),
                'condition': self.condition,
                'seed': self.case.get('seed'),
                'episode_id': self.episode_id,
                'harness_variant': self.harness_variant,
                'editor_version': self.editor_version,
                'patch_id': patch_id,
            }, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        except Exception as error:
            raise RuntimeError(
                f'无法写入 RSI 上下文 {path}: {type(error).__name__}: {error}'
            ) from error
        self._rsi_context_written = True
        return str(path)

    @staticmethod
    def _snapshot_public_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
        """Copy only RGB data needed for the next public transition check."""

        vision = obs.get('vision')
        if not isinstance(vision, Mapping):
            return {'vision': {}}
        snapshot: dict[str, Any] = {'vision': {}}
        for name, frame in vision.items():
            if isinstance(frame, Mapping) and 'color' in frame:
                snapshot['vision'][name] = {'color': np.asarray(frame['color']).copy()}
            else:
                snapshot['vision'][name] = np.asarray(frame).copy()
        return snapshot

    def _install_actor_audit(self, transform):
        original = self.engine.model.sample_actions
        tokenizer = transform.paligemma_tokenizer

        def audited_sample_actions(*args, **kwargs):
            # 检查实际交给动作生成器的 tokens，避免仅验证上游字符串。
            token_ids = kwargs['lang_tokens'][0]
            mask = kwargs['lang_masks'][0].bool()
            ids = token_ids[mask].detach().cpu().tolist()
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            policy = getattr(self, '_active_actor_policy', self)
            pending = policy._pending_actor_text
            if pending.get('execution_mode') == 'atomic':
                checks = verify_actor_text(decoded, pending['execution_prompt'], '', execution_mode='whole_task')
            else:
                checks = verify_actor_text(decoded, pending['instruction'], pending['execution_prompt'],
                                           execution_mode=pending.get('execution_mode', 'skill'))
            policy._log({'event': 'actor_conditioning', 'episode_id': policy.episode_id,
                       'step_index': len(policy.decisions) - 1, 'token_ids': ids,
                       'decoded': decoded, 'checks': checks, **pending})
            if not all(checks.values()):
                raise RuntimeError('实际动作输入丢失任务或技能文本，停止该尝试')
            return original(*args, **kwargs)

        self.engine.model.sample_actions = audited_sample_actions

    def reset(self):
        # RoboTwin's official loop calls prepare_case(...) and immediately
        # calls reset().  Preserve that just-received task/seed for this one
        # reset, while still clearing native inference and decision history.
        preserve_case = bool(getattr(self, '_preserve_case_on_next_reset', False))
        case = dict(getattr(self, 'case', {})) if preserve_case else {}
        episode_id = getattr(self, 'episode_id', None) if preserve_case else None
        super().reset()
        self.case = case
        self.decisions = []
        self.history = []
        self.plan_state = None
        self.episode_id = episode_id
        self._pending_harness_event = None
        self._rsi_context_written = False
        self._pending_actor_text = None
        self._last_public_observation = None
        self._last_public_state = None
        self._last_issued_plan_index = None
        self._hierarchy_previous_images = None
        self._hierarchy_calls = 0
        self._hierarchy_cards = None
        self._execution_catalog = None
        self._bounded_execution = None
        self._reentry_controller = None
        self._native_atomic_controller = None
        if hasattr(self, '_atomic_actors'):
            self._atomic_actors.reset()
        if hasattr(self, '_native_fallback'):
            self._native_fallback.reset()
        self._active_actor_policy = self
        if getattr(self, 'execution_contract', 'legacy') == 'bounded_v2':
            from bounded_execution import BoundedExecution
            self._bounded_execution = BoundedExecution()
        self._preserve_case_on_next_reset = False

    def _reentry_actor_text(self, text, mode):
        controller = (self._native_atomic_controller if self.execution_contract == 'native_atomic_v1'
                      else self._reentry_controller)
        self._pending_actor_text = {'instruction': controller.instruction,
                                    'execution_prompt': text,
                                    'execution_mode': 'atomic' if mode == 'atomic' else 'whole_task',
                                    'actor_adapter': mode,
                                    'conditioned_instruction': text}

    def _reentry_log(self, event):
        record = {**event, 'episode_id': self.episode_id, 'task': self.case['task_name'],
                  'condition': self.condition, 'seed': self.case['seed'], 'split': self.split,
                  'model_version': self.model_version, 'instruction': self._reentry_controller.instruction,
                  'observation': {'summary': 'current public RGB and qpos'},
                  'terminal': False, 'evaluation_updates_blocked': True}
        if event['event'] == 'astra_reentry_decision':
            record.update(execution_feedback='unknown', source='rollout',
                          execution_mode='whole_task' if event['selected_skill_id'] == 'whole_task' else 'atomic')
            self.decisions.append(record)
        self._log(record)

    def _predict_reentry(self, state, images, instruction):
        if self._reentry_controller is None:
            from astra_reentry import ReentryController, load_atomic_cards
            cards = load_atomic_cards(self._atomic_catalog_path, self.case['task_name'])
            if not all(self._atomic_actors.supports(card) for card in cards):
                raise ValueError('技能库含训练 actor 或完成头不支持的绑定')
            self._reentry_controller = ReentryController(
                self._atomic_actors, self.planner, cards, instruction, self._reentry_max_steps,
                self._reentry_log, call_limit=self._reentry_call_limit, task_name=self.case['task_name'],
                native_fallback=getattr(self, '_native_fallback', None))
        elif self._reentry_controller.instruction != instruction:
            raise ValueError('回合内原任务指令不得改变')
        actions = self._reentry_controller.step(state, images, self._retrieved_memories(instruction))
        return unpack_robot_state(actions, 'joint', self.dim_info, source_type='obs')

    def _native_atomic_log(self, event):
        record = {**event, 'episode_id': self.episode_id, 'task': self.case['task_name'],
                  'condition': self.condition, 'seed': self.case['seed'], 'split': self.split,
                  'model_version': self.model_version, 'instruction': self._native_atomic_instruction,
                  'observation': {'summary': 'current public RGB and qpos'},
                  'planner_backend': 'native', 'terminal': False, 'evaluation_updates_blocked': True}
        if event['event'] == 'native_atomic_decision':
            record.update(execution_feedback='unknown', source='rollout', execution_mode='atomic')
            self.decisions.append(record)
        self._log(record)

    def _predict_native_atomic(self, state, images, instruction):
        if self._native_atomic_controller is None:
            from astra_reentry import load_atomic_cards
            from native_atomic import NativeAtomicController
            self._native_atomic_instruction = instruction
            cards = load_atomic_cards(self._atomic_catalog_path, self.case['task_name'])
            self._native_atomic_controller = NativeAtomicController(
                self._atomic_actors, self.planner, cards, instruction, self.case['task_name'],
                self.condition, self._reentry_max_steps, self._native_atomic_log,
                patch=self.harness_patch, runtime_editor=self.runtime_editor,
                stage_spec=(json.loads(Path(self.native_stage_spec_path).read_text()).get('native_stage_spec')
                            if self.native_stage_spec_path else None))
            self.plan_state = self._native_atomic_controller.plan_state
        elif instruction != self._native_atomic_controller.instruction:
            raise ValueError('回合内原任务指令不得改变')
        actions = self._native_atomic_controller.step(state, images, self._retrieved_memories(instruction))
        return unpack_robot_state(actions, 'joint', self.dim_info, source_type='obs')

    def prepare_case(self, case_meta=None):
        # Clear any previous episode before recording the official task/seed.
        self._preserve_case_on_next_reset = False
        self.reset()
        self.case = dict(case_meta or {})
        self.episode_id = (
            f"{self.variant}:{self.harness_variant}:{self.condition}:"
            f"{self.case['task_name']}:{self.case['seed']}"
        )
        self._preserve_case_on_next_reset = True
        # 保留回合级动作噪声种子；API 采样机制在 planner_output 单独记录。
        actor_seed = int(self.case['seed'])
        torch.manual_seed(actor_seed)
        torch.cuda.manual_seed_all(actor_seed)

    def _log(self, event):
        with (self.run_dir / 'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def _retrieved_memories(self, instruction):
        memory_result = self.memory.retrieve({'task': self.case['task_name'],
            'condition': self.condition, 'instruction': instruction}, limit=6, split='train')
        relevant = [episode for episode in memory_result['episodes']
                    if episode.get('task') == self.case['task_name']
                    and episode.get('execution_feedback') in {'success', 'failure'}]
        memories = []
        for outcome in ('failure', 'success'):
            episode = next((item for item in relevant if item['execution_feedback'] == outcome), None)
            if episode is None:
                continue
            sequence = []
            for item in episode.get('skill_history', []):
                label = item.get('execution_prompt') or item.get('selected_skill_id')
                if label and (not sequence or sequence[-1] != label):
                    sequence.append(label)
            memories.append({'text': 'Previously attempted sequence: ' + ' -> '.join(sequence[-6:])
                             + '. Whole-episode outcome only; individual steps are not labelled.',
                             'skill_id': episode.get('selected_skill_id'),
                             'outcome': episode.get('execution_feedback'),
                             'failure_reason': episode.get('failure_source'),
                             'metadata': {'condition': episode.get('condition'),
                                          'episode_id': episode.get('episode_id')}})
        memories.extend({'text': item['correction'], 'outcome': 'validated'}
                        for item in memory_result['validated_experiences'][:2])
        return memories

    def _predict_hierarchy(self, env_id, obs, instruction, images, state):
        from astra_hierarchy import HierarchyState, AstraPlannerError
        from harness_evolution import load_harness
        if self.plan_state is None:
            harness = load_harness(self.harness_rules_path, self.case['task_name'], self.condition)
            self.plan_state = HierarchyState(self.library, harness)
            self._hierarchy_cards = retrieve_skill_cards(instruction, self.case['task_name'],
                limit=self.hierarchy_retrieval_limit, library=self.library)
        if self._hierarchy_calls >= self.hierarchy_call_limit:
            raise AstraPlannerError('已达本回合 Astra 调用预算；不重试、不回退或增加动作预算')
        transition = None
        if self._last_public_observation is not None:
            transition = build_transition_evidence(before=self._last_public_observation, after=obs,
                before_qpos=self._last_public_state, after_qpos=state)
        before = self.plan_state.export()
        memories = self._retrieved_memories(instruction)
        context = {'mode': 'plan_and_monitor', 'task_name': self.case['task_name'],
                   'condition': self.condition, 'harness_rules': self.plan_state.harness,
                   'plan_state': before, 'previous_qpos': self._last_public_state,
                   'public_transition': transition, 'remaining_calls': self.hierarchy_call_limit - self._hierarchy_calls,
                   'action_chunk_steps': 30, 'environment_step_limit': self.hierarchy_max_steps,
                   'decision_index': self._hierarchy_calls, 'phase': self.experiment_phase,
                   'terminal_feedback_available': False}
        self._hierarchy_calls += 1
        output = self.planner.plan_and_monitor(images, self._hierarchy_previous_images, state,
                                              instruction, self._hierarchy_cards, memories, context)
        chosen, application = self.plan_state.apply(output)
        index = len(self.decisions)
        event = {'episode_id': self.episode_id, 'task': self.case['task_name'], 'condition': self.condition,
                 'seed': self.case['seed'], 'split': self.split, 'experiment_phase': self.experiment_phase,
                 'model_version': self.model_version, 'instruction': instruction,
                 'selected_skill_id': chosen['skill_id'], 'selected_base_skill_id': chosen['base_skill_id'],
                 'execution_prompt': chosen['execution_prompt'], 'plan_index': chosen['plan_index'],
                 'observation': {'summary': f'public RGB and qpos at decision {index}'},
                 'execution_feedback': 'unknown', 'terminal': False, 'step_index': index, 'source': 'rollout',
                 'weak_label': False, 'planner_backend': 'astra_hierarchy', 'actor_vlm': 'paligemma2',
                 'plan_before': before, 'plan_after': self.plan_state.export(), 'memory': memories,
                 'retrieved_cards': self._hierarchy_cards, 'planner_output': output,
                 'harness_application': application, 'harness_version': self.plan_state.harness['version'],
                 'status_is_prediction': True, 'evaluation_updates_blocked': True,
                 'call_count': self._hierarchy_calls, 'call_limit': self.hierarchy_call_limit,
                 'decision_time': time.time()}
        self.decisions.append(self.memory.append_event(event))
        self._log(event)
        self.history.append({'skill_id': chosen['skill_id'], 'execution_prompt': chosen['execution_prompt'],
                             'plan_index': chosen['plan_index'], 'outcome': 'unconfirmed'})
        self._last_public_observation = self._snapshot_public_observation(obs)
        self._last_public_state = state.copy()
        self._hierarchy_previous_images = {key: value.copy() for key, value in images.items()}
        conditioned = dict(obs, instruction=compose_instruction(instruction, chosen['execution_prompt']))
        self._pending_actor_text = {'instruction': instruction, 'execution_prompt': chosen['execution_prompt'],
                                    'selected_skill_id': chosen['skill_id'], 'plan_index': chosen['plan_index'],
                                    'conditioned_instruction': conditioned['instruction']}
        self._observations[env_id] = conditioned
        try:
            actions = super()._predict(env_id)
            self.plan_state.mark_issued()
            return actions
        finally:
            self._observations[env_id] = obs

    def _predict(self, env_id):
        if self.variant == 'baseline':
            return super()._predict(env_id)
        if not self.episode_id:
            raise RuntimeError('prepare_case 尚未收到有效的回合元数据')
        obs = self._observations[env_id]
        instruction = obs.get('instruction', obs.get('instructions'))
        if isinstance(instruction, list) and len(instruction) == 1:
            instruction = instruction[0]
        if not isinstance(instruction, str):
            raise ValueError('需要一条实际评测指令')
        state = np.asarray(pack_robot_state(obs, 'joint', self.dim_info), dtype=np.float32)
        images = {native: np.asarray(obs['vision'][standard]['color'])
                  for native, standard in self.cameras.items()}
        if getattr(self, 'execution_contract', 'legacy') == 'native_atomic_v1':
            return self._predict_native_atomic(state, images, instruction)
        if getattr(self, 'execution_contract', 'legacy') == 'reentry_atomic_v1':
            self._native_current_observation = (env_id, obs)
            return self._predict_reentry(state, images, instruction)
        if self.planner_backend == 'astra_hierarchy':
            return self._predict_hierarchy(env_id, obs, instruction, images, state)
        transition = None
        if self._last_public_observation is not None:
            transition = build_transition_evidence(
                before=self._last_public_observation,
                after=obs,
                before_qpos=self._last_public_state,
                after_qpos=state,
            )
        if self.plan_state is None:
            plan = build_execution_plan(instruction, self.case['task_name'], library=self.library)
            if getattr(self, '_bounded_execution', None) is not None:
                from skill_library import execution_skill_cards
                self._execution_catalog = execution_skill_cards(self.case['task_name'], instruction, self.library)
                plan = [card for card in self._execution_catalog if card['kind'] != 'whole_task']
            self.plan_state = PlanState(plan, patch=self.harness_patch)
            self._log({'event': 'plan_created', 'episode_id': self.episode_id,
                       'task': self.case['task_name'], 'condition': self.condition,
                       'split': self.split, 'seed': self.case['seed'],
                       'source': 'public_task_recipe', 'plan': plan,
                       'harness_patch': self.plan_state.patch,
                       'patch_applied': self.plan_state.patch['patch_id'] != 'no_change',
                       'completion_detection': 'not_verified'})
        before_plan = self.plan_state.export()
        if self.decisions and self._last_issued_plan_index is not None:
            # 下一次动作请求意味着上一段动作已交给环境尝试，不等于技能完成。
            if self._last_issued_plan_index == self.plan_state.current_index:
                # mark_executed consumes the evidence once; eligible() below
                # reads that state without consuming it a second time.
                self.plan_state.mark_executed(
                    self._last_issued_plan_index,
                    transition_evidence=transition,
                )
            else:
                self._log({
                    'event': 'harness_mark_skipped',
                    'episode_id': self.episode_id,
                    'last_issued_plan_index': self._last_issued_plan_index,
                    'current_plan_index': self.plan_state.current_index,
                    'reason': 'index_changed_before_mark',
                    'transition_evidence': transition,
                })
        # Close the previous editor transition with the next genuine public
        # observation before making a new edit.  At episode end, the missing
        # post-observation is recorded explicitly in on_trial_end().
        self._log_harness_post_observation(obs)
        if self.runtime_editor is not None:
            context = safe_context(self.case['task_name'], self.condition,
                                   self.plan_state.export(), transition)
            edit = self.runtime_editor.select(context)
            self.plan_state.apply_harness_patch(edit['patch'])
            self._log({'event': 'runtime_edit', 'episode_id': self.episode_id,
                       'decision_index': len(self.decisions), 'context': context,
                       'editor_version': self.editor_version, **edit,
                       'evaluation_updates_blocked': True})
        candidates = self.plan_state.eligible()
        if not candidates:
            raise RuntimeError('harness editor left no eligible skill candidates')
        after_plan = self.plan_state.export()
        patch_id = self._log_harness_transition(obs, before_plan, after_plan, transition)
        memories = self._retrieved_memories(instruction)
        plan_context = {'ordered_plan': [step['execution_prompt'] for step in self.plan_state.plan],
                        'eligible_plan_indices': [step['plan_index'] for step in candidates],
                        'progress_is_prediction': True}
        controller = getattr(self, '_bounded_execution', None)
        whole_task_card = None
        if controller is not None:
            whole_task_card = self._execution_catalog[-1]
            plan_context.update(mode='guided_execution', task_name=self.case['task_name'],
                                execution_catalog=self._execution_catalog,
                                execution_control=controller.export(), public_transition=transition,
                                previous_qpos=self._last_public_state,
                                terminal_feedback_available=False)
            if self.planner_backend == 'astra_guided':
                candidates = candidates + [whole_task_card]
        if controller is not None and controller.whole_task:
            selected = {'skill_id': whole_task_card['skill_id'],
                        'selection_distribution': 'bounded_execution',
                        'reason': controller.reason}
        else:
            planning_images = images
            if self.planner_backend == 'astra_guided':
                planning_images = {'current_' + key: value for key, value in images.items()}
                planning_images.update({'previous_' + key: value for key, value in
                                        (getattr(self, '_hierarchy_previous_images', None) or {}).items()})
            selected = self.planner.rank(planning_images, state, instruction, candidates, memories,
                                         history=[plan_context] + self.history[-4:])
        if hasattr(selected, '__dict__'):
            selected = vars(selected)
        if self.planner_backend == 'native' and not (controller and controller.whole_task):
            selected = dict(selected, selection_distribution='greedy', selection_logprob=None)
        chosen = next((s for s in candidates if s['skill_id'] == selected['skill_id']), None)
        if controller is not None:
            controller.observe(selected)
            if selected['skill_id'] == whole_task_card['skill_id']:
                controller.delegate('planner_requested_whole_task')
            if controller.whole_task:
                chosen = whole_task_card
        if chosen is None:
            raise ValueError('System2 返回了候选集外技能')
        execution_mode = 'whole_task' if controller and controller.whole_task else 'skill'
        if execution_mode == 'skill':
            self.plan_state.choose(chosen['plan_index'])
        idx = len(self.decisions)
        observation = {'summary': f'public RGB and qpos at decision {idx}'}
        feature_path = None
        rsi_context_path = None
        if self.planner_backend == 'native' and execution_mode == 'skill':
            rsi_context_path = self._write_rsi_context(
                candidates=candidates,
                features=self.planner.last_candidate_features,
                patch_id=patch_id,
            )
        if self.split == 'train':
            path = self.run_dir / 'observations' / f"{self.case['seed']}_{idx:04d}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, state=state, **images)
            observation = {'ref': str(path)}
            feature_path = path.with_suffix('.pt')
            torch.save(self.planner.last_candidate_features, feature_path)
        event = {'episode_id': self.episode_id, 'task': self.case['task_name'],
                 'condition': self.condition, 'seed': self.case['seed'], 'split': self.split,
                 'model_version': self.model_version, 'instruction': instruction,
                 'planner_backend': self.planner_backend,
                 'actor_vlm': 'paligemma2',
                 'execution_mode': execution_mode,
                 'execution_control': controller.export() if controller else None,
                 'selected_skill_id': chosen['skill_id'], 'observation': observation,
                 'requested_skill_id': selected['skill_id'],
                 'selected_base_skill_id': chosen.get('base_skill_id', chosen['skill_id']),
                 'execution_prompt': chosen['execution_prompt'],
                 'plan_index': chosen['plan_index'], 'plan_before': before_plan,
                 'plan_after': self.plan_state.export(),
                 'execution_feedback': 'unknown', 'terminal': False, 'step_index': idx,
                 'source': 'rollout', 'history': list(self.history[-8:]),
                 'candidates': candidates, 'memory': memories, 'weak_label': False,
                 'harness_variant': self.harness_variant,
                 'editor_version': self.editor_version,
                 'harness_patch': self.plan_state.patch,
                 'patch_applied': self.plan_state.patch['patch_id'] != 'no_change',
                 'patch_id': patch_id,
                 'plan_after_patch': after_plan,
                 'selection_logprob': selected.get('selection_logprob'),
                 'selection_distribution': selected.get('selection_distribution', 'greedy'),
                 'decision_time': time.time(), 'planner_output': selected}
        if feature_path is not None:
            event['features'] = str(feature_path)
            event['feature_source'] = 'same_frozen_vlm_forward_used_for_decision'
        if rsi_context_path is not None:
            event['rsi_context'] = rsi_context_path
            event['rsi_context_source'] = 'first_decision_frozen_vlm'
        self.decisions.append(self.memory.append_event(event))
        self._log(event)
        self.history.append({'skill_id': chosen['skill_id'],
                             'execution_prompt': chosen['execution_prompt'],
                             'plan_index': chosen['plan_index'], 'outcome': 'unconfirmed'})
        self._last_issued_plan_index = chosen['plan_index'] if execution_mode == 'skill' else None
        self._last_public_observation = self._snapshot_public_observation(obs)
        self._last_public_state = state
        if self.planner_backend == 'astra_guided':
            self._hierarchy_previous_images = {key: value.copy() for key, value in images.items()}
        # done 只作为模型预测记录；任务终止由官方评测决定。
        conditioned = dict(obs)
        conditioned['instruction'] = (instruction if execution_mode == 'whole_task' else
                                      compose_instruction(instruction, chosen['execution_prompt']))
        self._pending_actor_text = {'instruction': instruction,
                                    'execution_mode': execution_mode,
                                    'execution_prompt': chosen['execution_prompt'],
                                    'selected_skill_id': chosen['skill_id'],
                                    'plan_index': chosen['plan_index'],
                                    'conditioned_instruction': conditioned['instruction']}
        self._observations[env_id] = conditioned
        try:
            actions = super()._predict(env_id)
            if controller is not None:
                controller.note_issued(chosen['skill_id'])
            return actions
        finally:
            self._observations[env_id] = obs

    def on_trial_end(self, result=None):
        self._preserve_case_on_next_reset = False
        result = dict(result or {})
        if (result.get('attempt_status') == 'aborted' or result.get('included_in_sr') is False
                or result.get('error_category') in {'api_error', 'infrastructure_error'}):
            self._log({'event': 'episode_aborted', 'episode_id': self.episode_id,
                       'attempt_status': 'aborted', 'task_success': None, 'included_in_sr': False,
                       'execution_feedback': 'aborted', 'error_category': result.get('error_category'),
                       'terminal_task_outcome_observed': False})
            self._pending_harness_event = None
            return
        if self._pending_harness_event is not None:
            self._log({
                'event': 'harness_post_observation_missing',
                'episode_id': self.episode_id,
                'patch_id': self._pending_harness_event['patch_id'],
                'decision_index': self._pending_harness_event['decision_index'],
                'post_observation': None,
                'post_observation_status': 'missing_official_terminal_callback',
            })
            self._pending_harness_event = None
        if not self.decisions:
            return
        last = self.decisions[-1]
        terminal = {key: last[key] for key in ('episode_id', 'task', 'condition', 'seed',
            'split', 'model_version', 'instruction', 'selected_skill_id', 'observation')}
        terminal.update(record_type='episode_outcome', terminal=True,
                        execution_feedback='success' if result.get('success') else 'failure',
                        failure_source='unknown', step_index=len(self.decisions), source='rollout',
                        plan_state=(self._reentry_controller.export() if
                                    getattr(self, '_reentry_controller', None) is not None else
                                    self.plan_state.export()), label_scope='episode',
                        harness_variant=self.harness_variant,
                        editor_version=self.editor_version)
        self.memory.append_event(terminal)
        self._log(terminal)

