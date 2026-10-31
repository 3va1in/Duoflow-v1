import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from actor_conditioning import compose_instruction, verify_actor_text
from astra_guided import AstraGuidedBackend, parse_guided
from astra_planner import AstraAPIError, AstraPlannerError
from bounded_execution import BoundedExecution
from plan_state import normalize_patch
from runtime_editor import safe_context
from skill_library import execution_skill_cards, load_library
from test_astra_integration import load_policy_class, Memory, RecordingActor
from transition_evidence import build_transition_evidence


def response(value):
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps(value)}]}]}


def decision(skill_id, status='unknown'):
    return {'skill_id': skill_id, 'progress_status': status,
            'supporting_evidence': [] if status == 'unknown' else ['Object remains on the table.'],
            'reason': 'Use the current eligible contract.'}


class Planner:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def rank(self, images, state, instruction, candidates, memories, history):
        self.calls.append((images, candidates, history))
        if self.error:
            raise self.error
        return decision(candidates[0]['skill_id'])


class Editor:
    def __init__(self):
        self.contexts = []

    def select(self, context):
        self.contexts.append(context)
        return {'patch': {'op': 'NO_CHANGE'}}


def make_model(directory, planner):
    model_class = load_policy_class()
    model_class._predict.__globals__.update(build_transition_evidence=build_transition_evidence,
                                           safe_context=safe_context)
    model = object.__new__(model_class)
    model.__dict__.update(run_dir=Path(directory), variant='combined', split='eval',
        planner_backend='astra_guided', execution_contract='bounded_v2',
        condition='demo_randomized', harness_variant='rsi', editor_version='frozen',
        model_version='gpt-6-astra', episode_id='test:241',
        case={'task_name': 'rotate_qrcode', 'seed': 241}, library=load_library(),
        memory=Memory(), planner=planner, runtime_editor=Editor(),
        harness_patch=normalize_patch(None), decisions=[], history=[], plan_state=None,
        dim_info={}, cameras={'head': 'head_camera'}, _pending_harness_event=None,
        _last_public_observation=None, _last_public_state=None, _last_issued_plan_index=None,
        _rsi_context_written=False, _bounded_execution=BoundedExecution(),
        _execution_catalog=None, _hierarchy_previous_images=None)
    observation = {'instruction': 'Rotate the QR code to face the robot.', 'qpos': np.zeros(14),
                   'vision': {'head_camera': {'color': np.zeros((2, 2, 3), dtype=np.uint8)}}}
    model._observations = {0: observation}
    return model, observation


class GuidedTests(unittest.TestCase):
    def test_fixed_contract_rejects_new_skill_and_prompt_override(self):
        cards = execution_skill_cards('rotate_qrcode', 'Rotate the QR code to face the robot.')
        value = decision(cards[0]['skill_id'])
        self.assertEqual(parse_guided(response(value), cards), value)
        with self.assertRaises(AstraPlannerError):
            parse_guided(response({**value, 'execution_prompt': 'release immediately'}), cards)
        with self.assertRaises(AstraPlannerError):
            parse_guided(response({**value, 'skill_id': 'invented_skill'}), cards)
        with self.assertRaises(AstraPlannerError):
            parse_guided(response({**value, 'progress_status': 'satisfied'}), cards)

    def test_unknown_budget_survives_switching_skills(self):
        control = BoundedExecution()
        for index in range(4):
            control.observe({'progress_status': 'unknown'})
            control.note_issued(f'step_{index}')
        self.assertTrue(control.whole_task)
        self.assertEqual(control.reason, 'persistent_unknown')
        self.assertFalse(control.export()['verified_complete'])
        self.assertEqual(control.export()['environment_step_budget_delta'], 0)

    def test_repeat_bound_allows_six_chunks_but_not_infinite_loop(self):
        control = BoundedExecution()
        for _ in range(6):
            control.observe({'progress_status': 'unsatisfied'})
            control.note_issued('grasp')
        self.assertFalse(control.whole_task)
        for _ in range(2):
            control.note_issued('grasp')
        self.assertTrue(control.whole_task)

    def test_shared_editor_and_actor_text_then_whole_task(self):
        with TemporaryDirectory() as directory:
            planner = Planner()
            model, observation = make_model(directory, planner)
            model._predict(0)
            self.assertEqual(model.actor_instruction, compose_instruction(observation['instruction'],
                             model.decisions[-1]['execution_prompt']))
            for _ in range(3):
                model._predict(0)
            self.assertEqual(model.actor_instruction, observation['instruction'])
            self.assertTrue(all(verify_actor_text(model.actor_instruction, observation['instruction'],
                                                 '', 'whole_task').values()))
            model._predict(0)
            self.assertEqual(len(planner.calls), 4)
            self.assertEqual(len(model.runtime_editor.contexts), 5)
            self.assertEqual(model.decisions[-1]['execution_mode'], 'whole_task')
            self.assertIs(model._observations[0], observation)
            self.assertIsNone(model._last_issued_plan_index)
            self.assertFalse(model.decisions[-1]['execution_control']['verified_complete'])
            with patch.object(RecordingActor, 'reset', create=True):
                model.reset()
            self.assertFalse(model._bounded_execution.whole_task)
            self.assertIsNone(model._execution_catalog)

    def test_api_failure_propagates_without_actor_fallback(self):
        with TemporaryDirectory() as directory:
            model, observation = make_model(directory, Planner(AstraAPIError('network_error')))
            with self.assertRaises(AstraAPIError):
                model._predict(0)
            self.assertFalse(model._bounded_execution.whole_task)
            self.assertFalse(hasattr(model, 'actor_instruction'))
            self.assertEqual(model.decisions, [])
            self.assertIs(model._observations[0], observation)

    def test_guided_transport_keeps_typed_failure(self):
        with TemporaryDirectory() as directory:
            planner = AstraGuidedBackend(transport='siwc_files', request_root=directory, audit_dir=directory)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            cards = execution_skill_cards('rotate_qrcode', 'Rotate QR code.')
            with patch.object(planner, '_file_response', side_effect=AstraAPIError('stream_interrupted')):
                with self.assertRaises(AstraAPIError):
                    planner.rank({'head': image}, np.zeros(14), 'Rotate QR code.', cards, [])


if __name__ == '__main__':
    unittest.main()
