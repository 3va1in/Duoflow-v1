import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from astra_planner import AstraAPIError
from astra_reentry import ReentryController


class Actors:
    threshold = 0.9

    def __init__(self, probabilities=()):
        self.probabilities = iter(probabilities)
        self.whole_calls, self.atomic_calls = [], []
        self.condition_to_index = {'pick_and_lift|Pick up the QR-code sign with the left arm.': 0}

    def whole(self, state, images, instruction):
        self.whole_calls.append(instruction)
        return np.zeros((50, 14), dtype=np.float32)

    def predict(self, state, images, skill_text, skill_id, reset=False):
        self.atomic_calls.append((skill_id, reset))
        return {'actions': np.zeros((50, 14), dtype=np.float32),
                'completion_prob': next(self.probabilities, 0.01)}


class Planner:
    def __init__(self, choices):
        self.choices, self.calls = iter(choices), []

    def rank(self, images, state, instruction, candidates, memories, history):
        self.calls.append((images, candidates, history))
        choice = next(self.choices)
        if isinstance(choice, Exception):
            raise choice
        return {'skill_id': choice, 'progress_status': 'unknown',
                'supporting_evidence': [], 'reason': 'test'}


def controller(choices, budget=250, probabilities=()):
    actor, planner, events = Actors(probabilities), Planner(choices), []
    cards = [{'skill_id': 'pick_left', 'atomic_skill_id': 'pick_and_lift', 'task': 'rotate_qrcode',
              'execution_prompt': 'Pick up the QR-code sign with the left arm.',
              'max_policy_steps': budget}]
    value = ReentryController(actor, planner, cards, 'Rotate the QR code.', 400, events.append)
    return value, actor, planner, events


def step(value):
    return value.step(np.zeros(14, dtype=np.float32),
                      {'cam_high': np.zeros((2, 2, 3), dtype=np.uint8)}, [])


class ReentryTests(unittest.TestCase):
    def test_shared_backbone_restores_original_actor_and_resets_cache(self):
        source = Path(__file__).with_name('reentry_actors.py')
        definition = next(node for node in ast.parse(source.read_text()).body
                          if isinstance(node, ast.ClassDef))
        namespace = {'np': np, 'torch': SimpleNamespace(no_grad=nullcontext,
                     inference_mode=lambda: lambda function: function)}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
        actors = object.__new__(namespace['SharedAtomicActors'])
        values, resets = {'weight': 1}, []
        parameter = SimpleNamespace(copy_=lambda value: values.update(weight=value))
        def predict(state, images, instruction, reset):
            resets.append(reset)
            return np.full((50, 14), values['weight'], dtype=np.float32)
        actors.__dict__.update(parameters={'lora': parameter},
            states={'whole': {'lora': 1}, 'atomic': {'lora': 2}},
            mode='whole', last_input=None, engine=SimpleNamespace(predict=predict),
            audit_text=lambda *args: None)
        args = (np.zeros(14), {})
        self.assertEqual(actors._actions('whole', *args, 'original')[0, 0], 1)
        self.assertEqual(actors._actions('atomic', *args, 'skill')[0, 0], 2)
        self.assertEqual(actors._actions('whole', *args, 'original')[0, 0], 1)
        actors._actions('whole', *args, 'original')
        self.assertEqual(resets, [True, True, True, False])

    def test_whole_task_returns_to_astra_after_90_actions(self):
        value, actor, planner, events = controller(['whole_task', 'pick_left'])
        for _ in range(3):
            self.assertEqual(len(step(value)), 30)
        self.assertEqual(value.total_steps, 90)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(step(value)), 10)
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(value.active, 'pick_left')
        self.assertIn('previous_decision_cam_high', planner.calls[-1][0])
        self.assertEqual(len(actor.whole_calls), 3)

    def test_skill_review_keeps_cumulative_budget(self):
        value, actor, planner, events = controller(['pick_left', 'pick_left', 'whole_task'], budget=65)
        for _ in range(6):
            step(value)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(step(value)), 5)
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(value.used_steps['pick_left'], 65)
        step(value)
        self.assertEqual(value.active, 'whole_task')
        self.assertEqual(value.used_steps['pick_left'], 65)
        self.assertEqual([r for r in events if r['event'] == 'atomic_skill_return'][-1]['status'], 'timeout')
        self.assertNotIn('pick_left', [c['skill_id'] for c in planner.calls[-1][1]])
        self.assertTrue(actor.atomic_calls[0][1])
        self.assertFalse(actor.atomic_calls[-1][1])

    def test_api_error_does_not_invoke_actor_or_fallback(self):
        value, actor, planner, events = controller([AstraAPIError('network_error')])
        with self.assertRaises(AstraAPIError):
            step(value)
        self.assertEqual(actor.whole_calls, [])
        self.assertEqual(actor.atomic_calls, [])
        self.assertEqual(value.total_steps, 0)

    def test_completion_prediction_returns_planning_without_terminal(self):
        value, actor, planner, events = controller(['pick_left', 'whole_task'], probabilities=[0.95, 0.96])
        self.assertEqual(len(step(value)), 1)
        self.assertEqual(len(step(value)), 30)
        returned = [r for r in events if r['event'] == 'atomic_skill_return']
        self.assertEqual(returned[0]['status'], 'predicted_complete')
        self.assertEqual(value.total_steps, 31)
        self.assertFalse(value.export()['verified_complete'])
        self.assertEqual(value.active, 'whole_task')

    def test_confirmation_hold_does_not_overrun_observation_window(self):
        value, actor, planner, events = controller(['pick_left', 'pick_left'],
                                                 probabilities=[0.95, 0.1] + [0.1] * 10)
        for _ in range(7):
            step(value)
        self.assertEqual(value.total_steps, 60)
        self.assertEqual(value.executors['pick_left'].steps, 60)
        step(value)
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(value.executors['pick_left'].steps, 70)


if __name__ == '__main__':
    unittest.main()
