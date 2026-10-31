from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from astra_planner import AstraAPIError
from astra_reentry import ReentryController
from native_fallback import NativeFallback, missing_execution_contracts
from test_astra_reentry import Actors, Planner, step


class Native:
    def __init__(self):
        self.calls, self.suspends = 0, 0

    def step(self, state, images, instruction):
        self.calls += 1
        return np.ones((30, 14), dtype=np.float32)

    def suspend(self, state, images):
        self.suspends += 1


def controller(choices=(), task='beat_block_hammer', probabilities=(), budget=200):
    actor, planner, native, events = Actors(probabilities), Planner(choices), Native(), []
    cards = [{'skill_id': 'pick', 'atomic_skill_id': 'pick_and_lift', 'task': task,
              'execution_prompt': 'Pick up the QR-code sign with the left arm.',
              'max_policy_steps': budget}]
    value = ReentryController(actor, planner, cards, 'Do the task.', 400, events.append,
                              native_fallback=native)
    return value, actor, planner, native, events


class NativeFallbackTests(unittest.TestCase):
    def test_missing_chain_uses_native_before_any_atomic_or_api_call(self):
        value, actor, planner, native, events = controller(task='handover_block')
        for _ in range(4):
            self.assertTrue((step(value) == 1).all())
        self.assertEqual(native.calls, 4)
        self.assertEqual((planner.calls, actor.atomic_calls, actor.whole_calls), ([], [], []))
        self.assertEqual(value.total_steps, 120)
        self.assertTrue(events[0]['before_first_action'])
        self.assertFalse(events[-1]['returns_to_astra'])

    def test_full_native_window_returns_to_astra_and_keeps_budget(self):
        value, actor, planner, native, events = controller(['whole_task', 'pick'])
        for _ in range(3):
            step(value)
        self.assertEqual(value.total_steps, 90)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(step(value)), 10)
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(native.suspends, 1)
        self.assertEqual(native.calls, 3)
        self.assertEqual(actor.whole_calls, [])

    def test_early_completion_and_timeout_use_native_without_api_retry(self):
        for probabilities, budget in (([.99, .99], 200), ([], 10)):
            value, actor, planner, native, events = controller(['pick'], probabilities=probabilities,
                                                               budget=budget)
            step(value)
            self.assertEqual(len(step(value)), 30)
            self.assertEqual(native.calls, 1)
            self.assertEqual(len(planner.calls), 1)
            self.assertIn('pick', value.blocked_skills)

    def test_api_failure_never_calls_native(self):
        value, actor, planner, native, events = controller([AstraAPIError('network_error')])
        with self.assertRaises(AstraAPIError):
            step(value)
        self.assertEqual(native.calls, 0)
        self.assertEqual(value.total_steps, 0)

    def test_both_object_arm_placement_contracts_are_required(self):
        cards = [{'atomic_skill_id': 'place_and_release_block',
                  'bindings': {'arm': arm, 'object': obj}}
                 for arm in ('left', 'right') for obj in ('the red block', 'the green block')]
        self.assertEqual(missing_execution_contracts(cards, 'stack_blocks_two'), [])
        self.assertEqual(len(missing_execution_contracts(cards[:-1], 'stack_blocks_two')), 1)

    def test_native_suspension_consumes_only_its_last_action_once(self):
        marked, closed = [], []
        model = SimpleNamespace(_last_public_observation=None, _last_public_state=None,
            _last_issued_plan_index=2,
            plan_state=SimpleNamespace(current_index=2,
                mark_executed=lambda *args, **kwargs: marked.append((args, kwargs))),
            _log_harness_post_observation=lambda obs: closed.append(obs))
        owner = SimpleNamespace(_native_current_observation=(0, {'qpos': np.zeros(14)}))
        fallback = NativeFallback(owner, None, None, 'checkpoint')
        fallback.model, fallback.active = model, True
        fallback.suspend(np.zeros(14), {})
        fallback.suspend(np.zeros(14), {})
        self.assertEqual(len(marked), 1)
        self.assertEqual(len(closed), 1)
        self.assertIsNone(model._last_issued_plan_index)
        self.assertIsNone(model._last_public_observation)

    def test_native_restores_original_weights_before_selector_and_resets_on_reentry(self):
        order, resets = [], []
        owner = SimpleNamespace(_atomic_actors=SimpleNamespace(
            _activate=lambda mode: order.append(('activate', mode))),
            _native_current_observation=(0, {'instruction': 'original'}), dim_info={})
        def predict(_):
            order.append(('native_selector_and_actor', 'whole'))
            resets.append(model._reset_pending)
            model._reset_pending = False
            return [np.zeros(14) for _ in range(30)]
        model = SimpleNamespace(_reset_pending=False, _predict=predict)
        fallback = NativeFallback(owner, None, None, 'checkpoint')
        fallback.model = model
        dependency = SimpleNamespace(pack_robot_state=lambda obs, *args: obs['state'])
        with patch.dict(sys.modules, {'XPolicyLab.utils.process_data': dependency}):
            fallback.step(np.zeros(14), {}, 'original')
            fallback.step(np.zeros(14), {}, 'original')
            fallback.active = False
            fallback.step(np.zeros(14), {}, 'original')
        self.assertEqual(resets, [True, False, True])
        self.assertEqual(order, [('activate', 'whole'), ('native_selector_and_actor', 'whole')] * 3)
        self.assertIs(owner._active_actor_policy, owner)


if __name__ == '__main__':
    unittest.main()
