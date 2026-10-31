import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from execution_outcomes import AstraAPIError, AstraPlannerError, classify_execution_error
from outcome_summary import summarize_outcomes
from paired_scenes import make_paired_evaluator


class WsError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


WsError.__module__ = 'client_server.ws.protocol.exceptions'


class OutcomeTests(unittest.TestCase):
    def evaluate(self, root, error=None):
        config = {'task': 'place_object_basket', 'condition': 'demo_randomized',
                  'seed': 201, 'scene_manifest_path': str(root / 'scenes.json')}
        (root / 'scenes.json').write_text(json.dumps({**config, 'status': 'complete_before_policy',
            'scenes': [{'scene_index': 0, 'seed': 20200001, 'instruction': 'put object',
                        'public_initial_fingerprint': {'public': 1}}]}))
        env = SimpleNamespace(eval_video_path=None, render_freq=0, eval_success=False,
                              take_action_cnt=0, closed=0)
        env.setup_demo = lambda **kwargs: None
        env.set_instruction = lambda **kwargs: None
        env.get_obs = lambda: {'public': 1}
        def close(**kwargs):
            env.closed += 1
        def action(*args, **kwargs):
            env.take_action_cnt += 1
        env.close_env, env.take_action = close, action
        events, notifications = [], []
        official = SimpleNamespace(
            robotwin_obs_to_xpolicylab=lambda obs, **kwargs: obs,
            safe_close_env=lambda env: env.close_env(),
            prepare_policy_case=lambda *args: None, reset_policy=lambda *args: None,
            is_episode_end=lambda env: env.take_action_cnt >= 2,
            normalize_action_chunk=lambda raw: raw,
            xpolicylab_action_to_robotwin=lambda action, **kwargs: (action, 'joint'),
            notify_trial_end=lambda client, task, seed, success: notifications.append(success))
        def call(*, func_name, **kwargs):
            if func_name == 'get_action':
                if error is not None:
                    raise error
                return [[0]]
        client = SimpleNamespace(call=call)
        evaluator = make_paired_evaluator(official, config,
            lambda event, **row: events.append({'event': event, **row}), lambda *args: None)
        with patch('paired_scenes.public_fingerprint', side_effect=lambda obs: obs):
            if error is not None:
                with self.assertRaises(type(error)):
                    evaluator(config['task'], env, {'clear_cache_freq': 5}, client, 20200001,
                              usr_args={'action_type': 'joint'}, test_num=1)
            else:
                evaluator(config['task'], env, {'clear_cache_freq': 5}, client, 20200001,
                          usr_args={'action_type': 'joint'}, test_num=1)
        return summarize_outcomes(events), events, notifications, env

    def test_api_abort_does_not_count_failure_or_notify_policy(self):
        wrapped = WsError('call_failed', str(AstraAPIError('network_error', request_id='request-1')))
        with tempfile.TemporaryDirectory() as folder:
            summary, events, notifications, env = self.evaluate(Path(folder), wrapped)
        self.assertEqual(summary['sr_denominator'], 0)
        self.assertEqual(summary['api_error_count'], 1)
        self.assertEqual(summary['normal_failures'], 0)
        self.assertIsNone(summary['outcomes'][0]['task_success'])
        self.assertFalse(summary['outcomes'][0]['included_in_sr'])
        self.assertEqual(notifications, [])
        self.assertEqual(env.test_num, 0)
        self.assertFalse(any(row['event'] == 'policy_error' for row in events))

    def test_normal_action_budget_failure_counts_in_sr(self):
        with tempfile.TemporaryDirectory() as folder:
            summary, events, notifications, env = self.evaluate(Path(folder))
        self.assertEqual(summary['sr_denominator'], 1)
        self.assertEqual(summary['normal_failures'], 1)
        self.assertEqual(summary['sr'], 0)
        self.assertEqual(summary['aborted_episodes'], 0)
        self.assertEqual(notifications, [False])
        self.assertEqual(env.test_num, 1)

    def test_malformed_output_stays_policy_error(self):
        wrapped = WsError('call_failed', 'API 技能选择不是可解析的 JSON')
        with tempfile.TemporaryDirectory() as folder:
            summary, events, notifications, env = self.evaluate(Path(folder), wrapped)
        self.assertEqual(summary['policy_error_count'], 1)
        self.assertEqual(summary['infrastructure_error_count'], 0)
        self.assertEqual(notifications, [])

    def test_rpc_timeout_is_not_a_program_timeout(self):
        self.assertEqual(classify_execution_error(WsError('timeout', 'timeout waiting'),
            phase='policy_rpc')['error_category'], 'infrastructure_error')
        self.assertEqual(classify_execution_error(TimeoutError('calculation exceeded'),
            phase='action_execution')['error_category'], 'policy_error')

    def test_bridge_known_upstream_failure_and_invalid_request_are_distinct(self):
        from astra_planner import AstraPlannerBackend
        upstream_codes = ('network_error', 'upstream_result_unknown',
                          'upstream_call_budget_exhausted', 'existing_sharing_session_required')
        for code, exception in [(code, AstraAPIError) for code in upstream_codes] + [('invalid_request', AstraPlannerError)]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'responses').mkdir()
                (root / 'responses/request-1.json').write_text(json.dumps(
                    {'ok': False, 'error': {'code': code}}))
                planner = AstraPlannerBackend(transport='siwc_files', request_root=root)
                with patch('astra_planner.uuid.uuid4', return_value=SimpleNamespace(hex='request-1')):
                    with self.assertRaises(exception) as caught:
                        planner._file_response({}, None)
                self.assertEqual(isinstance(caught.exception, AstraAPIError), code in upstream_codes)
                self.assertEqual(len(list((root / 'requests').glob('*.json'))), 1)


    def test_bridge_http504_requires_explicit_upstream_receipt(self):
        import copy
        from astra_planner import AstraPlannerBackend
        response = {'ok': False, 'error': {'code': 'api_error', 'status': 504},
                    'received_delta_count': 0, 'stream_text': '',
                    'transport': {'stage': 'upstream', 'upstream_attempts': 1, 'automatic_retry': False}}
        cases = [('504', response, True)]
        for name, path, value in [('auth', ('error', 'status'), 401),
                                  ('quota', ('error', 'status'), 429),
                                  ('local', ('transport', 'stage'), 'local'),
                                  ('unknown', ('error', 'code'), 'invalid_request'),
                                  ('missing_transport', ('transport',), {}),
                                  ('partial_output', ('received_delta_count',), 1)]:
            changed = copy.deepcopy(response)
            target = changed if len(path) == 1 else changed[path[0]]
            target[path[-1]] = value
            cases.append((name, changed, False))
        for name, value, is_api in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'responses').mkdir()
                (root / 'responses/request-504.json').write_text(json.dumps(value))
                planner = AstraPlannerBackend(transport='siwc_files', request_root=root)
                with patch('astra_planner.uuid.uuid4', return_value=SimpleNamespace(hex='request-504')):
                    with self.assertRaises(AstraPlannerError) as caught:
                        planner._file_response({}, None)
                self.assertEqual(isinstance(caught.exception, AstraAPIError), is_api)
                if is_api:
                    self.assertEqual(caught.exception.payload,
                                     {'code': 'transport_timeout', 'request_id': 'request-504', 'http_status': 504})
                self.assertEqual(len(list((root / 'requests').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
