import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from real_robot.cameras import CAMERAS
from real_robot.remote import RemotePolicy, decode_images, encode_images, receive, ssh_command, transmit
from real_robot.runner import Journal, run_trial
from real_robot.test_airbot import Cameras, Client, SDK, configuration
from real_robot.airbot import DualAirbot

ROOT = Path(__file__).resolve().parents[1]


def fake_worker(root):
    from real_robot.policy_worker import serve

    class Model:
        def __init__(self, config, route, skill_id, run_dir, repo, journal):
            self.instruction, self.calls = config['instruction'], 0
            print('模型加载日志不应进入协议')

        def predict(self, state, images):
            if self.instruction == 'slow':
                time.sleep(5)
            if self.instruction == 'fail':
                raise RuntimeError('测试模型失败')
            if self.instruction == 'disconnect':
                os._exit(7)
            self.calls += 1
            return np.tile(state, (2, 1)), f'call_{self.calls}'

    wire = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    code = serve(sys.stdin.buffer, wire, Path(root), Model)
    wire.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = configuration()
        self.config['inference_timeout_s'] = 2
        self.journal = Journal(self.root / 'local')
        self.command = [sys.executable, '-u', '-c',
                        'from real_robot.test_remote import fake_worker; '
                        f'fake_worker({str(self.root)!r})']

    def policy(self):
        policy = RemotePolicy(self.command, self.config, 'native', None, self.journal, startup_timeout=5)
        self.addCleanup(policy.close)
        return policy

    def test_lossless_three_camera_roundtrip_and_persistent_policy(self):
        images = {name: np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)
                  for i, name in enumerate(CAMERAS)}
        for name, decoded in decode_images(encode_images(images)).items():
            np.testing.assert_array_equal(decoded, images[name])
        policy = self.policy()
        for call in (1, 2):
            state = np.arange(14, dtype=np.float32) * 0.01 * call
            actions, status = policy.predict(state, images)
            np.testing.assert_array_equal(actions, np.tile(state, (2, 1)))
            self.assertEqual(status, f'call_{call}')
        policy.close()
        self.assertEqual(policy.process.returncode, 0)
        worker = next((self.root / 'runs/remote_policy').iterdir())
        result = json.loads((worker / 'RESULT.json').read_text())
        self.assertEqual(result['predictions'], 2)
        self.assertEqual(result['status'], 'closed_by_client')
        for name in CAMERAS:
            from PIL import Image
            with Image.open(worker / 'observation_0001' / f'{name}.png') as image:
                np.testing.assert_array_equal(np.array(image), images[name])

    def test_worker_model_error_is_returned_without_retry(self):
        self.config['instruction'] = 'fail'
        policy = self.policy()
        with self.assertRaisesRegex(RuntimeError, '测试模型失败'):
            policy.predict(np.zeros(14), Cameras().read()[0])
        policy.close()
        worker = next((self.root / 'runs/remote_policy').iterdir())
        self.assertEqual(len(list(worker.glob('observation_*'))), 1)

    def test_remote_predictions_feed_existing_local_execution_loop(self):
        policy = self.policy()
        Client.instances = []
        Client.deny_right = Client.fail_right = Client.stale = False
        robot = DualAirbot(self.config['robot'], self.journal, sdk=SDK, arm_factory=Client)
        result = run_trial(self.config, 'execute', robot, Cameras(), policy, self.journal)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['submitted_action_pairs'], 3)
        self.assertEqual(policy.sequence, 2)
        self.assertEqual(policy.process.returncode, 0)
        for client in Client.instances:
            self.assertEqual(len([c for c in client.calls if isinstance(c, tuple)]), 3)
        self.assertIsNone(result['task_success'])

    def test_transport_failure_stops_robot_and_closes_worker(self):
        self.config['instruction'] = 'disconnect'
        policy = self.policy()
        Client.instances = []
        Client.deny_right = Client.fail_right = Client.stale = False
        robot = DualAirbot(self.config['robot'], self.journal, sdk=SDK, arm_factory=Client)
        result = run_trial(self.config, 'execute', robot, Cameras(), policy, self.journal)
        self.assertEqual(result['submitted_action_pairs'], 0)
        self.assertEqual(result['status'], 'error')
        self.assertTrue(all('PLANNING_POS' in c.calls for c in Client.instances))
        self.assertTrue(policy.closed)
        self.assertIsNotNone(policy.process.poll())

    def test_remote_timeout_cannot_produce_late_actions(self):
        self.config['instruction'] = 'slow'
        policy = self.policy()
        self.config['inference_timeout_s'] = 0.05
        Client.instances = []
        Client.deny_right = Client.fail_right = Client.stale = False
        robot = DualAirbot(self.config['robot'], self.journal, sdk=SDK, arm_factory=Client)
        result = run_trial(self.config, 'execute', robot, Cameras(), policy, self.journal)
        self.assertIn('TimeoutError', result['error'])
        self.assertEqual(result['submitted_action_pairs'], 0)
        self.assertFalse(any(isinstance(c, tuple) for arm in Client.instances for c in arm.calls))
        self.assertIsNotNone(policy.process.poll())

    def test_worker_stops_while_inference_is_running_on_input_eof(self):
        self.config['instruction'] = 'slow'
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, bufsize=0)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        self.addCleanup(process.stdout.close)
        transmit(process.stdin, {'type': 'init', 'version': 1, 'config': self.config,
                                 'route': 'native', 'skill_id': None})
        self.assertEqual(receive(process.stdout)['type'], 'ready')
        transmit(process.stdin, {'type': 'predict', 'seq': 1, 'state': [0] * 14,
                                 'images': encode_images(Cameras().read()[0])})
        time.sleep(0.1)
        process.stdin.close()
        self.assertEqual(process.wait(timeout=2), 1)

    def test_wrong_response_sequence_and_nonfinite_actions_are_rejected(self):
        from real_robot.remote import validate_actions
        with self.assertRaisesRegex(ValueError, '有限'):
            validate_actions([[float('nan')] * 14])
        policy = self.policy()
        policy._exchange = lambda *_: {'type': 'prediction', 'seq': 999, 'actions': [], 'status': 'running'}
        with self.assertRaisesRegex(ValueError, '序号'):
            policy.predict(np.zeros(14), Cameras().read()[0])

    def test_ssh_command_quotes_paths_and_rejects_options_as_host(self):
        command = ssh_command('dawn@test', '/data/a b', '/env/py thon')
        self.assertEqual(command[-1], "cd '/data/a b' && exec '/env/py thon' -u scripts/run_policy_worker.py")
        with self.assertRaises(ValueError):
            ssh_command('-oProxyCommand=bad', '/repo', '/python')
        stream = io.BytesIO()
        transmit(stream, {'type': 'close'})
        stream.seek(0)
        self.assertEqual(receive(stream), {'type': 'close'})


if __name__ == '__main__':
    unittest.main()
