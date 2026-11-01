import copy
import builtins
from importlib.metadata import PackageNotFoundError
import json
from enum import Enum
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np

from real_robot.airbot import DualAirbot, JointMapping, MotionLimits, load_sdk
from real_robot.policy import bundle_paths, model_config, model_observation
from real_robot.runner import Journal, run_trial, validate_config

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    config = json.loads((ROOT / 'examples/airbot_dual.json').read_text())
    config['calibration']['verified'] = True
    config.update(max_steps=3, frequency_hz=100, inference_timeout_s=0.2)
    config['limits'] = {'position_min': [-1] * 14, 'position_max': [1] * 14,
                        'max_velocity': [1] * 14, 'max_tracking_error': [0.1] * 14}
    return config


class State(Enum):
    IDLE = 3
    APPLOADED = 5


class RobotMode(Enum):
    PLANNING_POS = 10
    SERVO_JOINT_POS = 20


SDK = SimpleNamespace(State=State, RobotMode=RobotMode)


class Client:
    instances = []
    deny_right = fail_right = stale = False

    def __init__(self, endpoint, config, sdk):
        self.port = endpoint['port']
        self.position = np.zeros(7)
        self.calls = []
        self.latest, self.feedback_error = True, None
        self.instances.append(self)

    def read(self):
        if self.stale:
            raise RuntimeError('关节反馈过期')
        return self.position.copy()

    def state(self):
        return State.APPLOADED if self.deny_right and self.port == 50052 else State.IDLE

    def mode(self):
        return RobotMode.PLANNING_POS

    def switch(self, value):
        self.calls.append(value.name)

    def begin(self):
        self.calls.append('begin')

    def send(self, target):
        self.calls.append(('move', target.tolist()))
        if self.fail_right and self.port == 50052:
            raise RuntimeError('右臂命令流断连')
        self.position = target.copy()

    def close(self):
        self.calls.append('close')


class Cameras:
    fail = False
    def wait_ready(self):
        pass

    def read(self):
        if self.fail:
            raise RuntimeError('相机断连')
        return ({name: np.zeros((8, 8, 3), dtype=np.uint8)
                 for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}, {})

    def close(self):
        return []


class Policy:
    def __init__(self, actions=None, delay=0):
        self.actions = np.ones((3, 14)) * 0.005 if actions is None else actions
        self.delay = delay
        self.observations = []

    def predict(self, state, images):
        self.observations.append(state.copy())
        time.sleep(self.delay)
        return self.actions.copy(), 'running'


class SdkLoadingTests(unittest.TestCase):
    def test_missing_sdk_identifies_current_python(self):
        with patch('importlib.metadata.version', side_effect=PackageNotFoundError('airbot_py')):
            with self.assertRaises(RuntimeError) as caught:
                load_sdk()
        self.assertIn(sys.executable, str(caught.exception))
        self.assertIn('未安装', str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, PackageNotFoundError)

    def test_wrong_version_does_not_attempt_import(self):
        with patch('importlib.metadata.version', return_value='5.2.0'):
            with self.assertRaisesRegex(RuntimeError, '已安装 5.2.0'):
                load_sdk()

    def test_dependency_error_keeps_original_reason(self):
        original_import = builtins.__import__
        failure = ModuleNotFoundError("No module named 'grpc'", name='grpc')

        def import_sdk(name, *args, **kwargs):
            if name == 'airbot_py.arm':
                raise failure
            return original_import(name, *args, **kwargs)

        with patch('importlib.metadata.version', return_value='5.1.6'):
            with patch('builtins.__import__', side_effect=import_sdk):
                with self.assertRaises(RuntimeError) as caught:
                    load_sdk()
        self.assertIn("No module named 'grpc'", str(caught.exception))
        self.assertIn('已安装，但导入失败', str(caught.exception))
        self.assertIs(caught.exception.__cause__, failure)


class AirbotTests(unittest.TestCase):
    def setUp(self):
        Client.instances = []
        Client.deny_right = Client.fail_right = Client.stale = False
        self.config = configuration()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Journal(Path(self.temp.name) / 'run')
        self.robot = DualAirbot(self.config['robot'], self.log, sdk=SDK, arm_factory=Client)

    def run_mode(self, mode='execute', policy=None, cameras=None):
        return run_trial(self.config, mode, self.robot, cameras or Cameras(), policy or Policy(), self.log)

    def test_shadow_and_observe_never_acquire_or_move(self):
        result = self.run_mode('shadow')
        self.assertEqual(result['status'], 'shadow_complete')
        self.assertIsNone(result['task_success'])
        self.assertTrue(all(c.calls == ['close'] for c in Client.instances))

    def test_execute_combines_grippers_and_joints_then_releases(self):
        result = self.run_mode()
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['submitted_action_pairs'], 3)
        for client in Client.instances:
            moves = [x for x in client.calls if isinstance(x, tuple)]
            self.assertEqual(len(moves), 3)
            self.assertEqual(moves[0][1][6], 0.005)
            self.assertEqual(client.calls[-2:], ['PLANNING_POS', 'close'])
        self.assertIsNone(result['task_success'])

    def test_limits_reject_entire_chunk(self):
        result = self.run_mode(policy=Policy(np.ones((3, 14)) * 0.5))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['submitted_action_pairs'], 0)
        for client in Client.instances:
            self.assertFalse(any(isinstance(x, tuple) for x in client.calls))
            self.assertIn('PLANNING_POS', client.calls)

    def test_active_external_app_is_not_taken_over(self):
        Client.deny_right = True
        result = self.run_mode()
        self.assertEqual(result['status'], 'error')
        self.assertTrue(all(c.calls == ['close'] for c in Client.instances))
        self.assertEqual(self.robot.locks, [])

    def test_one_arm_command_failure_stops_both_and_logs_partial_submission(self):
        Client.fail_right = True
        result = self.run_mode()
        self.assertEqual(result['submitted_action_pairs'], 0)
        events = [json.loads(x) for x in (self.log.root / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([x['arm'] for x in events if x['event'] == 'command_submitted'], [0])
        self.assertTrue(all('PLANNING_POS' in c.calls for c in Client.instances))

    def test_inference_timeout_never_executes_late_prediction(self):
        self.config['inference_timeout_s'] = 0.01
        result = self.run_mode(policy=Policy(delay=0.05))
        time.sleep(0.07)
        self.assertIn('TimeoutError', result['error'])
        self.assertFalse(any(isinstance(x, tuple) for c in Client.instances for x in c.calls))

    def test_camera_and_state_faults_prevent_motion(self):
        cameras = Cameras()
        cameras.fail = True
        result = self.run_mode(cameras=cameras)
        self.assertIn('相机断连', result['error'])
        self.assertFalse(any(isinstance(x, tuple) for c in Client.instances for x in c.calls))
        Client.stale = True
        with self.assertRaisesRegex(RuntimeError, '过期'):
            self.robot.read()

    def test_mapping_and_explicit_execution_configuration(self):
        calibration = copy.deepcopy(self.config['calibration'])
        calibration['model_from_robot'] = list(reversed(range(14)))
        calibration['scale'] = [-2] * 14
        calibration['offset'] = [0.3] * 14
        mapping = JointMapping(calibration)
        state = np.linspace(-0.5, 0.5, 14)
        np.testing.assert_allclose(mapping.to_robot(mapping.to_model(state)[None])[0], state, atol=1e-7)
        example = json.loads((ROOT / 'examples/airbot_dual.json').read_text())
        validate_config(example, 'observe')
        with self.assertRaisesRegex(ValueError, '标定'):
            validate_config(example, 'execute')
        with self.assertRaisesRegex(RuntimeError, '跳变'):
            MotionLimits(self.config['limits'], 100).check_chunk(np.ones((2, 14)), np.zeros(14))

    def test_duoflow_uses_existing_full_native_and_correct_observations(self):
        paths = bundle_paths(ROOT / 'pretrained/simulation', self.config['task'])
        with self.assertRaisesRegex(ValueError, '真机画面'):
            model_config(self.config, 'duoflow', paths, self.log.root, ROOT)
        self.config.update(allow_cloud_observations=True, astra_request_root=str(self.log.root / 'bridge'))
        cfg = model_config(self.config, 'duoflow', paths, self.log.root, ROOT)
        self.assertTrue(cfg['native_fallback'])
        self.assertEqual(cfg['execution_contract'], 'reentry_atomic_v1')
        self.assertEqual(cfg['astra_transport'], 'siwc_files')
        obs = model_observation(np.arange(14), Cameras().read()[0], 'test')
        np.testing.assert_array_equal(obs['state']['right_arm_joint_state'], np.arange(7, 13))
        self.assertEqual(obs['state']['left_ee_joint_state'][0], 6)


if __name__ == '__main__':
    unittest.main()
