"""使用已安装的官方 5.1.6 protobuf，与本机模拟 gRPC 服务交互；不连接机械臂。"""
from concurrent.futures import ThreadPoolExecutor
import time
import unittest

import numpy as np

from real_robot.airbot import Arm516, load_sdk


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        sdk = load_sdk()
        self.sdk, self.requests = sdk, []
        self.mode = sdk.RobotMode.PLANNING_POS.value
        self.reject = False
        self.feedback_enabled = True
        self.state_delay = 0
        self.position = np.zeros(7)
        owner = self

        class Control(sdk.arm_pb2_grpc.ControlServiceServicer):
            def SetMode(self, request, context):
                owner.mode = request.modes[0]
                return sdk.arm_pb2.SetModeResponse(modes=request.modes)

            def SetState(self, requests, context):
                for request in requests:
                    owner.requests.append(request)
                    if owner.reject:
                        yield sdk.arm_pb2.SetStateResponse(status=sdk.arm_pb2.Status.REJECTED)
                        return
                    owner.position = np.r_[request.commands[0].joint_states.position,
                                           request.commands[1].joint_states.position]
                    yield sdk.arm_pb2.SetStateResponse(status=sdk.arm_pb2.Status.RUNNING)

        class Feedback(sdk.arm_pb2_grpc.FeedbackServiceServicer):
            def GetMode(self, request, context):
                return sdk.arm_pb2.GetModeResponse(modes=[owner.mode])

            def GetState(self, request, context):
                while context.is_active():
                    if owner.feedback_enabled:
                        yield sdk.arm_pb2.GetStateResponse(state=sdk.types_pb2.JointStates(
                            name=[f'joint{i}' for i in range(1, 7)] + ['gripper_mapping_controller'],
                            position=owner.position.tolist()))
                    time.sleep(0.005)

        class Utils(sdk.utils_pb2_grpc.UtilsServiceServicer):
            def GetState(self, request, context):
                time.sleep(owner.state_delay)
                return sdk.utils_pb2.GetStateResponse(state=sdk.State.IDLE.value)

        self.server = sdk.grpc.server(ThreadPoolExecutor(max_workers=6))
        sdk.arm_pb2_grpc.add_ControlServiceServicer_to_server(Control(), self.server)
        sdk.arm_pb2_grpc.add_FeedbackServiceServicer_to_server(Feedback(), self.server)
        sdk.utils_pb2_grpc.add_UtilsServiceServicer_to_server(Utils(), self.server)
        port = self.server.add_insecure_port('127.0.0.1:0')
        self.server.start()
        self.addCleanup(lambda: self.server.stop(0).wait(2))
        self.arm = Arm516({'host': '127.0.0.1', 'port': port},
                          {'rpc_timeout_ms': 100, 'max_state_age_s': 0.2}, sdk)
        self.addCleanup(self.arm.close)
        self.wait_for(lambda: self.arm.latest is not None)

    def wait_for(self, condition):
        deadline = time.monotonic() + 2
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(condition())

    def test_official_wire_messages_feedback_and_stream_shutdown(self):
        self.assertEqual(self.arm.state(), self.sdk.State.IDLE)
        self.assertEqual(self.arm.mode(), self.sdk.RobotMode.PLANNING_POS)
        self.arm.begin()
        target = np.array([0.01] * 6 + [0.02])
        self.arm.send(target)
        self.wait_for(lambda: len(self.requests) == 1 and np.allclose(self.arm.read(), target))
        request = self.requests[0]
        self.assertEqual(list(request.component_names), ['arm', 'eef'])
        self.assertEqual(list(request.commands[0].joint_states.name), [f'joint{i}' for i in range(1, 7)])
        self.assertEqual(list(request.commands[1].joint_states.name), ['eef'])
        self.assertTrue(request.request_id)
        self.arm.switch(self.sdk.RobotMode.PLANNING_POS)
        self.assertIsNone(self.arm.motion_thread)
        self.assertEqual(self.mode, self.sdk.RobotMode.PLANNING_POS.value)
        self.arm.begin()
        self.arm.send(target * 2)
        self.wait_for(lambda: len(self.requests) == 2 and np.allclose(self.arm.read(), target * 2))
        self.assertEqual(len(self.requests), 2)

    def test_rejected_stream_stale_feedback_and_rpc_deadline(self):
        self.reject = True
        self.arm.begin()
        try:
            self.arm.send(np.zeros(7))
        except RuntimeError:
            pass  # 拒绝可在 send 返回之前或之后由响应线程收到。
        self.wait_for(lambda: self.arm.motion_error is not None)
        with self.assertRaisesRegex(RuntimeError, 'REJECTED'):
            self.arm.read()
        self.arm.stop_stream()
        self.arm.motion_error = None
        self.feedback_enabled = False
        time.sleep(0.3)
        with self.assertRaisesRegex(RuntimeError, '过期'):
            self.arm.read()
        self.state_delay = 0.3
        started = time.monotonic()
        with self.assertRaises(self.sdk.grpc.RpcError) as caught:
            self.arm.state()
        self.assertEqual(caught.exception.code(), self.sdk.grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertLess(time.monotonic() - started, 0.25)


if __name__ == '__main__':
    unittest.main()
