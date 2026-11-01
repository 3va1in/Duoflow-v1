"""AIRBOT 5.1.6 的官方 gRPC 消息；限时调用和可取消流，不修改厂商 SDK。"""
import time
import fcntl
import hashlib
from pathlib import Path
import queue
import threading
import uuid

import numpy as np


def load_sdk():
    from importlib.metadata import version
    try:
        installed = version('airbot_py')
        if installed != '5.1.6':
            raise RuntimeError(f'本驱动对应 airbot_py 5.1.6，已安装 {installed}')
        import airbot_py.arm as sdk
    except ImportError as exc:
        raise RuntimeError('请在当前环境安装厂商 airbot_py-5.1.6 wheel；无需更换服务端或固件') from exc
    return sdk


def vector(value, name, length=14):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} 必须为 {length} 个有限数值')
    return result


class JointMapping:
    def __init__(self, config):
        self.order = config['model_from_robot']
        if (not isinstance(self.order, list) or
                any(type(x) is not int for x in self.order) or
                sorted(self.order) != list(range(14))):
            raise ValueError('model_from_robot 必须是 0..13 的排列')
        self.scale = vector(config['scale'], 'calibration.scale')
        self.offset = vector(config['offset'], 'calibration.offset')
        if np.any(np.abs(self.scale) < 1e-8):
            raise ValueError('标定 scale 不能为零')

    def to_model(self, robot_state):
        return (vector(robot_state, 'robot_state')[self.order] * self.scale + self.offset).astype(np.float32)

    def to_robot(self, model_actions):
        actions = np.asarray(model_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
            raise ValueError('模型动作必须为有限的 [H,14] 数组')
        result = np.empty_like(actions)
        result[:, self.order] = (actions - self.offset) / self.scale
        return result


class MotionLimits:
    def __init__(self, config, frequency):
        self.low = vector(config['position_min'], 'position_min')
        self.high = vector(config['position_max'], 'position_max')
        self.velocity = vector(config['max_velocity'], 'max_velocity')
        self.tracking = vector(config['max_tracking_error'], 'max_tracking_error')
        if (np.any(self.low >= self.high) or np.any(self.velocity <= 0) or
                np.any(self.tracking <= 0)):
            raise ValueError('运动范围、速度或跟踪误差限值非法')
        self.delta = self.velocity / frequency

    def check_state(self, state):
        state = vector(state, 'feedback')
        if np.any(state < self.low) or np.any(state > self.high):
            raise RuntimeError('实测关节/夹爪位置越界')

    def check_tracking(self, state, target):
        self.check_state(state)
        if np.any(np.abs(state - target) > self.tracking):
            raise RuntimeError('关节/夹爪跟踪误差超过配置上限')

    def check_chunk(self, actions, current):
        self.check_state(current)
        if not len(actions) or np.any(actions < self.low) or np.any(actions > self.high):
            raise RuntimeError('预测动作越界或为空；本动作块不下发')
        difference = np.diff(np.vstack([current, actions]), axis=0)
        if np.any(np.abs(difference) > self.delta + 1e-8):
            raise RuntimeError('预测动作跳变超过配置速度；本动作块不下发，不自动裁剪')


class Arm516:
    def __init__(self, endpoint, config, sdk):
        self.sdk = sdk
        self.rpc_s = config['rpc_timeout_ms'] / 1000
        self.max_age = config['max_state_age_s']
        self.client_id = str(uuid.uuid4())
        self.channel = sdk.grpc.insecure_channel(f"{endpoint['host']}:{endpoint['port']}",
                                                 options=[('grpc.enable_http_proxy', 0)])
        self.control = sdk.arm_pb2_grpc.ControlServiceStub(self.channel)
        self.feedback = sdk.arm_pb2_grpc.FeedbackServiceStub(self.channel)
        self.utils = sdk.utils_pb2_grpc.UtilsServiceStub(self.channel)
        self.lock = threading.Lock()
        self.latest, self.received, self.feedback_error = None, 0, None
        self.motion_error, self.motion = None, None
        self.motion_thread = None
        self.commands = queue.Queue(maxsize=1)
        self.motion_stop, self.closed = threading.Event(), threading.Event()
        request = self.request(sdk.arm_pb2.GetStateRequest)
        request.component_names[:] = ['arm']
        request.get_joint = True
        request.get_transform = False
        self.feedback_call = self.feedback.GetState(request)
        self.feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self.feedback_thread.start()

    def request(self, message):
        result = message()
        now = time.time_ns()
        result.stamp.sec, result.stamp.nanosec = divmod(now, 10**9)
        result.request_id = self.client_id
        return result

    def _feedback_loop(self):
        try:
            for feedback in self.feedback_call:
                with self.lock:
                    self.latest, self.received = feedback, time.monotonic()
                if self.closed.is_set():
                    break
            if not self.closed.is_set():
                raise RuntimeError('关节反馈流结束')
        except Exception as exc:
            if not self.closed.is_set():
                self.feedback_error = str(exc)

    def read(self):
        if self.feedback_error or self.motion_error:
            raise RuntimeError(f'AIRBOT 通信错误: {self.feedback_error or self.motion_error}')
        with self.lock:
            feedback, received = self.latest, self.received
        if feedback is None or time.monotonic() - received > self.max_age:
            raise RuntimeError('AIRBOT 关节反馈过期或尚未就绪')
        # 与 AIRBOTArm.get_joint_pos/get_eef_pos 使用完全相同的 SDK 关节名。
        positions = dict(zip(feedback.state.name, feedback.state.position))
        names = [f'joint{i}' for i in range(1, 7)] + ['gripper_mapping_controller']
        if any(name not in positions for name in names):
            raise RuntimeError(f'关节反馈缺少字段，实际为 {sorted(positions)}')
        return vector([positions[name] for name in names], 'AIRBOT feedback', 7)

    def state(self):
        response = self.utils.GetState(self.request(self.sdk.utils_pb2.GetStateRequest), timeout=self.rpc_s)
        return self.sdk.State(response.state)

    def mode(self):
        request = self.request(self.sdk.arm_pb2.GetModeRequest)
        request.component_names[:] = ['arm']
        response = self.feedback.GetMode(request, timeout=self.rpc_s)
        if len(response.modes) != 1:
            raise RuntimeError('没有唯一的机械臂控制模式回执')
        return self.sdk.RobotMode(response.modes[0])

    def switch(self, mode):
        stream_error = None
        try:
            self.stop_stream()
        except Exception as exc:
            stream_error = exc
        request = self.request(self.sdk.arm_pb2.SetModeRequest)
        request.component_names[:] = ['arm']
        request.modes[:] = [mode.value]
        response = self.control.SetMode(request, timeout=self.rpc_s)
        if list(response.modes) != [mode.value]:
            raise RuntimeError(f'切换 {mode.name} 失败')
        if stream_error is not None:
            raise RuntimeError(f'模式已切换，但旧命令流未正常退出: {stream_error}')

    def _requests(self, commands, stopped):
        while not stopped.is_set():
            try:
                request, submitted = commands.get(timeout=0.05)
            except queue.Empty:
                continue
            if stopped.is_set():
                return
            submitted.set()
            yield request

    def _responses(self, motion, stopped):
        try:
            for response in motion:
                if response.status != self.sdk.arm_pb2.Status.RUNNING:
                    raise RuntimeError(f'SERVO 状态={self.sdk.arm_pb2.Status.Name(response.status)}')
            if not stopped.is_set():
                raise RuntimeError('SERVO 命令流意外结束')
        except Exception as exc:
            if not stopped.is_set():
                self.motion_error = str(exc)

    def begin(self):
        self.switch(self.sdk.RobotMode.SERVO_JOINT_POS)
        # 每块独立队列/停止事件，已取消流的迭代器不能消费下一块命令。
        self.motion_stop = threading.Event()
        self.motion_error = None
        self.commands = queue.Queue(maxsize=1)
        self.motion = self.control.SetState(self._requests(self.commands, self.motion_stop))
        self.motion_thread = threading.Thread(target=self._responses,
                                              args=(self.motion, self.motion_stop), daemon=True)
        self.motion_thread.start()

    def send(self, target):
        if self.motion is None or self.motion_error:
            raise RuntimeError(f'SERVO 流不可用: {self.motion_error}')
        request = self.request(self.sdk.arm_pb2.SetStateRequest)
        request.component_names[:] = ['arm', 'eef']
        arm = self.sdk.types_pb2.JointStates()
        arm.name[:] = [f'joint{i}' for i in range(1, 7)]
        arm.position[:] = target[:6].tolist()
        eef = self.sdk.types_pb2.JointStates()
        eef.name[:] = ['eef']
        eef.position[:] = [float(target[6])]
        request.commands.add(joint_states=arm)
        request.commands.add(joint_states=eef)
        submitted = threading.Event()
        self.commands.put((request, submitted), timeout=self.rpc_s)
        if not submitted.wait(self.rpc_s):
            raise TimeoutError('SERVO 命令提交超时')
        if self.motion_error:
            raise RuntimeError(self.motion_error)
        # 这是 gRPC 请求迭代器消费回执，不是逐动作物理完成或服务端 ACK。

    def stop_stream(self):
        self.motion_stop.set()
        if self.motion is not None:
            self.motion.cancel()
        if self.motion_thread is not None:
            self.motion_thread.join(timeout=1)
            if self.motion_thread.is_alive():
                raise TimeoutError('SERVO 响应线程未退出')
        self.motion, self.motion_thread = None, None

    def close(self):
        self.closed.set()
        errors = []
        for cleanup in (self.stop_stream, self.feedback_call.cancel, self.channel.close):
            try:
                cleanup()
            except Exception as exc:
                errors.append(str(exc))
        self.feedback_thread.join(timeout=1)
        if self.feedback_thread.is_alive():
            errors.append('AIRBOT 反馈线程未退出')
        if errors:
            raise RuntimeError('; '.join(errors))


class DualAirbot:
    def __init__(self, config, log, sdk=None, arm_factory=Arm516):
        self.sdk = sdk if sdk is not None else load_sdk()
        self.log, self.config = log, config
        self.clients, self.owned, self.locks = [], [], []
        endpoints = [config[side] for side in ('left', 'right')]
        if endpoints[0] == endpoints[1]:
            raise ValueError('左右臂必须连接两个不同的 AIRBOT 服务端')
        try:
            for endpoint in endpoints:
                self.clients.append(arm_factory(endpoint, config, self.sdk))
            deadline = time.monotonic() + 3
            while any(c.latest is None and not c.feedback_error for c in self.clients) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        return np.concatenate([client.read() for client in self.clients])

    def acquire(self, limits):
        # 5.1.6 没有服务端租约；此锁只排斥本机使用同一入口的其他进程。
        for side in ('left', 'right'):
            endpoint = self.config[side]
            key = hashlib.sha256(f"{endpoint['host']}:{endpoint['port']}".encode()).hexdigest()[:20]
            handle = (Path('/tmp') / f'duoflow-airbot-{key}.lock').open('a')
            self.locks.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'{side}: 另一 DuoFlow 进程正使用该服务') from exc
        for client in self.clients:
            if client.state() != self.sdk.State.IDLE or client.mode() != self.sdk.RobotMode.PLANNING_POS:
                raise RuntimeError('执行前需 IDLE / PLANNING_POS；请先退出遥操作和其他控制程序')
        self.owned = [0, 1]
        self.log({'event': 'local_control_claimed', 'server_lease_available': False})

    def hold(self):
        for index in self.owned:
            self.clients[index].switch(self.sdk.RobotMode.PLANNING_POS)

    def begin_chunk(self):
        for index in self.owned:
            if self.clients[index].state() != self.sdk.State.IDLE:
                raise RuntimeError('AIRBOT 已退出 IDLE 状态')
            self.clients[index].begin()

    def send(self, target):
        if self.owned != [0, 1]:
            raise RuntimeError('未取得本次双臂执行资格')
        starts = []
        for index, client in enumerate(self.clients):
            starts.append(time.monotonic())
            part = target[index * 7:(index + 1) * 7]
            self.log({'event': 'command_attempt', 'arm': index, 'target': part.tolist()})
            client.send(part)
            self.log({'event': 'command_submitted', 'arm': index, 'server_ack': False})
        skew = starts[1] - starts[0]
        if skew > self.config['max_command_skew_s']:
            raise RuntimeError(f'双臂下发间隔过大: {skew:.3f}s')
        return skew

    def stop(self, emergency):
        errors = []
        for index in self.owned:
            try:
                self.clients[index].switch(self.sdk.RobotMode.PLANNING_POS)
            except Exception as exc:
                errors.append(f'arm={index} stop: {exc}')
        self.owned.clear()
        if emergency:
            self.log({'event': 'abort_stop', 'mechanism': 'cancel_stream_and_PLANNING_POS',
                      'hardware_emergency_stop_available': False})
        for handle in self.locks:
            handle.close()
        self.locks.clear()
        return errors

    def close(self):
        errors = []
        for client in self.clients:
            try:
                client.close()
            except Exception as exc:
                errors.append(str(exc))
        return errors
