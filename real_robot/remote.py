"""经已有 SSH 标准输入/输出传输观测与预测；每个回合独立进程，不重连或重发。"""
import base64
import io
import json
import os
from pathlib import PurePosixPath
import select
import shlex
import struct
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from real_robot.airbot import vector
from real_robot.cameras import CAMERAS

MAX_MESSAGE = 32 * 1024 * 1024
POLICY_FIELDS = ('bundle', 'task', 'instruction', 'seed', 'max_steps', 'planner_call_limit',
                 'allow_cloud_observations', 'astra_request_root', 'inference_timeout_s')


def _wait(stream, writing, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('远端策略传输超时；本回合不重发')
    ready = select.select([] if writing else [stream], [stream] if writing else [], [], remaining)
    if not (ready[1] if writing else ready[0]):
        raise TimeoutError('远端策略传输超时；本回合不重发')


def _read(stream, count, deadline):
    parts = []
    while count:
        if deadline is None:
            part = stream.read(count)
        else:
            _wait(stream, False, deadline)
            try:
                part = os.read(stream.fileno(), count)
            except BlockingIOError:
                continue
        if not part:
            raise ConnectionError('远端策略连接已关闭')
        parts.append(part)
        count -= len(part)
    return b''.join(parts)


def receive(stream, deadline=None):
    size = struct.unpack('!I', _read(stream, 4, deadline))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError('远端协议头非法；请确认两端代码一致，SSH 登录脚本未向 stdout 输出文字')
    message = json.loads(_read(stream, size, deadline))
    if not isinstance(message, dict):
        raise ValueError('远端消息必须为 JSON 对象')
    return message


def transmit(stream, message, deadline=None):
    payload = json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError('观测消息超过 32 MiB')
    data = memoryview(struct.pack('!I', len(payload)) + payload)
    while data:
        if deadline is None:
            count = stream.write(data)
        else:
            _wait(stream, True, deadline)
            try:
                count = os.write(stream.fileno(), data[:65536])
            except BlockingIOError:
                continue
        if not count:
            raise ConnectionError('远端策略写入失败')
        data = data[count:]
    stream.flush()


def encode_images(images):
    if set(images) != set(CAMERAS):
        raise ValueError('必须传输完整三路 RGB')
    encoded = {}
    for name in CAMERAS:
        rgb = np.asarray(images[name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f'{name}: 需要 HWC uint8 RGB')
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format='PNG', compress_level=1)
        encoded[name] = base64.b64encode(buffer.getvalue()).decode('ascii')
    return encoded


def decode_images(encoded):
    if not isinstance(encoded, dict) or set(encoded) != set(CAMERAS):
        raise ValueError('缺少三路 RGB')
    images = {}
    for name in CAMERAS:
        with Image.open(io.BytesIO(base64.b64decode(encoded[name], validate=True))) as image:
            if image.format != 'PNG' or image.mode != 'RGB' or image.width * image.height > 4096**2:
                raise ValueError(f'{name}: 非 RGB PNG 或分辨率过大')
            images[name] = np.array(image, copy=True)
    return images


def validate_actions(value):
    actions = np.asarray(value, dtype=np.float64)
    if actions.shape == (0,):
        actions = actions.reshape(0, 14)
    if (actions.ndim != 2 or actions.shape[1] != 14 or len(actions) > 50 or
            not np.isfinite(actions).all()):
        raise ValueError('远端预测必须为有限的 [H,14]，H<=50')
    return actions


def ssh_command(host, repository, python):
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('remote-host 需为 SSH 别名或 user@host')
    if not PurePosixPath(repository).is_absolute() or not PurePosixPath(python).is_absolute():
        raise ValueError('remote-repo 与 remote-python 需为 5090 上的绝对路径')
    command = ('cd ' + shlex.quote(repository) + ' && exec ' +
               shlex.join([python, '-u', 'scripts/run_policy_worker.py']))
    return ['ssh', '-T', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
            '-o', 'ServerAliveCountMax=3', host, command]


class RemotePolicy:
    def __init__(self, command, config, route, skill_id, journal, startup_timeout=600):
        if route == 'duoflow' and config.get('allow_cloud_observations') is not True:
            raise ValueError('DuoFlow 会向 Astra 发送真机画面；请设置 allow_cloud_observations=true')
        self.journal, self.timeout = journal, config['inference_timeout_s']
        self.sequence, self.closed = 0, False
        self.busy = threading.Event()
        self.stderr = (journal.root / 'remote_stderr.log').open('wb')
        self.process = None
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.stderr, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            policy_config = {key: config[key] for key in POLICY_FIELDS}
            request = {'type': 'init', 'version': 1, 'config': policy_config,
                       'route': route, 'skill_id': skill_id}
            response = self._exchange(request, startup_timeout)
            if response.get('type') != 'ready' or response.get('version') != 1:
                raise ValueError('远端未返回 ready/version=1')
            if response.get('config') != policy_config or response.get('route') != route:
                raise ValueError('两端策略配置不一致')
            journal({'event': 'remote_ready', 'command': command, 'remote': response})
        except BaseException:
            self.close()
            raise

    def _exchange(self, message, timeout):
        if self.closed or self.busy.is_set():
            raise RuntimeError('远端回合已关闭或仍有在途请求')
        self.busy.set()
        try:
            deadline = time.monotonic() + timeout
            transmit(self.process.stdin, message, deadline)
            response = receive(self.process.stdout, deadline)
            if response.get('type') == 'error':
                raise RuntimeError(f"远端策略错误: {response.get('error')}")
            return response
        finally:
            self.busy.clear()

    def predict(self, state, images):
        self.check_connection()
        self.sequence += 1
        started = time.monotonic()
        request = {'type': 'predict', 'seq': self.sequence, 'state': vector(state, 'model_state').tolist(),
                   'images': encode_images(images)}
        response = self._exchange(request, self.timeout)
        if response.get('type') != 'prediction' or response.get('seq') != self.sequence:
            raise ValueError('远端响应序号不匹配；不执行动作')
        status = response.get('status')
        if not isinstance(status, str) or not status:
            raise ValueError('远端缺少执行状态')
        actions = validate_actions(response.get('actions'))
        self.journal({'event': 'remote_prediction', 'seq': self.sequence,
                      'roundtrip_s': time.monotonic() - started, 'remote_status': status})
        return actions, status

    def check_connection(self):
        if self.closed or self.process.poll() is not None:
            raise ConnectionError('本回合远端策略已退出；不继续发送动作')

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is not None:
            if process.poll() is None and not self.busy.is_set():
                try:
                    transmit(process.stdin, {'type': 'close'}, time.monotonic() + 1)
                    process.wait(timeout=3)
                except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired):
                    pass  # 下方终止本次 SSH 子进程；不重试策略请求。
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()
            self.journal({'event': 'remote_closed', 'returncode': process.returncode})
        self.stderr.close()
