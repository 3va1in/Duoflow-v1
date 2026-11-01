"""三路 RealSense RGB 按序列号绑定；持续取帧，推理期间不积压旧画面。"""
import threading
import time

import numpy as np

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def list_cameras():
    import pyrealsense2 as rs
    return [{'serial': d.get_info(rs.camera_info.serial_number),
             'name': d.get_info(rs.camera_info.name)} for d in rs.context().query_devices()]


class CameraRig:
    def __init__(self, config, rs=None):
        if rs is None:
            import pyrealsense2 as rs
        self.rs, self.config = rs, config
        devices = config['serials']
        if set(devices) != set(CAMERAS) or any(not isinstance(x, str) or not x for x in devices.values()):
            raise ValueError('必须配置头部、左腕、右腕三路 RealSense 序列号字符串')
        if len(set(devices.values())) != 3:
            raise ValueError('三路相机序列号不能相同')
        available = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        missing = set(devices.values()) - available
        if missing:
            raise RuntimeError(f'RealSense 未连接或序列号错误: {sorted(missing)}；使用 --list-cameras 查看')
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.frames, self.errors, self.threads = {}, {}, []
        self.cleanup_errors = []
        for name, serial in devices.items():
            thread = threading.Thread(target=self._capture, args=(name, serial), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _capture(self, name, serial):
        started = False
        try:
            pipeline = self.rs.pipeline()
            stream = self.rs.config()
            stream.enable_device(serial)
            stream.enable_stream(self.rs.stream.color, self.config['width'], self.config['height'],
                                 self.rs.format.rgb8, self.config['fps'])
            pipeline.start(stream)
            started = True
            previous = -1
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError(f'{name}: 缺少 RGB 帧')
                received = time.monotonic()
                number = color.get_frame_number()
                if number <= previous:
                    raise RuntimeError(f'{name}: 相机帧号未递增')
                previous = number
                rgb = np.asanyarray(color.get_data()).copy()
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise RuntimeError(f'{name}: 不是 uint8 RGB 图像')
                metadata = {'serial': serial, 'received_monotonic': received, 'frame_number': number,
                            'device_timestamp_ms': color.get_timestamp(),
                            'timestamp_domain': str(color.get_frame_timestamp_domain())}
                with self.lock:
                    self.frames[name] = (rgb, metadata)
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception as exc:
                    with self.lock:
                        self.cleanup_errors.append(f'{name} pipeline.stop: {exc}')

    def read(self):
        with self.lock:
            if self.errors:
                raise RuntimeError(f'相机错误: {self.errors}')
            if len(self.frames) != 3:
                raise RuntimeError('相机尚无完整三路画面')
            rows = dict(self.frames)
        stamps = [row[1]['received_monotonic'] for row in rows.values()]
        if time.monotonic() - min(stamps) > self.config['max_age_s']:
            raise RuntimeError('相机画面过期')
        if max(stamps) - min(stamps) > self.config['max_skew_s']:
            raise RuntimeError('三路相机接收时差超限')
        return ({key: row[0].copy() for key, row in rows.items()},
                {key: dict(row[1]) for key, row in rows.items()})

    def wait_ready(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                ready = len(self.frames) == 3 or bool(self.errors)
            if ready:
                self.read()
                return
            time.sleep(0.02)
        raise TimeoutError('5 秒内未收到完整三路相机画面')

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1.2)
        with self.lock:
            errors = list(self.cleanup_errors)
        if any(t.is_alive() for t in self.threads):
            errors.append('相机读取线程未及时退出')
        return errors
