import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np

from real_robot.cameras import CameraRig, CAMERAS


class StreamConfig:
    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        self.stream = args


class Pipeline:
    def __init__(self, owner):
        self.owner, self.number, self.stopped = owner, 0, False
        owner.pipelines.append(self)

    def start(self, config):
        self.config = config

    def wait_for_frames(self, timeout_ms):
        time.sleep(0.01)
        self.number += 1
        return self

    def get_color_frame(self):
        return self

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.number * 10.0

    def get_frame_timestamp_domain(self):
        return 'hardware_clock'

    def get_data(self):
        # RGB 三通道不同，能够检测误加 BGR 转换。
        return np.full((4, 6, 3), [int(self.config.serial), 20, 30], dtype=np.uint8)

    def stop(self):
        self.stopped = True


class RealSense:
    camera_info = SimpleNamespace(serial_number='serial')
    stream, format = SimpleNamespace(color='color'), SimpleNamespace(rgb8='rgb8')
    config = StreamConfig

    def __init__(self):
        self.pipelines = []

    def context(self):
        return SimpleNamespace(query_devices=lambda: [
            SimpleNamespace(get_info=lambda key, s=s: s) for s in ('1', '2', '3')])

    def pipeline(self):
        return Pipeline(self)


class CameraTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'examples/airbot_dual.json'
        self.config = json.loads(path.read_text())['cameras']
        self.config['serials'] = dict(zip(CAMERAS, ('3', '1', '2')))

    def test_serial_binding_rgb_receipts_and_all_pipelines_stop(self):
        rs = RealSense()
        rig = CameraRig(self.config, rs)
        try:
            rig.wait_ready()
            images, receipts = rig.read()
            for name, serial in self.config['serials'].items():
                np.testing.assert_array_equal(images[name][0, 0], [int(serial), 20, 30])
                self.assertEqual(receipts[name]['serial'], serial)
                self.assertGreater(receipts[name]['frame_number'], 0)
                self.assertEqual(receipts[name]['timestamp_domain'], 'hardware_clock')
            self.assertTrue(all(p.config.stream[-2] == 'rgb8' for p in rs.pipelines))
        finally:
            self.assertEqual(rig.close(), [])
        self.assertTrue(all(p.stopped for p in rs.pipelines))
        with rig.lock:
            for _, receipt in rig.frames.values():
                receipt['received_monotonic'] = time.monotonic() - 10
        with self.assertRaisesRegex(RuntimeError, '过期'):
            rig.read()

    def test_duplicate_or_missing_device_is_rejected_before_streaming(self):
        rs = RealSense()
        self.config['serials']['cam_high'] = '1'
        with self.assertRaisesRegex(ValueError, '相同'):
            CameraRig(self.config, rs)
        self.config['serials']['cam_high'] = '999'
        with self.assertRaisesRegex(RuntimeError, '未连接'):
            CameraRig(self.config, rs)
        self.assertEqual(rs.pipelines, [])


if __name__ == '__main__':
    unittest.main()
