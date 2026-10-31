"""运行原有控制逻辑测试，不加载模型、访问网络或连接机器人。"""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'method'))
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_astra_reentry',
    'test_native_fallback',
    'test_execution_outcomes',
])
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
