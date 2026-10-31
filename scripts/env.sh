#!/usr/bin/env bash
# 在仓库根目录或任意位置 source 此文件；不修改全局环境文件。
DUOFLOW_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DUOFLOW_CODE_ROOT
export PYTHONPATH="$DUOFLOW_CODE_ROOT/method:$DUOFLOW_CODE_ROOT/training:$DUOFLOW_CODE_ROOT/runtime:$DUOFLOW_CODE_ROOT/runtime/XPolicyLab/policy/TwinTimeFM/vendor/backbone_lib${PYTHONPATH:+:$PYTHONPATH}"
export TWINTIMEFM_ASSETS="${TWINTIMEFM_ASSETS:-$DUOFLOW_CODE_ROOT/models/base}"
