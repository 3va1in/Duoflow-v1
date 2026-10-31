# Ubuntu 20.04 / RTX 5090 / ChatGPT 账号接入

适用硬件信息：RTX 5090，Ubuntu 20.04.6，NVIDIA 驱动 580.82.09，有可登录桌面。系统 `python` 指向 2.7.18 不影响独立环境。DuoFlow 使用 Python 3.11；GPU 包用 PyTorch 2.7.1 / torchvision 0.22.1 / cu128。`nvidia-smi` 的 CUDA 13.0 表示驱动支持能力，无需将它改为 12.8。

## 1. 独立环境

已有 Conda/Miniforge 可直接创建环境。没有时先从 [Miniforge 官方仓库](https://github.com/conda-forge/miniforge) 安装 Linux x86_64 版；不要覆盖已有安装，也不要替换 `/usr/bin/python`。

仅在没有 Conda、且 `~/miniforge3` 不存在时执行：

```bash
curl -fL -o /tmp/Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

以后新开终端，先执行上面的 `source` 再激活环境；已有 Conda 则沿用原来的激活方式。

```bash
conda create -n duoflow -c conda-forge python=3.11 nodejs=22 -y
conda activate duoflow
cd /你的路径/Duoflow-v1
git pull --ff-only

python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-control.txt -r requirements-model.txt
source scripts/env.sh
python -m pip check
python scripts/check_controller.py
```

检查真实 GPU 运算：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(64, 64, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('GPU calculation:', y.shape, bool(torch.isfinite(y).all()))
PY
```

这不加载 DuoFlow 权重；权重目录与真机适配见 [README](../README.md) 和 [真机接口](REAL_ROBOT.md)。无须安装 RoboTwin/SAPIEN。

## 2. 安装 Astra 登录桥

在当前已登录的 Ubuntu 桌面终端执行，不使用 root：

```bash
sudo apt-get update
sudo apt-get install -y git libsecret-tools gnome-keyring
conda activate duoflow
cd /你的路径/Duoflow-v1
bash scripts/setup_astra.sh
```

安装脚本需要 Node.js >=22.12，只下载固定版本的官方 Sign in with ChatGPT DevKit、应用图像与结构化回执适配并构建 local SDK，不安装 macOS 示例应用。依赖说明及非商业许可见 [NOTICE](../astra_bridge/NOTICE.md)。

## 3. 用本账号授权 DuoFlow

```bash
node astra_bridge/cli.mjs login
```

在浏览器打开终端显示的链接，登录与现有 Codex 相同的 ChatGPT 账号、选择相同工作区，授权 **DuoFlow Astra** 使用 ChatGPT 计划用量。成功后应有 `status: connected` 和 `sharing: true`。

```bash
node astra_bridge/cli.mjs status
node astra_bridge/cli.mjs models
```

模型列表需包含 `gpt-6-astra`；否则程序停止，不会换模型。列表只是权限初查，实际完成一次请求才证明本次推理可用。

这走官方 ChatGPT plan usage，不需要 Platform API key；同账号/工作区的计划限制依然生效，新增主机不会增加额度。第三方应用的计划用量权限与限制以授权页面和 [ChatGPT 用量设置](https://chatgpt.com/settings/usage) 为准，不承诺等同于 Codex UI 显示的每一个额度桶。Codex CLI 登录与 DuoFlow 授权是不同入口，本桥不读取 `~/.codex/auth.json`。

凭据在 `~/.config/duoflow-astra/` 中加密保存，密钥在当前 Ubuntu 用户的 Secret Service 密钥环；不要把这些文件提交 Git。若提示密钥环不可用，先登录 Ubuntu 桌面并解锁“密码和密钥”中的登录密钥环，再在该桌面终端重试；不会回退到明文保存。

若从 SSH 发起登录且系统密钥环已可用，可在本地建立回调转发后运行登录命令：

```bash
ssh -L 14555:127.0.0.1:14555 用户名@新主机
```

此时把登录链接打开在本地浏览器。无需复制旧 Mac 的登录文件或 host ID。登录命令不会发起推理。

## 4. 启动桥并做一次实际连接检查

终端 A（同一 Ubuntu 用户，保持运行）：

```bash
conda activate duoflow
cd /你的路径/Duoflow-v1
node astra_bridge/cli.mjs serve \
  --request-root "$PWD/runs/astra_real" --max-requests 100
```

100 是该请求目录中累计领取数量的上限（含失败，不随重启归零），供首次连接和小规模调试。并发为 1；不会自动重试、补测或切换其他账号。Ctrl-C 等待当前请求结束后退出。不能同时启动同目录的多个桥。

终端 B：

```bash
conda activate duoflow
cd /你的路径/Duoflow-v1
source scripts/env.sh
python scripts/check_astra_connection.py --request-root "$PWD/runs/astra_real"
```

该命令消耗一次真实模型请求，使用三张合成黑色图像、零状态和连接检查文本，不发送真实相机图像、不加载动作权重、不连接机器人。成功时打印 `skill_id`、模型、usage 与延迟。失败时保留错误，不会自动再次调用。

实际控制器继续使用原客户端：

```python
from pathlib import Path
from astra_reentry import AstraNativeFallbackBackend

root = Path('runs/astra_real').resolve()
planner = AstraNativeFallbackBackend(
    transport='siwc_files',
    request_root=str(root),
    audit_dir=str(root / 'planner_audit'),
)
```

把 `planner` 交给既有 `ReentryController`；相机、actor、技能目录与 native 控制器仍需按真机配置接入。桥只负责推理，不执行伺服动作。输入的 RGB、状态、技能卡、历史和训练记忆会发到 OpenAI；只向控制器提供允许上传的观测。

## 5. 故障与用量

- CLI 失败时输出 JSON，保留命令、SDK 错误码、脱敏说明，以及可用的 HTTP `status`、`requestId`、`param` 和 `responseShape`。若旧版本只显示 `api_error`，先 `git pull --ff-only origin main`；排查时提供错误 JSON，不提供授权链接或登录文件。仅凭 `api_error` 不能判断额度耗尽。
- 每个请求写入 `requests/<id>.json`，桥保存唯一 `claims/<id>.json`，完成后原子写入 `responses/<id>.json`。响应保留真实 usage 与延迟。
- 网络、登录、额度、未知错误都会写回错误并停止桥。修正原因后可重启，已经有响应的 ID 不再发送。旧控制器回合已中止，需要由操作者决定新回合。
- 若有 claim 却无 response，桥报告 `prior_request_outcome_unknown` 并停下；先检查旧进程与原请求结果，不删除 claim 来重发。
- API 故障不会选择 native 回退；真实机器人驱动必须在等待/异常时处理保持和停止。
- 取消本机授权：`node astra_bridge/cli.mjs disconnect`；账号侧可在用量设置管理授权。

官方依据：[ChatGPT plan usage](https://developers.openai.com/siwc/token-sharing-open-source)、[登录流程](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)、[模型与推理](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)、[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 2.7.1 安装](https://pytorch.org/get-started/previous-versions/)。
