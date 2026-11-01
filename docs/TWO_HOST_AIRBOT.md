# 机械臂电脑采集与执行，dawn / RTX 5090 推理

适用布局：两台 AIRBOT Play 和三路 D435i 连接设备电脑；DuoFlow 权重、CUDA 与 Astra 登录桥在另一台 5090 主机 dawn。设备电脑能通过 SSH 登录 dawn。

SSH 登录后运行的命令位于远端主机。普通 SSH 会话不会把设备电脑的相机自动接到 dawn，所以在 dawn 执行 `--list-cameras` 返回 `[]` 可以是正常结果。本方案将观测和动作作为数据通过 SSH 传输。

```text
设备电脑：三路 RGB + 实测关节
        ↓ 本机标定映射
        ── SSH：无损 RGB + 模型状态 ──→ dawn：DuoFlow / GPU / Astra
        ←──────── 模型动作块 ────────
        ↓ 本机逆映射、限位与跟踪检查
设备电脑：AIRBOT 动作执行 → 反馈 → 下一次观测
```

| 位置 | 运行内容 | 需要的安装 |
|---|---|---|
| 设备电脑 | `airbot_server`、相机采集、`run_airbot.py`、动作与停止处理 | Python 3.11、配套 `airbot_py 5.1.6`、`requirements-real.txt` |
| dawn / 5090 | 单回合远端策略进程、模型与可选 Astra 桥 | 已有 DuoFlow 模型环境、基础模型及技能权重 |

设备电脑不需要安装 CUDA、PyTorch 或下载基础大模型。dawn 不需要安装 AIRBOT SDK、RealSense 驱动或接入 USB 设备。两端使用同一版仓库；不需要另开推理 HTTP 端口。

## 1. 更新两端代码

两台电脑各自在仓库根目录执行：

```bash
git pull --ff-only origin main
```

如果设备电脑还没有仓库，只在设备电脑克隆一次：

```bash
git clone https://github.com/3va1in/Duoflow-v1.git
cd Duoflow-v1
```

设备电脑的 Python 环境需能导入当前厂商 `airbot_py 5.1.6`。在这个环境安装采集端依赖；不替换已有厂商服务端或固件：

```bash
python -m pip install -r requirements-real.txt
python -m pip show airbot_py
python scripts/run_airbot.py --list-cameras
```

以上命令在**设备电脑本机终端**执行，不能在 SSH 登录 dawn 后执行。若 SDK 装在别的 Python 环境，使用对应环境，或在独立 Python 3.11 环境安装同一个厂商 wheel。

### 当前环境找不到 AIRBOT SDK

`requirements-real.txt` 不包含厂商 wheel。dawn 或系统 Python 中安装过 SDK，也不会自动安装到设备电脑新建的 conda 环境。先在报错的同一个终端检查：

```bash
python -c "import sys; print(sys.executable)"
python -m pip show airbot_py
python -c "import airbot_py.arm"
```

若当前环境未安装，下载[厂商 5.1.6 SDK 包](https://docs.discover-robotics.com/document/assets/airbot-play/5.1.6/5.1.6.zip)，解压后用当前 Python 安装其中的 wheel。以下命令仅提取和安装 Python SDK，不运行包里的驱动安装或固件升级程序：

```bash
mkdir -p runs/vendor/airbot516
curl -fL --output runs/vendor/airbot516/5.1.6.zip \
  https://docs.discover-robotics.com/document/assets/airbot-play/5.1.6/5.1.6.zip
python -m zipfile -e runs/vendor/airbot516/5.1.6.zip runs/vendor/airbot516
python -m pip install runs/vendor/airbot516/5.1.6/airbot_py-5.1.6-py3-none-any.whl
python -c "import airbot_py.arm; print('AIRBOT SDK import OK')"
```

普通 wheel 安装会按厂商声明安装 `grpcio==1.70.0`、`grpcio-tools==1.70.0`、`typeguard==4.4.0` 等依赖，不要加 `--no-deps`。若 `pip show` 已显示 `5.1.6` 但 `import airbot_py.arm` 失败，以该导入命令的完整异常为准处理缺失依赖或版本冲突。不要用 `arm_sdk` 替代此入口所用的 `airbot_py 5.1.6`。

## 2. 设备电脑：填写相机和机械臂配置

关闭占用相机的 RealSense Viewer。创建配置时保留已有文件：

```bash
mkdir -p runs
cp -n examples/airbot_dual.json runs/airbot.json
```

编辑 `runs/airbot.json`：

- `cameras.serials`：填全局、左腕、右腕相机的真实序列号。
- `robot.left/right`：按设备电脑实际 AIRBOT 服务地址和端口填写。同机服务可保持 `127.0.0.1`，它指设备电脑。
- `bundle`：保持 `pretrained/simulation`，远端推理时相对于 **dawn 的仓库**解析；自定义绝对路径也应是 dawn 上的路径。
- `astra_request_root`：完整 DuoFlow 使用时填写 **dawn 上**的桥目录。
- `task`、`instruction`、`seed`、`max_steps`：本次实验设置；由设备端传给 dawn，避免两边各维护一份策略设置。

先读取一组实测观测：

```bash
python scripts/run_airbot.py --config runs/airbot.json --mode observe
```

`observe` 只在设备端保存三路图片和关节状态，既不连接 dawn，也不发动作。检查 `observation_0000/` 后，再试远端预测。

## 3. dawn：确认模型环境路径

SSH 登录 dawn 后，在已经配好的环境执行：

```bash
conda activate duoflow
cd ~/workspace/duoflow
python -c 'import sys; print(sys.executable)'
pwd
```

下面示例使用已报告的 `/home/dawn/miniforge3/envs/duoflow/bin/python` 和 `/home/dawn/workspace/duoflow`，如果输出不同，以实际路径为准。脚本使用绝对 Python 路径启动，不依赖非交互 SSH 自动激活 conda。

基础模型默认在 dawn 仓库的 `models/base/`，技能包在 `pretrained/simulation/`。已有权重不必重复下载。不需要在 dawn 手工启动相机程序或常驻推理服务器。

## 4. 设备电脑：通过 SSH 发起一次预测

退出到设备电脑终端，或另开设备电脑本机终端。把 `dawn@5090地址` 替换为你平时的 SSH 登录目标，也支持 `~/.ssh/config` 中已有的别名。非默认端口、跳板机等连接选项沿用 SSH 配置。

```bash
python scripts/run_airbot.py --config runs/airbot.json \
  --mode shadow --policy native \
  --remote-host dawn@5090地址 \
  --remote-repo /home/dawn/workspace/duoflow \
  --remote-python /home/dawn/miniforge3/envs/duoflow/bin/python
```

入口通过 `ssh -T` 启动 dawn 上的 `scripts/run_policy_worker.py`。SSH 沿用现有认证方式；需要密码时在终端输入，不在配置文件里保存密码。首次连接的主机身份仍由正常 SSH 检查，不关闭 host-key 校验。

等待远端模型加载，最多 600 秒；加载日志保存到设备端回合目录 `remote_stderr.log`。`shadow` 只做一次预测，保存结果后关闭该回合远端进程，不移动机械臂。

单技能替换为 `--policy skill --skill-id atomic_057`；具体任务卡片见设备端 `--list-skills`。这会加载匹配的技能 LoRA 和完成头。完整 DuoFlow 的用法见下一节。

实际动作仍需在设备端配置实测标定和运动限值，见 [AIRBOT 执行说明](AIRBOT_EXECUTION.md)。完成这些配置后，仅把 `--mode shadow` 改为 `--mode execute`，其余连接参数保持不变。

## 5. dawn：运行 Astra 桥

只在完整 DuoFlow 路线需要。桥在 dawn 的独立终端运行，沿用已有登录和网络代理：

```bash
conda activate duoflow
cd ~/workspace/duoflow
node astra_bridge/cli.mjs serve --request-root "$PWD/runs/astra_real" --max-requests 100
```

设备端 `runs/airbot.json` 填写：

```json
"astra_request_root": "/home/dawn/workspace/duoflow/runs/astra_real",
"allow_cloud_observations": true
```

然后在设备电脑运行上节命令，将 `--policy native` 换成 `--policy duoflow`。Astra 的凭据和密钥环留在 dawn；设备电脑不需要复制登录文件。该配置会将本次真机公开观测经 dawn 的桥发送给 Astra。

## 6. 传输、退出和记录

每条 SSH 会话对应一整个策略回合，模型状态在观测间保留。三路图像使用无损 PNG，14 维状态在设备端映射后发送；动作在设备端逆映射、检查并执行。设备端只有完成当前动作块后才发送下一次观测，影子模式不会虚构已执行历史。

无自动重连、无请求重发、无跨回合状态复用。响应顺序不匹配、检测到 SSH 退出或推理超时会终止回合。**现场动作停止仍由设备电脑负责**，使用既有流取消与 `PLANNING_POS` 请求；SSH 不能替代物理急停。动作循环检查 SSH 子进程是否退出，但网络故障在 SSH 尚未察觉时，已收到的本地动作块可能继续执行；Ctrl-C/STOP 与本机反馈异常仍按动作循环处理。

设备端回合目录保存 RGB、状态、动作、停止结果、`remote_stderr.log` 与往返耗时。`remote_ready` 事件记录 dawn 的回合目录：`runs/remote_policy/<回合>/`，其中保存收到的观测、控制器决策和远端退出结果。远端的 `predictions` 是返回预测的次数，不是物理动作数或成功率；真实执行记录以设备端为准。

设备端退出时关闭本次 SSH 子进程。远端在收到关闭请求或标准输入断开时结束本次推理进程；若网络处于半开状态，发现断连可能有延迟。Astra 桥由用户单独启动和关闭，不随每次回合重启。

本地测试覆盖跨进程 RGB 逐像素一致、连续策略状态、错误传播、超时、断连和无迟到动作。真实两机网络、5090 推理延迟与硬件闭环需要在现场测量。

```bash
python -m unittest real_robot.test_remote real_robot.test_airbot real_robot.test_cameras -v
```

SSH 数据通道依据：[OpenSSH 手册](https://man.openbsd.org/ssh)。
