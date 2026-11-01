# DuoFlow: Astra + native fallback

用于把 DuoFlow 方法源码迁移到新主机，开展双臂真机适配。仓库包含 Astra 技能编排、原子技能执行、完成预测、完整 native 回退控制器，以及技能训练代码。不包含 RoboTwin 场景、物体资源、专家采集程序或旧实验调度器。

当前交付包含**方法源码和独立 Astra 登录桥**。机器人驱动、真机数据转换及真机权重仍需根据实验设备配置；克隆仓库不会直接启动机器人。

已加入**仿真技能权重包**：原 20 任务与新增 30 任务两套技能 LoRA、匹配完成头、329 张技能卡，以及 native 动作权重和本地选择器。更新及基础模型下载命令见 [仿真权重迁移](docs/SIMULATION_WEIGHTS.md)。

RTX 5090 / Ubuntu 20.04 的环境安装与本账号 ChatGPT 计划用量接入，见 [逐步安装说明](docs/UBUNTU_5090_ASTRA.md)。该配置使用 Python 3.11 和 PyTorch 2.7.1 cu128。

## 代码结构

| 路径 | 内容 |
|---|---|
| `method/astra_reentry.py` | Astra 选择技能、短段委托和重新进入规划 |
| `method/reentry_actors.py` | 同一骨干上的技能/native LoRA 切换 |
| `method/atomic_skill_executor.py` | 技能动作块执行与完成预测确认 |
| `method/native_fallback.py` | 本地选择器、运行时控制和原始动作策略组成的完整 native 回退 |
| `method/skill_policy.py` | 原集成接口，仍使用 XPolicyLab 观测格式 |
| `method/astra_planner.py` | Astra 请求、结构化响应及文件传输客户端 |
| `training/` | 技能 LoRA、单帧 RGB/状态完成头训练及 catalog 导出 |
| `runtime/XPolicyLab/policy/TwinTimeFM/` | 原生动作模型加载、LoRA、数据工具和依赖源码 |
| `runtime/env_cfg/` | 历史动作维度注册信息，不是真机驱动配置 |

`TwinTimeFM` 是代码依赖中的历史名称，保留以维持导入和权重格式兼容。

## 新主机下载与控制逻辑检查

在新主机上使用有该私有仓库读取权限的 GitHub 账号克隆：

```bash
git clone https://github.com/3va1in/Duoflow-v1.git
cd Duoflow-v1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-control.txt
source scripts/env.sh
python scripts/check_controller.py
```

这项检查使用已有测试中的模拟 actor 和 planner，不调用 Astra、不加载模型权重、不连接机器人；它验证技能/native 切换、累计动作预算和 API 失败边界。

## 模型环境

采用独立 Python 3.11 环境。原策略环境以 PyTorch 2.7.1 / torchvision 0.22.1 为参考；按新主机驱动选择 PyTorch CUDA 安装包，再安装 `requirements-model.txt`。这里提供依赖参考，尚未在新主机验证 CUDA 模型加载。

`source scripts/env.sh` 加入方法、训练和模型依赖源码的导入路径。`TWINTIMEFM_ASSETS` 默认指向仓库下的 `models/base`，也可在执行前显式设置。基础模型目录结构为：

```text
models/base/
  backbone/               # 模型、inference_config.json、norm_stats.json
  paligemma2_tokenizer/
  fast_tokenizer/
```

原始资产来源见 `runtime/XPolicyLab/policy/TwinTimeFM/ASSET_SOURCES.json`。其中的 RoboTwin Clean 权重是仿真初始化来源，不是真机训练结果。

## 真机迁移

按 [真机迁移说明](docs/REAL_ROBOT.md) 接入观测、动作和数据。最初先完成本地 native 单任务执行，再加入技能与完成头，最后接入 Astra 编排及 native 回退。

原训练入口可通过以下命令查看参数：

```bash
python training/train_skills.py --help
python training/train_completion.py --help
python training/export_catalog.py --help
```

运行训练前须完成动作监督与真机归一化的适配；不要直接把真实测量关节状态当作旧仿真 drive-target。

## GitHub 与模型文件

GitHub 保存代码、文档、配置和 `pretrained/simulation/` 下明确列出的仿真推理权重。`git pull` 会取回这组约 55.2 MB 的文件。第三方基础大模型由 `scripts/download_base.py` 从固定版本下载。训练记忆、真机采集数据、其他模型和凭据仍由 `.gitignore` 排除。

Astra 客户端与 `astra_bridge/` 独立服务端随仓库提供，使用官方 Sign in with ChatGPT 取得本账号授权；新主机可以独立运行，不依赖旧 Mac。现有 guided 路线保持 `siwc_files` 协议，登录、安装和启动命令见 [Ubuntu / Astra 接入](docs/UBUNTU_5090_ASTRA.md)。桥不包含或读取旧实验的账号凭据、目录和预算。

## 来源与许可

方法与训练源码从现有实验代码复制，未改动执行算法。原生依赖来自已有本地版本，文件级来源列于 `SOURCE_MANIFEST.json`。第三方源码的许可保留在各自目录，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。未包含论文草稿、实验结果、主机连接清单或登录会话。
