# 仿真技能与权重迁移

这组文件来自 DuoFlow 全 50 任务使用的仿真模型。权重直接存放在 Git 仓库中，不使用 Git LFS；更新后即可在新主机读取。机器人配置仍是仿真的 Aloha-AgileX，AIRBOT Play 真机需要使用真实数据适配动作、归一化和完成判断。

## 下载到新主机

在已经克隆的仓库根目录、Python 3.11 的 `duoflow` 环境中运行：

```bash
git pull --ff-only origin main
python scripts/check_sim_weights.py
```

第二条命令只在 CPU 上检查发布文件、LoRA 形状、完成头及条件映射，不加载基础大模型、不调用 Astra、不连接机械臂。依赖来自现有的 `requirements-model.txt`，其中 PyTorch 按 [5090 安装说明](UBUNTU_5090_ASTRA.md) 安装。

## 包内文件

| 路径（相对于 `pretrained/simulation/`） | 内容 |
|---|---|
| `old20/skill_checkpoint.pt` | 原 20 任务共享技能 LoRA，5000 更新 |
| `old20/completion_checkpoint.pt` | 对应单帧 RGB/qpos 完成头，验证集选择 epoch 24 |
| `old20/skill_catalog.json` | 110 张技能卡；对应 112 个训练条件中的可导出条件及绑定 |
| `new30/skill_checkpoint.pt` | 新增 30 任务共享技能 LoRA，5000 更新 |
| `new30/completion_checkpoint.pt` | 对应单帧 RGB/qpos 完成头，验证集选择 epoch 21 |
| `new30/skill_catalog.json` | 219 张技能卡；对应 206 个训练条件中的可导出条件及绑定 |
| `native/twintimefm_adapter.safetensors` | native 整任务动作 LoRA，FT2T 10000 更新 |
| `native/selector.pt` | 完整 native 路线使用的本地技能选择器 |
| `native/runtime_editor.json` | native 运行时编辑策略的冻结评测参数 |
| `contracts/*.json` | 50 个仿真任务的执行阶段映射 |
| `norm_stats.json` | 两套技能及完成头匹配的仿真归一化统计 |
| `BUNDLE.json` | 任务到模型组映射、文件清单和版本校验值 |

总计约 55.2 MB、两套技能模型、329 张技能卡。技能卡数不等于独立神经网络数。两套模型的条件索引不同，不能将 new30 的完成头配给 old20，也不能简单拼接权重。模型保留原来的所有推理张量；导出时删除训练路径与非推理日志，并将技能卡中的 checkpoint 路径改为相对于仓库根目录的路径。

`place_bread_basket` 和 `put_bottles_dustbin` 保留原有的训练前缀及 whole-task 收尾契约。包中没有把这些任务的缺失阶段标成已经学会。

## 基础大模型

上面的权重是增量权重，仍依赖 GigaBrain 基础模型及两个 tokenizer。基础模型通过原发布源的固定 revision 下载，不重复上传到 GitHub：

```bash
source scripts/env.sh
python scripts/download_base.py
```

脚本只下载 `ASSET_SOURCES.json` 列出的模型和 tokenizer 文件，不下载 RoboTwin 场景、物体或数据集。`TWINTIMEFM_ASSETS` 可以指定已有基础模型的位置；默认目录为：

```text
models/base/
  backbone/
  paligemma2_tokenizer/
  fast_tokenizer/
```

这一步需要访问 Hugging Face，固定版本的下载文件合计约 14.2 GB，建议预留 20 GB 模型空间。已有完整的相同 revision 基础模型时无需重复下载。基础模型、PaliGemma tokenizer 与 FAST tokenizer 的许可分别遵循其发布源。

## 在代码中选择权重

以下代码生成已有模型接口所需的权重路径；它不是 AIRBOT 驱动，也不会发出动作：

```python
import json
from pathlib import Path

root = Path.cwd()  # 仓库根目录
bundle = root / "pretrained/simulation"
manifest = json.loads((bundle / "BUNDLE.json").read_text())
task = "handover_block"
bank = manifest["banks"][manifest["task_to_bank"][task]]
weight_config = {
    "model_path": str(bundle / "native"),
    "planner_checkpoint": str(bundle / manifest["native"]["selector"]),
    "runtime_editor_checkpoint": str(bundle / manifest["native"]["runtime_editor"]),
    "atomic_policy_root": str(root / "runtime/XPolicyLab/policy/TwinTimeFM"),
    "atomic_actor_checkpoint": str(bundle / bank["actor"]),
    "atomic_completion_checkpoint": str(bundle / bank["completion"]),
    "atomic_skill_catalog": str(bundle / bank["catalog"]),
    "native_stage_spec_path": str(bundle / "contracts" / f"{task}.json"),
}
```

如果只用仿真技能初始化真机技能训练，将对应 `skill_checkpoint.pt` 作为 `training/train_skills.py` 的 `--init-checkpoint`。先完成 [真机数据与动作适配](REAL_ROBOT.md)，再开始训练；完成头的最终参数与阈值使用真机验证集重新选择。

本包不包含训练演示、冻结经验记忆、API 凭据或 RoboTwin 资产。它提供已有模型初始化和加载所需的增量权重，不是旧仿真评测的全量复现实验归档。
