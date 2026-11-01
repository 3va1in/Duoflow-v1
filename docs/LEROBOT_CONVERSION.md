# LeRobot 数据转为 DuoFlow 技能训练数据

转换器读取本地 LeRobot **v2.1 和 v3.0** 数据集，输出 `train_skills.py` 与 `train_completion.py` 共用的训练目录。支持 Parquet 中的图像字节、数据集内的图像路径，以及 MP4 视频。v3 多个回合共用视频时，按回合元数据中的时间偏移取帧。

不需要安装 LeRobot，也不需要连接机器人或 Astra。源数据只读；转换结果包含解码后的 PNG，通常比源视频占用更多磁盘空间。

## 1. 更新并安装转换依赖

在已建立的 Python 3.11 环境、仓库根目录执行：

```bash
git pull --ff-only origin main
python -m pip install -r requirements-convert.txt
```

仅转换数据不需要 PyTorch。后续训练使用仓库已有模型环境和基础权重。

## 2. 从数据集生成标注模板

```bash
python training/convert_lerobot.py \
  --dataset /path/to/lerobot_dataset \
  --init-annotations /path/to/skill_annotations.json
```

模板列出实际特征名和每个 `episode_index`。填写下述映射及分段；未准备使用的回合从标注文件的 `episodes` 中移除即可，源数据不会被删除。完整示例见 [lerobot_annotations.example.json](../examples/lerobot_annotations.example.json)，示例的任务与帧号需要换成自己的标注。

### 相机、状态与动作

`camera_map` 将三个真实相机字段映射到 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。三个字段分别对应全局、左腕、右腕视角，不能用同一路相机重复填充。

`state.key` 指向实测关节/夹爪状态，`action.key` 指向实际下发的目标命令。两者独立读取，转换器不会用下一帧状态代替命令。

输出 14 维顺序固定为：

```text
左臂 joint1..joint6，左夹爪，右臂 joint1..joint6，右夹爪
```

各自通过 `indices` 指定源数组下标。可选的 `scale`、`offset` 是各 14 个数，作用于重排后的维度：`output = source[indices] * scale + offset`。例如源角度为度、训练接口使用弧度时，12 个关节维度的 scale 填 `0.017453292519943295`；夹爪按实测标定单独填写。默认不改变数值。

必须确认并填写：

```json
"mode": "absolute_joint",
"alignment": "same_row"
```

这里的动作是**以同一行观测为输入、随后下发的双臂绝对关节/夹爪目标**。若记录的是速度、增量、末端位姿，或前一时刻已经执行的动作，需要先按采集控制器的定义转换并对齐。只更改上述字符串不会转换动作含义。AIRBOT 主臂的读数也不能未经从臂映射直接当成最终下发命令。

脚本保留原帧率与时间戳，不插值、不重采样。视频取最近的 PTS 帧，误差超过半个采样周期会报错。转换报告记录每回合的采样间隔偏差；相机与机器人本身的时钟同步仍由采集程序保证。

### 回合与技能段

每个回合明确填写 `task`、`split` 和 `segments`。`task` 是目录名，使用字母、数字、下划线或连字符；`split` 为 `train` 或 `val`。按完整回合划分，不能将同一条轨迹的相邻帧随机分入训练与验证。

每个技能段包含：

| 字段 | 含义 |
|---|---|
| `skill_id` | 技能类型，如 `pick_and_lift` |
| `skill_text` | 完整技能指令，明确机械臂、对象和目标 |
| `start_frame` / `end_frame` | 本回合从 0 开始的帧号，均包含对应观测 |
| `action_train_valid` | 这段命令是否适合模仿；由采集者明确填写 |
| `completion_spans` | 完成标签区间，见下文 |
| `bindings` | 机械臂、对象、目标等绑定，导出技能卡使用 |
| `precondition` | 技能可调用的前提，导出技能卡使用 |
| `max_policy_steps` | 技能执行步数上限，导出技能卡使用 |

动作监督使用 **`start_frame <= frame < end_frame`**。`end_frame` 是执行后的末端观测，其动作可能属于下一个技能，因此不纳入前一段。相邻技能可共享一张边界观测，例如第一段 `[0, 99]`、第二段 `[99, 179]`；边界帧的动作只进入第二段。动作块不足模型所需的 50 步时填充并屏蔽填充部分，不跨技能取动作。

`skill_id + skill_text` 构成完成头的条件。训练集与验证集对同一技能绑定使用相同文本；验证集不能出现训练集没有的条件。这不是自动学习任意新技能文本的接口。

### 完成头标签

每个标签回答“**当前技能的完成条件，在这一帧是否已经满足**”：

- `0`：明确未满足。正常成功演示的执行前、中段也能提供负例。
- `1`：明确已满足，例如物体已抓稳并抬离桌面。
- 未填写或 `null`：未知，不参与完成头损失。

`completion_spans` 的起止帧均包含。也可以改用逐帧 `termination_labels`，例如 `{"0": 0, "97": 1, "98": 1, "99": 1}`，两种写法选择一种。

脚本不会把技能末帧、夹爪闭合或 episode 结束自动标成成功。接近成功但尚未完成的画面应标 `0`；无法确认的过渡帧留空。整任务是否成功与某个技能是否完成分别判断。失败尝试可设置 `action_train_valid: false`，同时保留明确的完成标签供完成头训练；成功段也可因遥操作质量差而不用于动作模仿。

当前训练入口需要训练集和验证集，并需要显式正、负完成标签。建议每个技能绑定在两个划分中都覆盖正负例。转换成功只表示格式有效，不代表演示数量或标签覆盖已经充分。

## 3. 转换

```bash
python training/convert_lerobot.py \
  --dataset /path/to/lerobot_dataset \
  --annotations /path/to/skill_annotations.json \
  --output /path/to/duoflow_real_data
```

输出目录必须不存在，避免覆盖已有数据。结构如下：

```text
duoflow_real_data/
  CONVERSION_REPORT.json
  pick_place_block/
    seed_000000/
      observations.jsonl
      segments.json
      images/cam_high/000000.png
      images/cam_left_wrist/000000.png
      images/cam_right_wrist/000000.png
    seed_000001/
      ...
```

`seed_000000` 只沿用旧读取器的目录命名，数字是源 `episode_index`，不代表真实实验的随机种子。每帧同时保存测量 `state`、下发 `action`、时间戳与图像路径。报告列出 train/val 回合数、动作监督帧数、完成标签数量和未选中的回合。

新分段带 `action_source: recorded_action`，读取器据此使用记录的动作。旧仿真分段没有这一字段，继续使用原来的后续状态监督规则。

## 4. 接入训练

转换目录可直接作为两个训练入口的 `--data-root`，不必再次汇总或改文件格式：

```bash
source scripts/env.sh
python training/train_skills.py \
  --policy-root runtime/XPolicyLab/policy/TwinTimeFM \
  --data-root /path/to/duoflow_real_data \
  --init-checkpoint pretrained/simulation/old20/skill_checkpoint.pt \
  --output runs/airbot_skills

python training/train_completion.py \
  --policy-root runtime/XPolicyLab/policy/TwinTimeFM \
  --data-root /path/to/duoflow_real_data \
  --checkpoint runs/airbot_skills/skill_checkpoint.pt \
  --output runs/airbot_completion
```

以上命令需在基础模型、GPU 环境和 AIRBOT 动作/归一化约定配置好后执行。仿真 checkpoint 仅用作动作初始化，新技能条件对应的完成头会重新训练。独立完成头训练读取本轮技能 checkpoint，因此按上述顺序执行。

**转换器不计算归一化统计。** 当前训练器要求传入统计与 `models/base/backbone/norm_stats.json` 一致；省略 `--norm-stats` 会沿用该文件，通常仍是下载的仿真统计。AIRBOT 的训练统计应只由训练集计算，并与骨干预处理、native、技能及部署一起配置；不能仅给 `--norm-stats` 传一份不同文件。这里完成的是数据和动作监督适配，未改动机器人驱动或 native 训练入口。

## 已验证范围

使用本地合成数据验证 v2.1/v3.0 的图像字节、图像路径和 MP4 三种存储，包含 v3 共用视频的第二回合偏移。转换结果已由真实 `SkillDataset`、`CompletionDataset` 读取，并用于完成头 CPU 前向/反向计算。测试还覆盖动作与状态区分、技能末端屏蔽、未知标签和旧数据兼容。

```bash
python -m unittest discover -s training -p test_lerobot_conversion.py -v
```

尚未收到真实采集数据，本次没有运行 AIRBOT 模型训练或机器人执行。

格式依据：[LeRobot v3 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、[v2.1 数据读取实现](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)、[v3 数据读取实现](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/lerobot_dataset.py)。
