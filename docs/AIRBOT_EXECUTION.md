# 双 AIRBOT Play + 三路 RealSense 执行入口

`scripts/run_airbot.py` 连接两台 AIRBOT Play（各带两指夹爪）和三路 RealSense。机械臂接口对应 **airbot_py 5.1.6**，相机按序列号固定为全局、左腕、右腕。一次运行记录一个回合。

**相机与机械臂在一台电脑、策略在 dawn/5090 时，请使用 [双机部署说明](TWO_HOST_AIRBOT.md)。** 本文的设备枚举与采集命令均在相机所在电脑执行；普通 SSH 登录不会把设备自动带到另一台主机。执行入口支持 `--remote-host`，可保留本机采集与控制，将推理放到 dawn。

| 策略 | 实际调用 |
|---|---|
| `--policy skill --skill-id atomic_057` | 指定技能 LoRA、匹配完成头与新观测完成确认 |
| `--policy native` | 原本地选择器、运行时控制器与 native 动作模型 |
| `--policy duoflow` | Astra 编排、技能执行、完整 native 回退及返回规划 |

DuoFlow 的切换仍由 `method/skill_policy.py` 和 `method/astra_reentry.py` 实现。API 错误终止本回合，不自动重试。当前入口加载已下载的仿真权重，不需要 RoboTwin 资产。

## 1. 环境与连接

沿用现有 Python 3.11 模型环境和 `requirements-model.txt`，安装 RealSense Python 接口：

```bash
conda activate duoflow
python -m pip install -r requirements-real.txt
python -m pip show airbot_py
python scripts/run_airbot.py --list-cameras
```

`airbot_py` 应显示 `5.1.6`。本入口使用该版本 wheel 附带的官方 gRPC 消息和枚举，不需要安装 `arm_sdk` 或升级服务端、固件。厂商 `servo_*` 方法只入后台队列，入口因此直接使用同一协议的可取消流，并为查询、模式切换设置超时。关节与夹爪目标放在同一条该臂命令中。

`--list-cameras` 仅列出 RealSense 名称和序列号，不加载模型或连接机械臂。如果列表为空，先用 `realsense-viewer` 检查三台设备、USB 连接和用户设备访问权限；关闭 viewer 后再运行采集，避免占用相机。

沿用你已能读取关节的服务。尚未启动时，在两个终端分别启动：

```bash
airbot_server -i can_left -p 50051
airbot_server -i can_right -p 50052
```

CAN 名称和端口以实际配置为准。已经启动时不要重复启动；脚本不会归零、加载或卸载遥操作应用。机器人 gRPC 通道关闭 HTTP 代理，Astra 桥可继续使用代理连接。

## 2. 三路 RealSense 与只读观察

在仓库根目录建立个人配置：

```bash
mkdir -p runs
cp examples/airbot_dual.json runs/airbot.json
python scripts/run_airbot.py --list-cameras
```

把实际序列号填入 `runs/airbot.json`，序列号必须保留为字符串：

```json
"cameras": {
  "serials": {
    "cam_high": "全局相机序列号",
    "cam_left_wrist": "左腕相机序列号",
    "cam_right_wrist": "右腕相机序列号"
  },
  "width": 640,
  "height": 480,
  "fps": 30,
  "max_age_s": 0.3,
  "max_skew_s": 0.1
}
```

随后运行：

```bash
python scripts/run_airbot.py --config runs/airbot.json --mode observe
```

`observe` 是默认模式，只保存一组 RGB 和双臂状态，不加载模型、不调用 Astra、不切换机器人模式。打开输出目录的 `observation_0000/`，确认三张画面及左右臂对应正确。

每台 RealSense 独立持续采集 `rgb8`，保留设备时间戳、时间戳域、帧号和主机接收时间。当前模型只使用 RGB，因此不启用深度流；没有 OpenCV 索引或 BGR 转换。序列号绑定不会因 USB 重插顺序变化而交换左右腕。配置限制的是画面年龄和主机接收时差，不代表三台设备已硬件同步。

## 3. 关节映射与运动范围

硬件向量顺序为 `[左关节1..6, 左夹爪, 右关节1..6, 右夹爪]`。5.1.6 的关节单位是**弧度**，两指夹爪是**开口宽度，米**。通过实测确认开合、零点、方向与范围，再填写映射：

```text
model[j] = robot[model_from_robot[j]] * scale[j] + offset[j]
robot_target[model_from_robot[j]] = (model_target[j] - offset[j]) / scale[j]
```

输入状态和输出动作使用同一组互逆变换。示例是单位映射，`verified=false`；不能据此认定仿真 Aloha 的关节与 AIRBOT 已对齐。该变换也不解决连杆长度、双臂基座或相机视角差异。零样本实验保留原权重与归一化，使用真机数据训练后另记为适配实验。

执行前填写 `limits` 中四个 14 维数组，全部按**硬件顺序和单位**：

| 字段 | 作用 |
|---|---|
| `position_min` / `position_max` | 允许的关节角与夹爪宽度范围 |
| `max_velocity` | 每秒允许的目标位置变化，决定相邻目标最大差值 |
| `max_tracking_error` | 目标与反馈位置允许的最大差值 |

这些值按实际设备和工作空间设置，示例不填通用数值。`max_velocity` 是入口对目标序列的检查，不是电机实际速度保证；驱动器的速度、电流及机械保护沿用现场配置。完成实测映射后设置 `calibration.verified=true`。

入口整体检查预测动作块，拒绝非有限值、位置越界和过大跳变，不自动裁剪动作。双臂碰撞与环境碰撞检测不在此入口中，首次执行应使用清空的工作空间和现场物理急停。

## 4. 单次预测与执行

先列出当前任务的技能：

```bash
python scripts/run_airbot.py --config runs/airbot.json --list-skills
python scripts/run_airbot.py --config runs/airbot.json \
  --mode shadow --policy skill --skill-id atomic_057
python scripts/run_airbot.py --config runs/airbot.json --mode shadow --policy native
```

`task` 自动选择 old20/new30 技能库和完成头。`atomic_057` 对应示例任务 `rotate_qrcode` 的左臂拿起技能，其他任务使用列表中的 ID。`shadow` 只对一组实测观测预测一次并保存，不切换机器人模式，也不会在未执行动作时推进技能历史。

填写标定和范围后，显式选择 `execute` 才会下发动作：

```bash
python scripts/run_airbot.py --config runs/airbot.json \
  --mode execute --policy skill --skill-id atomic_057
python scripts/run_airbot.py --config runs/airbot.json --mode execute --policy native
```

两臂均需处于 `IDLE / PLANNING_POS`。先退出遥操作和其他控制程序。5.1.6 没有服务端独占租约；入口的本机锁只防止同一主机上的另一份 DuoFlow 入口重复控制，不能排斥其他 SDK 客户端。

## 5. Astra 与完整 DuoFlow

按 [Astra 接入说明](UBUNTU_5090_ASTRA.md) 登录，在独立终端启动桥：

```bash
node astra_bridge/cli.mjs serve --request-root "$PWD/runs/astra_real" --max-requests 100
```

配置 `astra_request_root` 为同一**绝对路径**，设置 `allow_cloud_observations=true` 后运行：

```bash
python scripts/run_airbot.py --config runs/airbot.json --mode shadow --policy duoflow
python scripts/run_airbot.py --config runs/airbot.json --mode execute --policy duoflow
```

该设置允许发送本次真机 RGB、状态、任务及已执行历史给 Astra，权重与凭据不进入请求。三种策略共用设备、映射、限值和控制频率。`max_steps` 限制本回合动作数；`seed` 仅控制模型采样，不代表真机场景可以按 seed 复现。本入口使用空的冻结经验记忆，不写入测试反馈。

## 6. 停止与回合记录

每个动作块结束后取消命令流并切换 `PLANNING_POS`，推理期间不继续发送动作。按 Ctrl-C、发送 SIGTERM，或在回合目录创建 `STOP` 文件，会结束回合并对本入口已接管的两臂执行同样的停止请求。停止失败单独记录。

**这不是硬件急停。** 5.1.6 没有新版的急停 API，断网或进程被强制杀死时软件不能保证停止请求送达。硬件停机使用现场物理急停；入口不自动解除急停或切换重力补偿。模式切换后的实际保持行为需在设备上确认。

推理等待期间检查相机和关节反馈；关节新鲜度按主机接收时间判断。超时后结束回合，不采用迟到预测。两臂顺序提交命令并记录时间差，不宣称两臂硬件同步。当前示例以 10 Hz 下发动作，分块推理及块间保持属于真机执行设置。

每回合输出：

- `CONFIG.json`：策略、设备和标定配置。
- `events.jsonl`：观测、预测动作、逐臂提交、实测反馈、完成头输出与停止原因。
- `observation_*/`：每次决策的三路 RGB。
- `controller/decisions.jsonl`：完整控制器的选择、回退和返回规划记录。
- `RESULT.json`：停止状态、`submitted_action_pairs`、错误与清理结果。

`submitted_action_pairs` 只表示两臂命令被 gRPC 发送迭代器消费，不是逐动作服务端确认或物理完成；实际反馈另行保存。`task_success` 默认 `null`，任务成功以录像/实物结果标注，不把完成头判断、命令提交或预算耗尽当成物理成功。保留中断试次和已经提交的单臂动作日志。

## 7. 本地验证

无需硬件的控制与相机测试：

```bash
python -m unittest real_robot.test_airbot real_robot.test_cameras -v
```

安装了官方 `airbot_py 5.1.6` 后，可验证本机模拟 gRPC 服务与真实 protobuf 的往返、拒绝响应、超时及流退出；该测试只连接随机 localhost 端口：

```bash
python -m unittest real_robot.test_sdk516 -v
```

开发环境没有连接真实 AIRBOT、RealSense，也没有运行 RTX 5090 推理。真实反馈、摄像头采集、动作保持和任务 SR 在目标主机上验证。

接口依据：[厂商版本下载页](https://docs.discover-robotics.com/document/en/airbot-play/changelog.html)、[5.1.6 官方安装包](https://docs.discover-robotics.com/document/assets/airbot-play/5.1.6/5.1.6.zip)内的 `arm.py`、`servo_controller.py` 与 protobuf、[RealSense Python 示例](https://github.com/realsenseai/librealsense/tree/master/wrappers/python/examples)。
