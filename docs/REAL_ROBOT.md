# 真机适配边界

## 观测和动作

当前动作接口为左右各 6 个关节加 1 个夹爪，共 14 维；模型一次预测 50 步。原子技能通常只执行预测的前 10 步，native 路径通常执行前 30 步。真机控制频率、动作窗口与执行反馈需要联合确定，不能把这些步数直接理解成固定秒数。

相机输入是 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路已解码 HWC uint8 RGB。真实相机名称可以映射，但视角、同步和训练预处理需要一致。

以下位置含设备相关假设：

- `method/reentry_actors.py`：固定 `[50,14]` 动作和 checkpoint 归一化一致性。
- `method/atomic_skill_executor.py`：14 维 drive-target、完成确认时用当前状态构造保持动作。
- `runtime/XPolicyLab/policy/TwinTimeFM/runtime.py`：固定 embodiment、robot_type、相机与动作变换。
- `runtime/XPolicyLab/policy/TwinTimeFM/model.py`：双臂 6+1 维、XPolicyLab 观测封装。
- `method/native_fallback.py`：通过原模型适配器恢复完整 native 控制状态。

驱动适配层应明确关节顺序、单位、夹爪缩放、绝对/增量命令、时间戳和已执行动作数。机器人控制器负责限位、限速、急停及断联处理。native 回退是另一条策略执行路径，不是硬件安全控制器。

## 训练数据

原训练目录由 `task/seed_x/observations.jsonl`、相对路径图像和 `segments.json` 组成。分段包含训练/验证划分、任务、技能文本、起止帧和完成标签。

已有 [LeRobot v2.1/v3 转换脚本](LEROBOT_CONVERSION.md)，读取测量状态、实际下发的绝对关节/夹爪目标和人工技能标注。新数据使用同一行的 `action` 监督，技能末端以后的动作屏蔽。旧仿真数据继续使用后续 `state` 作为动作目标，不改变历史训练规则。源动作顺序、单位和时间对齐需与部署一致。

采集内容：同步 RGB、测量状态、下发动作、时间戳、任务文本、技能分段、完成标签及整任务成功标签。完成标签允许未知值；不可把未知标为失败或把专家程序返回视为完成。

同一批完整任务演示可分别用于 native 动作学习和技能分段学习。当前 `train_skills.py` 与 `train_completion.py` 是技能训练入口；参考 native FT 代码位于 `runtime/XPolicyLab/policy/TwinTimeFM/train_base.py`，其任务和数据划分也是历史设置，需改为真机数据配置。

真机归一化需来自训练集，并与 native、技能 actor、完成头及部署变换保持一致。仿真完成头阈值和技能配方也应在真机验证集重新确定。

## Astra 和控制节奏

`ReentryController` 接收 actor、planner 和 native 控制器对象。planner 的输入是当前观测、任务、已执行历史、技能卡和训练记忆，返回结构化技能选择；模型不会直接生成机器人伺服命令。

当前 `AstraGuidedBackend` 只接文件桥。仓库的 `astra_bridge/` 可在 Ubuntu 主机独立登录本账号并提供该文件协议，安装步骤见 [Ubuntu / Astra 接入](UBUNTU_5090_ASTRA.md)。不依赖旧全50任务目录。真机仍需任务配置；模型权限和实际连接由账号授权及一次完成请求确认。

原实现的 API 失败会抛出异常并中止该次调用，不自动转 native。真机外层应明确异常时的机器人停止行为。云端响应等待不能阻塞机器人自身的状态反馈和安全控制。

## 部署顺序

1. 克隆源码、建立模型环境、放置独立模型文件。
2. 接入只读相机和状态，核对观测时间与动作约定。
3. 采集真机演示，适配并训练 native 路径。
4. 用分段数据训练技能 LoRA 与完成头，导出真机技能目录。
5. 接入 Astra，检查技能调用、native 回退和重新进入的完整闭环。

当前目标设备为两台 AIRBOT Play 六轴机械臂，各带两指夹爪。仓库已提供数据转换入口；真机数据、驱动接入与实机训练仍需完成。
