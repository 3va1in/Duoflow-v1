# 本地检查

- 复用了原控制逻辑测试，共 19 项通过：技能动作预算、native 窗口后返回 Astra、原始 LoRA 恢复、技能完成/超时边界、API 中断与任务失败分类。
- 检查未访问上游 API、未加载模型权重、未连接机器人。
- Python 源文件已做语法解析；环境脚本通过 Bash 语法检查。
- 待提交文件已检查常见密钥格式、原主机个人目录及误入的大文件；未发现匹配项。运行数据和二进制权重由 `.gitignore` 排除。
- 本地测试使用 Python 3.12 CPU 环境；目标 Python 3.11/CUDA 模型加载、真机训练和机器人动作接口尚未验证。

首次打包测试发现缺少 `paired_scenes.py` 这一既有分类测试依赖，已补齐原文件后重新运行并通过；没有修改测试断言。

## Ubuntu Astra 桥新增检查（2026-10-31）

- `node --test astra_bridge/test_bridge.mjs`：4 项通过，覆盖完整响应/重复 ID、不完整流/零重试、未知领取/累计预算、未授权计划用量/固定模型。
- Python 原始 `AstraNativeFallbackBackend` → Node 文件桥 → Python 结构化解析往返通过；三路图像是合成数据，模型响应是明确标注的 mock，没有访问网络推理。
- 安装脚本实际下载固定版本 DevKit、应用补丁、安装 local workspace 并构建成功。初次审计发现 `undici@7.29.1` 漏洞，已将其固定到 7.30.1，修复后安装审计为 0 项。
- 本地 Node 版本 26.5.0，Python 3.12 CPU。未在目标 Ubuntu 主机验证 Secret Service、Python 3.11/CUDA 权重加载或真实 Astra 请求；用户需按部署文档在新主机授权并运行一次连接检查。
- 未读取、迁移或提交旧账号令牌；未修改旧实验桥、策略算法或权重。
