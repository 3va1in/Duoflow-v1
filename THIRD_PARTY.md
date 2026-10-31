# 第三方源码

- `runtime/XPolicyLab/` 中的接口工具来自 XPolicyLab 本地参考版本；许可位于 `runtime/XPolicyLab/LICENSE`。
- `runtime/XPolicyLab/policy/TwinTimeFM/vendor/backbone_lib/` 来自 Giga Models；保留其 `LICENSE`。
- `runtime/XPolicyLab/policy/TwinTimeFM/vendor/native_release/` 来自 GigaBrain 原生推理实现；保留其 `LICENSE`。
- 原始仓库和已记录版本见 `runtime/XPolicyLab/policy/TwinTimeFM/ASSET_SOURCES.json`。
- Astra 安装脚本下载固定版本的官方 Sign in with ChatGPT DevKit，补丁按其非商业许可提供；来源和修改说明见 `astra_bridge/NOTICE.md`，许可见 `astra_bridge/DEVKIT_LICENSE`。

这些代码沿用现有兼容修改，不代表上游最新版本。模型权重、数据与 tokenizer 需要按各自来源的许可单独获取。
