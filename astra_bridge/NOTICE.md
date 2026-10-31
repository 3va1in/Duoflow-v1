# Sign in with ChatGPT 依赖

安装脚本下载官方仓库 `https://github.com/openai/sign-in-with-chatgpt-devkit.git`，固定提交 `0a36fefeb913055c8c7a1a29b63d82b96b2e841a`，仅构建 `@siwc/local`。

`patches/siwc-multimodal.patch` 来自此前 DuoFlow 的本地适配：为 `packages/local/src/types.ts` 和 `responses.ts` 加入图像输入、严格 JSON schema、low reasoning、完整 Responses 回执及流中断检查。该补丁按随附 `DEVKIT_LICENSE`（Sign-in with ChatGPT DevKit Noncommercial License v1.0）提供；上游许可和第三方声明保留在下载目录。适配文件顶部注明修改。

SDK 原有的 OAuth、ID token 校验、刷新及账号隔离保持原实现。桥只调用官方 `https://api.openai.com/v1/responses`，不读取 Codex 的 auth.json，不复制旧主机的令牌或 host ID。

2026-10-31 同一补丁还修改 `packages/local/package.json` 和根 `package-lock.json`：将可选 HTTP 依赖明确固定为 `undici@7.30.1`，修复安装审计中原 7.29.1 的已知漏洞；未升级 OpenAI SDK 或方法依赖。

SDK 下载和构建产物放在 Git 忽略的 `.cache/`，凭据保存在用户配置目录并用 Linux Secret Service 中的密钥加密。本仓库不包含登录会话。
