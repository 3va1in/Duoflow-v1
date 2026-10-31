#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="$root/.cache/sign-in-with-chatgpt-devkit"
revision=0a36fefeb913055c8c7a1a29b63d82b96b2e841a
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22||(a===22&&b<12)) throw Error("需要 Node.js >=22.12")'
if [[ ! -d "$sdk" ]]; then
  mkdir -p "$root/.cache"
  git clone --no-checkout https://github.com/openai/sign-in-with-chatgpt-devkit.git "$sdk"
  git -C "$sdk" checkout --detach "$revision"
fi
[[ "$(git -C "$sdk" rev-parse HEAD)" == "$revision" ]] || { echo 'SDK 版本不符；保留已有目录，请检查。' >&2; exit 1; }
patch="$root/astra_bridge/patches/siwc-multimodal.patch"
if ! git -C "$sdk" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$sdk" apply --check "$patch"
  git -C "$sdk" apply "$patch"
fi
cd "$sdk"
npm ci --ignore-scripts --workspace @siwc/local --include-workspace-root
npm run build --workspace @siwc/local
echo 'Astra SDK 已安装；下一步：node astra_bridge/cli.mjs login（在 DuoFlow 仓库目录执行）。'
