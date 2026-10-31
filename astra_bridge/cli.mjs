import { homedir } from 'node:os';
import { mkdir, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { credentialEncryption } from './credentials.mjs';
import { serve } from './bridge.mjs';

async function main() {
  process.umask(0o077);
  const { values, positionals } = parseArgs({ allowPositionals: true, options: {
    'request-root': { type: 'string' }, 'max-requests': { type: 'string' },
    port: { type: 'string', default: '14555' }, help: { type: 'boolean' },
  } });
  const command = positionals[0];
  if (values.help || !command) {
    console.log('node astra_bridge/cli.mjs login|status|models|disconnect\nnode astra_bridge/cli.mjs serve --request-root /absolute/path --max-requests 100');
    return;
  }
  if (positionals.length !== 1 || !['login', 'status', 'models', 'disconnect', 'serve'].includes(command)) throw new Error('未知命令');
  const port = Number(values.port), maximum = Number(values['max-requests']);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('port 应在 1024–65535');
  if (command === 'serve' && (!isAbsolute(values['request-root'] || '') || !Number.isSafeInteger(maximum) || maximum < 1)) {
    throw new Error('serve 需要绝对 request-root 和正整数 max-requests');
  }
  const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const sdk = pathToFileURL(join(repo, '.cache/sign-in-with-chatgpt-devkit/packages/local/dist/index.js'));
  const { createChatGPT } = await import(sdk.href);
  const storageDir = join(homedir(), '.config/duoflow-astra');
  const savedAuth = await stat(join(storageDir, 'chatgpt-auth.json')).then(() => true, error => {
    if (error.code === 'ENOENT') return false;
    throw error;
  });
  // 已有加密记录时，密钥环暂不可用不能被当作首次安装而生成另一把密钥。
  const encryption = await credentialEncryption({ create: command === 'login' && !savedAuth });
  const client = createChatGPT({ appName: 'DuoFlow Astra', appId: 'duoflow-astra', redirectPort: port,
    storageDir, sendHostId: true,
    credentialEncryption: encryption,
    openBrowser(url) { console.log(`Continue with ChatGPT：请在浏览器打开此链接并授权本账号的计划用量。\n${url}`); },
  });
  if (command === 'login') console.log(JSON.stringify(await client.signIn({ reconsent: true }), null, 2));
  if (command === 'status') console.log(JSON.stringify(await client.getSession(), null, 2));
  if (command === 'disconnect') await client.disconnect();
  if (command === 'models' || command === 'serve') {
    const models = await client.listModels();
    if (command === 'models') console.log(JSON.stringify(models, null, 2));
    else {
      if (!models.some(model => model.slug === 'gpt-6-astra')) throw new Error('本账号模型列表没有 gpt-6-astra；不会自动替换模型。');
      const root = values['request-root'];
      await mkdir(root, { recursive: true, mode: 0o700 });
      const lockfile = createRequire(sdk.href)('proper-lockfile');
      const release = await lockfile.lock(root, { lockfilePath: join(root, '.bridge.lock'),
        stale: 120_000, update: 10_000, retries: 0 });
      try { await serve(root, client, maximum); } finally { await release(); }
    }
  }
}

main().catch(error => {
  console.error(error.code || error.message);
  process.exitCode = 1;
});
