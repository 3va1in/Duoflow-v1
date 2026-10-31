import { spawn } from 'node:child_process';
import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto';

const attributes = ['application', 'duoflow-astra', 'purpose', 'credential-encryption-v1'];

function secretTool(args, input = '') {
  return new Promise((resolve, reject) => {
    const child = spawn('secret-tool', args, { stdio: ['pipe', 'pipe', 'pipe'] });
    const output = [];
    child.stdout.on('data', chunk => output.push(chunk));
    child.stderr.resume();
    child.on('error', () => reject(new Error('需要安装 libsecret-tools，并解锁当前用户的 Secret Service 密钥环。')));
    child.on('close', code => resolve({ code, value: Buffer.concat(output).toString().trim() }));
    child.stdin.on('error', () => {});
    child.stdin.end(input);
  });
}

export async function credentialEncryption({ create = false } = {}) {
  let result = await secretTool(['lookup', ...attributes]);
  if (result.code === 1 && create) {
    const key = randomBytes(32).toString('base64');
    const stored = await secretTool(['store', '--label=DuoFlow Astra credentials', ...attributes], key);
    if (stored.code !== 0) throw new Error('无法写入系统密钥环；请在已登录的 Ubuntu 桌面会话中解锁密钥环。');
    result = await secretTool(['lookup', ...attributes]);
  }
  const key = Buffer.from(result.value, 'base64');
  if (result.code !== 0 || key.length !== 32) throw new Error('DuoFlow 密钥环尚未初始化或未解锁；请先运行 login。');
  return {
    id: 'linux-secret-service-aes-gcm-v1',
    isAvailable: () => true,
    encrypt(text) {
      const iv = randomBytes(12);
      const cipher = createCipheriv('aes-256-gcm', key, iv);
      const encrypted = Buffer.concat([cipher.update(text, 'utf8'), cipher.final()]);
      return Buffer.concat([iv, cipher.getAuthTag(), encrypted]);
    },
    decrypt(bytes) {
      const data = Buffer.from(bytes);
      const decipher = createDecipheriv('aes-256-gcm', key, data.subarray(0, 12));
      decipher.setAuthTag(data.subarray(12, 28));
      return Buffer.concat([decipher.update(data.subarray(28)), decipher.final()]).toString('utf8');
    },
  };
}
