import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile, copyFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { processRequest, requestOptions, serve } from './bridge.mjs';

const id = 'a'.repeat(32);
const decision = { skill_id: 'whole_task', progress_status: 'unknown', supporting_evidence: [], reason: 'test' };
const body = { model: 'gpt-6-astra', reasoning: { effort: 'low' }, store: false, stream: true,
  instructions: 'Choose a skill.', input: [{ role: 'user', content: 'test' }],
  text: { format: { type: 'json_schema', name: 'guided_skill_decision', strict: true, schema: {} } } };
const connected = async () => ({ status: 'connected', sharing: true });

test('CLI 帮助可在未登录、未安装 SDK 时加载全部本地模块', () => {
  const result = spawnSync(process.execPath, ['astra_bridge/cli.mjs', '--help'], {
    cwd: new URL('../', import.meta.url), encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /login\|status\|models\|disconnect/);
});

test('CLI 保留登录 API 错误诊断，不输出原始响应或凭据', async t => {
  const root = await mkdtemp(join(tmpdir(), 'duoflow-cli-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'astra_bridge'));
  const sdk = join(root, '.cache/sign-in-with-chatgpt-devkit/packages/local');
  await mkdir(join(sdk, 'dist'), { recursive: true });
  for (const name of ['cli.mjs', 'bridge.mjs']) {
    await copyFile(new URL(name, import.meta.url), join(root, 'astra_bridge', name));
  }
  await writeFile(join(root, 'astra_bridge/credentials.mjs'), 'export async function credentialEncryption() { return {}; }');
  const expected = { code: 'api_error', message: 'Safe diagnostic', status: 403,
    requestId: 'req_test', retryable: false, responseShape: 'undefined' };
  await writeFile(join(sdk, 'package.json'), '{"type":"module"}');
  await writeFile(join(sdk, 'dist/index.js'), `
    export class ChatGPTError extends Error {
      constructor() { super('Safe diagnostic'); this.rawResponse = 'private-token-sentinel'; }
      toJSON() { return ${JSON.stringify(expected)}; }
    }
    export function createChatGPT() {
      return { async signIn() { throw new ChatGPTError(); } };
    }
  `);
  const result = spawnSync(process.execPath, ['astra_bridge/cli.mjs', 'login'], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.deepEqual(JSON.parse(result.stderr), { command: 'login', ...expected });
  assert.equal((result.stdout + result.stderr).includes('private-token-sentinel'), false);
});

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'duoflow-bridge-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const dir of ['requests', 'responses', 'claims']) await mkdir(join(root, dir));
  await writeFile(join(root, 'requests', id + '.json'), JSON.stringify(body));
  return root;
}

test('完成响应原样交给 Python 格式，重复扫描不再次调用', async t => {
  const root = await fixture(t);
  let calls = 0;
  const response = { status: 'completed', output: [{ type: 'message', content: [{ type: 'output_text', text: JSON.stringify(decision) }] }], usage: { total_tokens: 10 } };
  const client = { getSession: connected, async streamResponse(options) {
    calls++; assert.equal(options.model, 'gpt-6-astra');
    return { response, requestId: 'test_request' };
  } };
  assert.equal(await processRequest(root, id, client), true);
  assert.deepEqual(JSON.parse(await readFile(join(root, 'responses', id + '.json'))).response, response);
  assert.equal(await processRequest(root, id, client), false);
  assert.equal(calls, 1);
});

test('中断保留实际增量且只调用一次，重新启动不重发旧 ID', async t => {
  const root = await fixture(t);
  let calls = 0;
  const client = { getSession: connected, async streamResponse(options) {
    calls++; options.onDelta('{'); throw Object.assign(new Error('private detail'), { code: 'stream_interrupted' });
  } };
  await assert.rejects(processRequest(root, id, client), { code: 'stream_interrupted' });
  const receipt = JSON.parse(await readFile(join(root, 'responses', id + '.json')));
  assert.equal(receipt.ok, false);
  assert.equal(receipt.stream_text, '{');
  assert.deepEqual(receipt.transport, { stage: 'upstream', upstream_attempts: 1, automatic_retry: false });
  assert.equal(JSON.stringify(receipt).includes('private detail'), false);
  assert.equal(await processRequest(root, id, client), false);
  assert.equal(calls, 1);
});

test('有领取无响应时停下，不重发，累计预算不因重启归零', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'claims', id + '.json'), '{}');
  await assert.rejects(serve(root, {}, 10), { code: 'prior_request_outcome_unknown' });
  await writeFile(join(root, 'responses', id + '.json'), '{"ok":false}');
  await assert.rejects(serve(root, {}, 1), { code: 'upstream_call_budget_exhausted' });
});

test('未授权计划用量时不发送推理请求，不接受其他模型或 store=true', async t => {
  const root = await fixture(t);
  let calls = 0;
  await assert.rejects(processRequest(root, id, {
    getSession: async () => ({ status: 'connected', sharing: false }),
    streamResponse: async () => { calls++; },
  }), { code: 'existing_sharing_session_required' });
  assert.equal(calls, 0);
  assert.throws(() => requestOptions({ ...body, model: 'another-model' }));
  assert.throws(() => requestOptions({ ...body, store: true }));
});
