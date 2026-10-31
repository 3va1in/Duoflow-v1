import { mkdir, open, readFile, readdir, rename, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

const fail = code => Object.assign(new Error(code), { code });
const exists = async path => stat(path).then(() => true, error => {
  if (error.code === 'ENOENT') return false;
  throw error;
});

export function requestOptions(body) {
  const allowed = ['model', 'reasoning', 'store', 'stream', 'instructions', 'input', 'text'];
  if (!body || Object.keys(body).some(key => !allowed.includes(key)) ||
      body.model !== 'gpt-6-astra' || body.store !== false || body.stream !== true ||
      body.reasoning?.effort !== 'low' || typeof body.instructions !== 'string' ||
      !Array.isArray(body.input) || body.text?.format?.type !== 'json_schema' ||
      body.text.format.strict !== true || body.text.format.name !== 'guided_skill_decision') {
    throw fail('invalid_duoflow_request');
  }
  return { model: body.model, reasoning: body.reasoning, instructions: body.instructions,
    input: body.input, text: body.text };
}

async function writeResponse(path, value) {
  // 客户端仅在 JSON 完整落盘后看到最终文件。
  const temporary = path + '.tmp';
  const file = await open(temporary, 'wx', 0o600);
  try { await file.writeFile(JSON.stringify(value) + '\n'); } finally { await file.close(); }
  await rename(temporary, path);
}

export async function processRequest(root, id, client) {
  if (!/^[a-f0-9]{32}$/.test(id)) throw fail('invalid_request_id');
  const output = join(root, 'responses', id + '.json');
  if (await exists(output)) return false;
  let claim;
  try { claim = await open(join(root, 'claims', id + '.json'), 'wx', 0o600); }
  catch (error) {
    if (error.code === 'EEXIST') throw fail('prior_request_outcome_unknown');
    throw error;
  }
  // 崩溃后不重发旧 ID：claim 无 response 需要人工检查。
  await claim.writeFile(JSON.stringify({ pid: process.pid, started_at: new Date().toISOString() }));
  await claim.close();
  const start = performance.now();
  let stage = 'validation', attempts = 0, streamText = '', deltas = 0, envelope;
  try {
    const options = requestOptions(JSON.parse(await readFile(join(root, 'requests', id + '.json'), 'utf8')));
    stage = 'session';
    const session = await client.getSession();
    if (session.status !== 'connected' || !session.sharing) throw fail('existing_sharing_session_required');
    stage = 'upstream';
    attempts = 1;
    const result = await client.streamResponse({ ...options, signal: AbortSignal.timeout(180_000),
      onDelta(delta) { streamText += delta; deltas++; } });
    stage = 'completed_response_validation';
    let response = result.response;
    if (response?.status !== 'completed' || !Array.isArray(response.output)) throw fail('invalid_completed_response');
    if (!response.output.length && streamText.trim()) {
      response = { ...response, output: [{ type: 'message', role: 'assistant',
        content: [{ type: 'output_text', text: streamText }] }] };
    }
    if (!response.output.length) throw fail('missing_structured_output');
    envelope = { ok: true, response, request_id: result.requestId ?? null };
  } catch (error) {
    const code = typeof error.code === 'string' && /^[a-zA-Z0-9_:-]{1,100}$/.test(error.code)
      ? error.code : 'local_bridge_error';
    envelope = { ok: false, error: { code, ...(Number.isInteger(error.status) ? { status: error.status } : {}) } };
  }
  envelope = { ...envelope, stream_text: streamText, received_delta_count: deltas,
    latency_seconds: (performance.now() - start) / 1000,
    transport: { stage, upstream_attempts: attempts, automatic_retry: false } };
  await writeResponse(output, envelope);
  console.log(JSON.stringify({ request_id: id, ok: envelope.ok, error: envelope.error?.code,
    usage: envelope.response?.usage, latency_seconds: envelope.latency_seconds }));
  // 异常时停止队列，不擅自改模型或重试；已写回的错误会由 Python 客户端抛出。
  if (!envelope.ok) throw fail(envelope.error.code);
  return true;
}

export async function serve(root, client, maxRequests) {
  for (const name of ['requests', 'responses', 'claims']) await mkdir(join(root, name), { recursive: true, mode: 0o700 });
  const claims = (await readdir(join(root, 'claims'))).filter(name => /^[a-f0-9]{32}\.json$/.test(name));
  for (const name of claims) {
    if (!await exists(join(root, 'responses', name))) throw fail('prior_request_outcome_unknown');
  }
  let claimed = claims.length;
  let stopping = false;
  const stop = () => { stopping = true; };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  console.log(`Astra 桥已监听 ${root}；串行，累计请求上限 ${maxRequests}，不自动重试。`);
  try {
    while (!stopping) {
      if (claimed >= maxRequests) throw fail('upstream_call_budget_exhausted');
      const names = (await readdir(join(root, 'requests'))).filter(name => /^[a-f0-9]{32}\.json$/.test(name)).sort();
      for (const name of names) {
        if (stopping || claimed >= maxRequests) break;
        if (await processRequest(root, name.slice(0, -5), client)) claimed++;
      }
      if (!stopping) await sleep(250);
    }
  } finally {
    process.removeListener('SIGINT', stop);
    process.removeListener('SIGTERM', stop);
  }
}
