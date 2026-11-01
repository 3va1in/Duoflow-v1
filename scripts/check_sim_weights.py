"""在 CPU 上核对仿真权重包，不连接机器人或调用 Astra。"""
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from skill_model import CompletionHead


def main():
    bundle = ROOT / 'pretrained/simulation'
    manifest = json.loads((bundle / 'BUNDLE.json').read_text())
    for name, record in manifest['files'].items():
        path = bundle / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('模型包文件缺失或与发布版本不同: ' + name)
    norm = json.loads((bundle / 'norm_stats.json').read_text())['norm_stats']
    native = load_file(str(bundle / manifest['native']['adapter']), device='cpu')
    summary = {}
    for bank, info in manifest['banks'].items():
        actor = torch.load(bundle / info['actor'], map_location='cpu', weights_only=True)
        head = torch.load(bundle / info['completion'], map_location='cpu', weights_only=True)
        catalog = json.loads((bundle / info['catalog']).read_text())
        assert actor['format'] == 'atomic_skill_lora_v1' and actor['run_complete']
        assert head['format'] == 'atomic_skill_completion_repair_v1'
        assert actor['condition_to_index'] == head['condition_to_index']
        assert actor['norm_stats'] == head['norm_stats'] == norm
        assert head['stop_config'] == info['stop_config'] == catalog['stop_config']
        lora = actor['adapter_state']['lora']
        assert not actor['adapter_state'].get('adapter') and lora.keys() == native.keys()
        assert all(lora[k].shape == native[k].shape for k in native)
        CompletionHead(len(head['condition_to_index'])).load_state_dict(head['completion_state'], strict=True)
        for card in catalog['skills']:
            assert actor['condition_to_index'][card['condition_key']] == card['condition_index']
            assert manifest['task_to_bank'][card['task']] == bank
            assert (ROOT / card['policy_checkpoint']).is_file()
            assert (ROOT / card['completion_checkpoint']).is_file()
        assert len(catalog['skills']) == info['card_count']
        summary[bank] = {'tasks': len(info['tasks']), 'cards': len(catalog['skills']),
                         'conditions': len(head['condition_to_index'])}
    selector = torch.load(bundle / manifest['native']['selector'], map_location='cpu', weights_only=True)
    assert selector['format'] == 'system2_selector_v1' and selector['scorer_state']
    assert len(manifest['task_to_bank']) == 50
    print(json.dumps({'status': 'ok', 'banks': summary, 'files': len(manifest['files']),
                      'check': 'CPU files, tensor shapes, completion heads and condition mappings',
                      'backbone_inference_tested': False, 'robot_connected': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
