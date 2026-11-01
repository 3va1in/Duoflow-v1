"""按仓库锁定版本下载基础模型及两个 tokenizer，不包含 RoboTwin 场景。"""
import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(os.environ.get('TWINTIMEFM_ASSETS', ROOT / 'models/base')))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    sources = json.loads((ROOT / 'runtime/XPolicyLab/policy/TwinTimeFM/ASSET_SOURCES.json').read_text())
    for key, directory in [('model', 'backbone'), ('paligemma2_tokenizer', 'paligemma2_tokenizer'),
                           ('fast_tokenizer', 'fast_tokenizer')]:
        source = sources[key]
        repo_id = source.get('repo_id') or source['repository']
        for filename in source['files']:
            print(f'{repo_id}@{source["revision"]}: {filename}', flush=True)
            hf_hub_download(repo_id=repo_id, revision=source['revision'], filename=filename,
                            local_dir=str(args.output / directory))
    raw = json.loads((args.output / 'backbone/norm_stats.json').read_text())
    stats = raw.get('norm_stats', raw)
    expected = json.loads((ROOT / 'pretrained/simulation/norm_stats.json').read_text())['norm_stats']
    if any(stats[key] != expected[key] for key in ('observation.state', 'action')):
        raise ValueError('基础模型归一化与已发布的仿真技能不一致')
    print('基础模型与 tokenizer 已下载到: ' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
