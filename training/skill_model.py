"""动作 LoRA 与独立学习的技能完成头；部署不接收物理谓词。"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from skill_dataset import CAMERAS, completion_images, condition_key, legacy_module, load_norm, skill_instruction


class CompletionHead(nn.Module):
    def __init__(self, condition_count):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1))
        self.condition = nn.Embedding(condition_count, 32)
        self.readout = nn.Sequential(nn.Linear(3 * 64 + 14 + 32, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, images, state, condition_index):
        batch, views = images.shape[:2]
        visual = self.visual(images.reshape(-1, *images.shape[2:])).reshape(batch, views * 64)
        return self.readout(torch.cat((visual, state, self.condition(condition_index)), dim=-1)).squeeze(-1)


def read_adapter(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        return {'lora': load_file(str(path), device='cpu'), 'adapter': {}}
    value = torch.load(path, map_location='cpu', weights_only=False)
    return value.get('adapter_state', value)


class AtomicSkillModel:
    def __init__(self, policy_root, condition_to_index, norm_stats, device='cuda', init_checkpoint=None):
        self.policy_root, self.device = str(Path(policy_root).resolve()), torch.device(device)
        self.condition_to_index, self.norm_stats = dict(condition_to_index), norm_stats
        self.runtime = legacy_module(self.policy_root, 'runtime')
        if load_norm(self.runtime.MODEL_ROOT / 'norm_stats.json') != norm_stats:
            raise ValueError('技能训练归一化必须与部署骨干的原统计一致')
        lora = legacy_module(self.policy_root, 'lora_policy')
        self.model = self.runtime.load_official().to(self.device)
        if (self.model.n_action_steps, self.model.max_action_dim) != (50, 32):
            raise ValueError('该训练入口要求原50步/32维动作骨干')
        self.actor = lora.TwinTimePolicy(self.model, mode='FT')
        self.actor.configure_trainable(rank=8, alpha=16)
        if init_checkpoint is not None:
            self.actor.load_trainable_state_dict(read_adapter(init_checkpoint), strict=True)
        self.model.eval()
        self.completion = CompletionHead(len(condition_to_index)).to(self.device)
        legacy = legacy_module(self.policy_root, 'dataset')
        self.norm = legacy._QuantileNorm(norm_stats['observation.state'], norm_stats['action'])

    @classmethod
    def load(cls, checkpoint_path, policy_root, device='cuda', norm_stats_path=None):
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if payload.get('format') != 'atomic_skill_lora_v1':
            raise ValueError('不是独立技能训练 checkpoint')
        stats = load_norm(norm_stats_path) if norm_stats_path else payload['norm_stats']
        instance = cls(policy_root, payload['condition_to_index'], stats, device=device)
        instance.actor.load_trainable_state_dict(payload['adapter_state'], strict=True)
        instance.completion.load_state_dict(payload['completion_state'], strict=True)
        instance.completion.eval()
        return instance

    @torch.inference_mode()
    def predict(self, state, images, skill_text, skill_id, reset=False):
        instruction = skill_instruction(skill_text)
        key = condition_key(skill_id, instruction)
        if key not in self.condition_to_index:
            raise ValueError(f'unsupported 技能绑定条件: {key}')
        raw_state = np.asarray(state, dtype=np.float32)
        if raw_state.shape != (14,):
            raise ValueError('state 必须是14维')
        rgb = {key: np.asarray(images[key], dtype=np.uint8) for key in CAMERAS}
        stop_images = completion_images(rgb).unsqueeze(0).to(self.device)
        stop_state = self.norm.apply(torch.from_numpy(raw_state), kind='state').unsqueeze(0).to(self.device)
        index = torch.tensor([self.condition_to_index[key]], device=self.device)
        probability = self.completion(stop_images, stop_state, index).sigmoid().item()
        request = {'observation.state': torch.from_numpy(raw_state), 'task': instruction,
                   'reset_observation_memory': bool(reset)}
        for key, value in rgb.items():
            request['observation.images.' + key] = torch.from_numpy(
                np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        actions = self.model.inference(request).detach().float().cpu().numpy()
        return {'actions': actions, 'completion_prob': float(probability)}
