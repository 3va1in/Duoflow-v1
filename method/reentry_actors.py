"""同一冻结骨干上显式切换原 LoRA 和技能 LoRA；完成头独立加载。"""
import numpy as np
import torch

from atomic_skill_dataset import completion_images, condition_key, legacy_module, load_norm
from atomic_skill_model import CompletionHead


class SharedAtomicActors:
    def __init__(self, engine, policy_root, actor_checkpoint, completion_checkpoint, audit_text):
        actor = torch.load(actor_checkpoint, map_location='cpu', weights_only=False)
        head = torch.load(completion_checkpoint, map_location='cpu', weights_only=False)
        if actor.get('format') != 'atomic_skill_lora_v1' or actor.get('run_complete') is not True:
            raise ValueError('技能 actor checkpoint 格式不符')
        self.completion_pair = head.get('format') == 'atomic_skill_completion_pair_v2'
        if head.get('format') not in {'atomic_skill_completion_repair_v1', 'atomic_skill_completion_pair_v2'}:
            raise ValueError('必须使用新训练的完成头 checkpoint')
        if (actor['condition_to_index'] != head['condition_to_index'] or
                actor['norm_stats'] != head['norm_stats']):
            raise ValueError('技能 actor 与完成头的条件映射或归一化不一致')
        runtime = legacy_module(policy_root, 'runtime')
        if load_norm(runtime.MODEL_ROOT / 'norm_stats.json') != actor['norm_stats']:
            raise ValueError('技能归一化与当前动作骨干不一致')
        if actor['adapter_state'].get('adapter'):
            raise ValueError('该执行路径只支持共享骨干上的 LoRA')
        parameters = dict(engine.model.named_parameters())
        expected = {k for k in parameters if k.endswith(('.lora_a', '.lora_b'))}
        weights = actor['adapter_state']['lora']
        if not expected or set(weights) != expected:
            raise ValueError('技能 LoRA 与当前原 actor 的参数集合不一致')
        self.parameters = {k: parameters[k] for k in sorted(expected)}
        self.states = {'whole': {}, 'atomic': {}}
        for name, parameter in self.parameters.items():
            if tuple(weights[name].shape) != tuple(parameter.shape):
                raise ValueError(f'技能 LoRA 参数 shape 不一致: {name}')
            self.states['whole'][name] = parameter.detach().clone()
            self.states['atomic'][name] = weights[name].to(parameter.device, parameter.dtype)
        self.device = next(iter(self.parameters.values())).device
        self.condition_to_index = head['condition_to_index']
        stop = head['stop_config']
        if stop['consecutive_observations'] != 2:
            raise ValueError('完成头要求两次真实观测确认')
        self.threshold = float(stop['threshold'])
        legacy = legacy_module(policy_root, 'dataset')
        self.norm = legacy._QuantileNorm(head['norm_stats']['observation.state'], head['norm_stats']['action'])
        if self.completion_pair:
            from pair_completion import PairCompletionPredictor
            self.completion = PairCompletionPredictor(head, self.device, self.norm)
        else:
            self.completion = CompletionHead(len(head['condition_to_index'])).to(self.device)
            self.completion.load_state_dict(head['completion_state'], strict=True)
            self.completion.eval()
        self.engine, self.audit_text = engine, audit_text
        self.mode, self.last_input = 'whole', None

    def reset(self):
        self.last_input = None

    def _activate(self, mode):
        if self.mode != mode:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(self.states[mode][name])
            self.mode, self.last_input = mode, None

    def supports(self, card):
        return condition_key(card['atomic_skill_id'], card['execution_prompt']) in self.condition_to_index

    @torch.inference_mode()
    def _actions(self, mode, state, images, instruction, reset=False):
        self._activate(mode)
        key = (mode, instruction)
        self.audit_text(instruction, mode)
        actions = np.asarray(self.engine.predict(state, images, instruction,
                             reset=bool(reset or self.last_input != key)), dtype=np.float32)
        self.last_input = key
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError('共享动作骨干必须返回有限的 [50,14] 动作')
        return actions

    def whole(self, state, images, instruction):
        return self._actions('whole', state, images, instruction)

    @torch.inference_mode()
    def predict(self, state, images, skill_text, skill_id, reset=False, completion_reference=None):
        key = condition_key(skill_id, skill_text)
        if key not in self.condition_to_index:
            raise ValueError(f'未训练的技能绑定: {key}')
        index = torch.tensor([self.condition_to_index[key]], device=self.device)
        if self.completion_pair:
            probability = self.completion.predict(state, images, index, reference=completion_reference)
        else:
            raw = torch.as_tensor(np.asarray(state, dtype=np.float32))
            head_images = completion_images(images).unsqueeze(0).to(self.device)
            head_state = self.norm.apply(raw, kind='state').unsqueeze(0).to(self.device)
            probability = float(self.completion(head_images, head_state, index).sigmoid().item())
        return {'completion_prob': probability,
                'actions': self._actions('atomic', state, images, skill_text, reset)}
