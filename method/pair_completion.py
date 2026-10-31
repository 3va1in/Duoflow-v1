"""以技能调用起点和当前公开观测判断完成；训练物理标签不进入推理。"""
import numpy as np
import torch
from torch import nn


class PairCompletionHead(nn.Module):
    def __init__(self, condition_count):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.condition = nn.Embedding(condition_count, 32)
        self.readout = nn.Sequential(nn.Linear(3 * 3 * 64 * 4 + 3 * 14 + 32, 256),
                                     nn.ReLU(), nn.Linear(256, 1))

    def forward(self, initial_images, initial_state, images, state, condition):
        batch = images.shape[0]
        paired = torch.cat((initial_images, images), dim=0)
        features = self.visual(paired.flatten(0, 1)).reshape(2 * batch, -1)
        start, current = features[:batch], features[batch:]
        inputs = torch.cat((start, current, current - start, initial_state, state,
                            state - initial_state, self.condition(condition)), dim=-1)
        return self.readout(inputs).squeeze(-1)


class PairCompletionPredictor:
    def __init__(self, checkpoint, device, norm):
        if checkpoint['format'] != 'atomic_skill_completion_pair_v2':
            raise ValueError('完成头不是成对公开观测版本')
        self.device, self.norm = device, norm
        self.model = PairCompletionHead(len(checkpoint['condition_to_index'])).to(device)
        self.model.load_state_dict(checkpoint['completion_state'], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, state, images, index, reference):
        if reference is None:
            raise ValueError('成对完成头缺少本次技能调用的起始公开观测')
        from atomic_skill_dataset import completion_images
        current_images = completion_images(images).unsqueeze(0).to(self.device)
        initial_images = completion_images(reference['images']).unsqueeze(0).to(self.device)
        current_state = self.norm.apply(torch.as_tensor(np.asarray(state, dtype=np.float32)),
                                       kind='state').unsqueeze(0).to(self.device)
        initial_state = self.norm.apply(torch.as_tensor(np.asarray(reference['state'], dtype=np.float32)),
                                       kind='state').unsqueeze(0).to(self.device)
        return float(self.model(initial_images, initial_state, current_images, current_state,
                                index).sigmoid().item())
