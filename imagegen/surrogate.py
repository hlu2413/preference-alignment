import torch
import torch.nn as nn
from typing import Tuple, Callable


class LatentSurrogate(nn.Module):
    def __init__(self, latent_shape: Tuple[int, int, int], hidden_dim: int = 128):
        super().__init__()
        c, h, w = latent_shape
        if h > 1 and w > 1:
            self.conv = nn.Sequential(
                nn.Conv2d(c, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            input_dim = 64
        else:
            self.conv = None
            input_dim = c * h * w
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.parameters()).dtype
        if z.dtype != param_dtype:
            z = z.to(dtype=param_dtype)
        if self.conv is not None:
            feats = self.conv(z)
        else:
            feats = z.flatten(1)
        return self.mlp(feats).squeeze(-1)


def create_surrogate_and_grad(latent_shape: Tuple[int, int, int], device: torch.device | None = None) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LatentSurrogate(latent_shape).to(device)
    def grad_fn(z: torch.Tensor) -> torch.Tensor:
        z = z.clone().requires_grad_(True)
        reward = model(z)
        reward_sum = reward.sum() if reward.ndim > 0 else reward
        grad = torch.autograd.grad(reward_sum, z, retain_graph=False, create_graph=False)[0]
        return grad
    return model, grad_fn


