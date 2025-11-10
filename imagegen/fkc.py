import torch
from typing import Tuple, Optional, Callable


def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen


def _compute_diversity_grad(z_flat: torch.Tensor, z_shape: Tuple) -> torch.Tensor:
    differences = z_flat[:, None, :] - z_flat[None, :, :]
    grad_flat = 2 * torch.sum(differences, dim=1)
    return grad_flat.reshape(z_shape) / z_flat.shape[0]


def compute_diversity_loss_gradient_with_history(z: torch.Tensor, 
                                                 historical_particles: Optional[torch.Tensor] = None) -> torch.Tensor:
    z_flat = z.flatten(1)
    grad_current = _compute_diversity_grad(z_flat, z.shape)
    
    if historical_particles is not None and historical_particles.numel() > 0:
        hist_flat = historical_particles.flatten(1)
        differences = z_flat[:, None, :] - hist_flat[None, :, :]
        grad_hist_flat = 2 * torch.sum(differences, dim=1)
        grad_historical = grad_hist_flat.reshape(z.shape)
        total = z.shape[0] + historical_particles.shape[0]
        return (grad_current * z.shape[0] + grad_historical) / total
    
    return grad_current


def gamma_schedule(step: int, total_steps: int, gamma_max: float = 0.05, gamma_min: float = 0.0) -> float:
    if total_steps <= 1:
        return gamma_max
    progress = step / (total_steps - 1)
    return gamma_max * (1 - progress) + gamma_min * progress


def run_fkc_simulation_image(
    z: torch.Tensor,
    w: torch.Tensor,
    reward_grad_fn: Callable,
    reward_fn: Callable,
    beta_t: float,
    gamma_t: float,
    n_steps: int,
    generator: Optional[torch.Generator] = None,
    score_fn: Optional[Callable] = None,
    historical_particles: Optional[torch.Tensor] = None,
    diversity_enabled: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = -1.0 / n_steps
    
    for i in range(n_steps):
        t = 1.0 + i * dt
        sub_gen = _sub_generator(generator)

        sigma_t = 0.1 * (1 - t)
        score = (-z if score_fn is None else score_fn(z, t))
        r_grad = reward_grad_fn(z)

        div_grad = torch.zeros_like(z)
        if diversity_enabled and historical_particles is not None:
            div_grad = compute_diversity_loss_gradient_with_history(z, historical_particles)

        drift = sigma_t**2 * (score + (beta_t / 2) * r_grad) + gamma_t * div_grad
        noise = sigma_t * torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=sub_gen)
        dt_abs = abs(dt)
        z = z + drift * dt + noise * torch.sqrt(torch.tensor(dt_abs, device=z.device, dtype=z.dtype))

        with torch.no_grad():
            rewards = reward_fn(z)

        r_grad_flat = r_grad.flatten(1)
        score_flat = score.flatten(1)
        term1 = rewards * dt_abs
        term2 = -beta_t * torch.sum(r_grad_flat * score_flat, dim=1) * dt_abs
        term3 = (sigma_t**2 / 2) * torch.sum(r_grad_flat * score_flat, dim=1) * dt_abs
        dw = term1 + term2 + term3
        if diversity_enabled:
            div_grad_flat = div_grad.flatten(1)
            dw += gamma_t * torch.sum(div_grad_flat * score_flat, dim=1) * dt_abs
        dw = torch.clamp(dw, -1.0, 1.0)
        w = torch.clamp(w + dw, -100.0, 100.0)

    w_min, w_max = torch.min(w), torch.max(w)
    if (w_max - w_min).item() > 1e-8:
        w = (w - w_min) / (w_max - w_min)
    else:
        w = torch.ones_like(w) * 0.5
    
    return z, w


