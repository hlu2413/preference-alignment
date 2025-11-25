import torch
from typing import Tuple, Optional, Callable

from imagegen_b.sd15_pipeline import SD15LatentModel


def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen


def gamma_schedule(step: int, total_steps: int, gamma_max: float = 0.05, gamma_min: float = 0.0) -> float:
    if total_steps <= 1:
        return gamma_max
    progress = step / (total_steps - 1)
    return gamma_max * (1 - progress) + gamma_min * progress


def beta_schedule(step: int, total_steps: int, beta_min: float = 0.5, beta_max: float = 2.0) -> float:
    if total_steps <= 1:
        return beta_max
    progress = step / (total_steps - 1)
    return beta_min * (1 - progress) + beta_max * progress


def lambda_schedule(step: int, total_steps: int, lambda_max: float = 1.0, lambda_min: float = 0.2) -> float:
    if total_steps <= 1:
        return lambda_max
    progress = step / (total_steps - 1)
    return lambda_max * (1 - progress) + lambda_min * progress


def run_fkc_simulation_image(
    z: torch.Tensor,
    w: torch.Tensor,
    reward_grad_fn: Callable,
    reward_fn: Callable,
    beta_t: float,
    gamma_t: float,
    lambda_t: float,  # soft-bound proximal penalty coefficient
    n_steps: int,
    sd_model: SD15LatentModel,
    generator: Optional[torch.Generator] = None,
    score_fn: Optional[Callable] = None,
    historical_particles: Optional[torch.Tensor] = None,
    diversity_enabled: bool = False  # Diversity disabled by default
) -> Tuple[torch.Tensor, torch.Tensor]:
    scheduler = sd_model.pipe.scheduler
    scheduler.set_timesteps(n_steps, device=sd_model.device)
    sd_model.sigmas = scheduler.sigmas.to(device=sd_model.device)
    timesteps = scheduler.timesteps
    if n_steps < len(timesteps):
        timesteps = timesteps[:n_steps]
    dt_abs = 1.0 / max(len(timesteps), 1)

    for idx, timestep in enumerate(timesteps):
        z_prev = z  # soft-bound reference
        sub_gen = _sub_generator(generator)
        if len(timesteps) > 1:
            t_normalized = 1.0 - idx / (len(timesteps) - 1)
        else:
            t_normalized = 1.0

        sigma_index = min(idx, sd_model.sigmas.shape[0] - 1)
        sigma_t = sd_model.sigmas[sigma_index].to(device=z.device, dtype=z.dtype)
        sigma_sq = torch.clamp(sigma_t ** 2, min=torch.tensor(1e-6, device=z.device, dtype=z.dtype))

        score = (-z if score_fn is None else score_fn(z, t_normalized)).to(dtype=z.dtype)
        r_grad = reward_grad_fn(z)

        # Diversity gradient is not computed (diversity disabled)
        div_grad = torch.zeros_like(z)

        # Normalize auxiliary gradients to score magnitude
        score_norm = torch.norm(score.flatten(1), dim=1, keepdim=True)
        r_grad_norm = torch.norm(r_grad.flatten(1), dim=1, keepdim=True)
        score_norm = torch.clamp(score_norm, min=1e-8)
        r_grad_norm = torch.clamp(r_grad_norm, min=1e-8)
        r_scale = (score_norm / r_grad_norm).view(-1, 1, 1, 1)
        r_scale = torch.clamp(r_scale, min=0.1, max=10.0)
        r_grad_normalized = r_grad * r_scale

        # Diversity gradient normalization is skipped (diversity disabled)
        div_grad_normalized = torch.zeros_like(z)

        # Proximal penalty gradient (soft bound)
        prox_grad = z - z_prev
        prox_norm = torch.norm(prox_grad.flatten(1), dim=1, keepdim=True)
        prox_norm = torch.clamp(prox_norm, min=1e-8)
        p_scale = (score_norm / prox_norm).view(-1, 1, 1, 1)
        p_scale = torch.clamp(p_scale, min=0.1, max=10.0)
        prox_grad_normalized = prox_grad * p_scale

        # Score total without diversity term
        score_total = score + (beta_t / 2.0) * r_grad_normalized
        # Diversity term is not added (diversity disabled)
        # subtract proximal pullback
        score_total = score_total - lambda_t * prox_grad_normalized

        noise_pred = -score_total * sigma_t
        step_result = scheduler.step(noise_pred, timestep, z, generator=sub_gen)
        z = step_result.prev_sample

        with torch.no_grad():
            rewards = reward_fn(z)

        r_grad_flat = r_grad.flatten(1)
        score_flat = score.flatten(1)
        term1 = rewards * dt_abs
        term2 = -beta_t * torch.sum(r_grad_flat * score_flat, dim=1) * dt_abs
        term3 = (sigma_t**2 / 2.0) * torch.sum(r_grad_flat * score_flat, dim=1) * dt_abs
        dw = term1 + term2 + term3
        # Diversity term is not added to weight update (diversity disabled)
        dw = torch.clamp(dw, -1.0, 1.0)
        w = torch.clamp(w + dw, -100.0, 100.0)

    w_min, w_max = torch.min(w), torch.max(w)
    w_range = w_max - w_min
    if w_range.item() > 1e-8:
        w = (w - w_min) / w_range
    else:
        w = w - w_min

    return z, w

