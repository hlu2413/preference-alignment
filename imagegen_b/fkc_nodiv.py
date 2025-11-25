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
        z_prev = z  # soft-bound reference for proximal penalty
        sub_gen = _sub_generator(generator)
        if len(timesteps) > 1:
            t_normalized = 1.0 - idx / (len(timesteps) - 1)
        else:
            t_normalized = 1.0

        sigma_index = min(idx, sd_model.sigmas.shape[0] - 1)
        sigma_t = sd_model.sigmas[sigma_index].to(device=z.device, dtype=z.dtype)
        sigma_sq = sigma_t ** 2

        # Get gradients
        score = (-z if score_fn is None else score_fn(z, t_normalized)).to(dtype=z.dtype)
        r_grad = reward_grad_fn(z)

        # Diversity gradient is not computed (diversity disabled)
        div_grad = torch.zeros_like(z)

        # Proximal penalty gradient (soft bound)
        prox_grad = z - z_prev

        # Derive baseline drift f_t from scheduler
        # Compute baseline drift manually to avoid calling scheduler.step() twice
        # For EulerAncestralDiscreteScheduler, we need to get the next sigma
        sigma_next_index = min(idx + 1, sd_model.sigmas.shape[0] - 1)
        sigma_next = sd_model.sigmas[sigma_next_index].to(device=z.device, dtype=z.dtype)
        
        # Compute what the scheduler would do for baseline step
        # EulerAncestralDiscreteScheduler step: prev_sample = sample - (sigma_t - sigma_next) * noise_pred
        # But we need the deterministic drift component (without the stochastic noise)
        baseline_noise_pred = -score * sigma_t
        sigma_diff = sigma_t - sigma_next
        baseline_prev_sample_deterministic = z - sigma_diff * baseline_noise_pred
        
        # The baseline drift f_t is the deterministic component
        f_t = (baseline_prev_sample_deterministic - z) / dt_abs

        # Construct FKC drift: dz = [σ_t²(score + (β_t/2)r_grad) - f_t - λ_t·prox_grad] dt + σ_t dW_t
        # (diversity term excluded since diversity_enabled=False)
        drift_fkc = (sigma_sq * (score + (beta_t / 2.0) * r_grad)
                     - f_t
                     - lambda_t * prox_grad)

        # Convert drift to noise prediction format for scheduler
        noise_pred_fkc = -drift_fkc / sigma_t

        # Apply FKC-modified step via scheduler
        step_result = scheduler.step(noise_pred_fkc, timestep, z, generator=sub_gen)
        z = step_result.prev_sample

        # Weight update: dw = (∂β_t/∂t)·r(z_t)·dt - ⟨β_t·∇r, f_t⟩·dt + ⟨β_t·∇r, (σ_t²/2)·score⟩·dt
        # (diversity term excluded since diversity_enabled=False)
        with torch.no_grad():
            rewards = reward_fn(z)

        beta_dot = 1.0  # Rate of change of beta_t (assuming maximum reward strength)

        # Flatten for dot products
        r_grad_flat = r_grad.flatten(1)
        score_flat = score.flatten(1)
        f_t_flat = f_t.flatten(1)

        # Term 1: Direct reward term
        term1 = beta_dot * rewards * dt_abs

        # Term 2: Reward-drift interaction (negative)
        term2 = -torch.sum(beta_t * r_grad_flat * f_t_flat, dim=1) * dt_abs

        # Term 3: Reward-score interaction (positive, noise-dependent)
        term3 = torch.sum(beta_t * r_grad_flat * (sigma_sq / 2.0) * score_flat, dim=1) * dt_abs

        dw = term1 + term2 + term3
        dw = torch.clamp(dw, -1.0, 1.0)
        w = torch.clamp(w + dw, -100.0, 100.0)

    w_min, w_max = torch.min(w), torch.max(w)
    w_range = w_max - w_min
    if w_range.item() > 1e-8:
        w = (w - w_min) / w_range
    else:
        w = w - w_min

    return z, w

