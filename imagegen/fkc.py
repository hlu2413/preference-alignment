import torch
from typing import Tuple, Optional, Callable

from imagegen.sd15_pipeline import SD15LatentModel


def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen


def _compute_diversity_grad(z_flat: torch.Tensor, z_shape: Tuple) -> torch.Tensor:
    # Compute pairwise differences: differences[i, j] = z_flat[i] - z_flat[j]
    # Gradient of normalized L2 squared distance loss: ∇L_norm = (2/d) · (1/(n-1)) · ∑_{j≠i} (x_i - x_j)
    # where d is latent dimension, normalized to match reward scale
    # The (i,i) term is zero, so summing over all j is equivalent to summing over j≠i
    differences = z_flat[:, None, :] - z_flat[None, :, :]
    grad_flat = 2 * torch.sum(differences, dim=1)
    # Divide by (n-1) * latent_dim to normalize by dimension and number of particles
    n = z_flat.shape[0]
    latent_dim = z_flat.shape[1]
    return grad_flat.reshape(z_shape) / (max(n - 1, 1) * latent_dim)


def compute_diversity_loss_value(z: torch.Tensor,
                                 historical_particles: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute diversity loss value (scalar per particle): L(z_i) = (1/d) · (1/(n-1)) · ∑_{j≠i} ||z_i - z_j||²
    where d is the latent dimension, normalized to match reward scale [0,1].
    
    Returns:
        loss_values: (n_particles,) tensor of diversity loss values per particle (normalized by dimension)
    """
    z_flat = z.flatten(1)
    n_current = z_flat.shape[0]
    latent_dim = z_flat.shape[1]
    
    # Compute squared distances to other particles in current batch
    differences = z_flat[:, None, :] - z_flat[None, :, :]  # (n, n, d)
    squared_distances = torch.sum(differences ** 2, dim=2)  # (n, n)
    # Sum over all j (including self, but self-distance is 0)
    loss_current = torch.sum(squared_distances, dim=1)  # (n,)
    # Normalize by (n-1) * latent_dim to get average squared distance per dimension
    loss_current = loss_current / (max(n_current - 1, 1) * latent_dim)
    
    if historical_particles is not None and historical_particles.numel() > 0:
        hist_flat = historical_particles.flatten(1)
        n_historical = hist_flat.shape[0]
        # Compute squared distances to historical particles
        differences_hist = z_flat[:, None, :] - hist_flat[None, :, :]  # (n, n_hist, d)
        squared_distances_hist = torch.sum(differences_hist ** 2, dim=2)  # (n, n_hist)
        loss_historical = torch.sum(squared_distances_hist, dim=1) / (n_historical * latent_dim)  # (n,)
        
        # Combine: weighted average
        total_other_particles = (n_current - 1) + n_historical
        return (loss_current * (n_current - 1) + loss_historical * n_historical) / total_other_particles
    
    return loss_current


def compute_diversity_loss_gradient_with_history(z: torch.Tensor,
                                                 historical_particles: Optional[torch.Tensor] = None) -> torch.Tensor:
    z_flat = z.flatten(1)
    n_current = z_flat.shape[0]
    latent_dim = z_flat.shape[1]
    
    # Compute gradient from current batch (normalized by (n_current - 1) * latent_dim)
    grad_current = _compute_diversity_grad(z_flat, z.shape)

    if historical_particles is not None and historical_particles.numel() > 0:
        hist_flat = historical_particles.flatten(1)
        n_historical = hist_flat.shape[0]
        # Compute differences to historical particles: differences[i, j] = z_flat[i] - hist_flat[j]
        differences = z_flat[:, None, :] - hist_flat[None, :, :]
        # Gradient of normalized L2 squared distance loss w.r.t. current particles
        grad_hist_flat = 2 * torch.sum(differences, dim=1)
        grad_historical = grad_hist_flat.reshape(z.shape) / (n_historical * latent_dim)
        
        # Combine: grad_current is per (n_current-1) other particles, 
        # grad_historical is per n_historical historical particles
        # Weight by number of particles each represents, then normalize by total
        total_other_particles = (n_current - 1) + n_historical
        return (grad_current * (n_current - 1) + grad_historical * n_historical) / total_other_particles

    return grad_current


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


def run_fkc_simulation_image(
    z: torch.Tensor,
    w: torch.Tensor,
    reward_grad_fn: Callable,
    reward_fn: Callable,
    beta_t: float,
    gamma_t: float,
    n_steps: int,
    sd_model: SD15LatentModel,
    generator: Optional[torch.Generator] = None,
    score_fn: Optional[Callable] = None,
    historical_particles: Optional[torch.Tensor] = None,
    diversity_enabled: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    scheduler = sd_model.pipe.scheduler
    scheduler.set_timesteps(n_steps, device=sd_model.device)
    sd_model.sigmas = scheduler.sigmas.to(device=sd_model.device)
    timesteps = scheduler.timesteps
    if n_steps < len(timesteps):
        timesteps = timesteps[:n_steps]
    dt_abs = 1.0 / max(len(timesteps), 1)

    for idx, timestep in enumerate(timesteps):
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

        # Combine reward and diversity: new_reward = reward + diversity
        # Since both are scalar functions, we can linearly combine their gradients
        div_grad = torch.zeros_like(z)
        if diversity_enabled:
            div_grad = compute_diversity_loss_gradient_with_history(z, historical_particles)
        
        # Combined reward gradient: ∇(reward + γ_t * diversity) = ∇reward + γ_t * ∇diversity
        combined_r_grad = r_grad + gamma_t * div_grad

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

        # Construct FKC drift: dz = [σ_t²(score + (β_t/2)(r_grad + γ_t·∇div)) - f_t] dt + σ_t dW_t
        # Diversity is now part of the combined reward gradient
        drift_fkc = sigma_sq * (score + (beta_t / 2.0) * combined_r_grad) - f_t

        # Convert drift to noise prediction format for scheduler
        # The relationship: drift = σ_t²(score + (β_t/2)(r_grad + γ_t·∇div)) - f_t
        # For the scheduler: noise_pred = -score * sigma_t
        # We convert drift to noise_pred by: noise_pred = -drift / σ_t
        # This works because the scheduler internally handles the conversion
        noise_pred_fkc = -drift_fkc / sigma_t

        # Apply FKC-modified step via scheduler
        step_result = scheduler.step(noise_pred_fkc, timestep, z, generator=sub_gen)
        z = step_result.prev_sample

        # Weight update: dw = (∂β_t/∂t)·r(z_t)·dt - ⟨β_t·∇(r + γ_t·∇div), f_t⟩·dt + ⟨β_t·∇(r + γ_t·∇div), (σ_t²/2)·score⟩·dt
        # Diversity only appears in gradient (∇div), not in reward value
        # Combined reward gradient: ∇(r + γ_t·div) = ∇r + γ_t·∇div (already computed above)
        with torch.no_grad():
            rewards = reward_fn(z)

        beta_dot = 1.0  # Rate of change of beta_t (assuming maximum reward strength)

        # Flatten for dot products
        combined_r_grad_flat = combined_r_grad.flatten(1)
        score_flat = score.flatten(1)
        f_t_flat = f_t.flatten(1)

        # Term 1: Direct reward term (diversity only in gradient, not in value)
        term1 = beta_dot * rewards * dt_abs

        # Term 2: Combined reward-drift interaction (negative) - uses combined gradient (∇r + γ_t·∇div)
        term2 = -torch.sum(beta_t * combined_r_grad_flat * f_t_flat, dim=1) * dt_abs

        # Term 3: Combined reward-score interaction (positive, noise-dependent) - uses combined gradient (∇r + γ_t·∇div)
        term3 = torch.sum(beta_t * combined_r_grad_flat * (sigma_sq / 2.0) * score_flat, dim=1) * dt_abs

        dw = term1 + term2 + term3
        
        # Clear intermediate tensors
        del combined_r_grad_flat, score_flat, f_t_flat, term1, term2, term3, rewards
        
        dw = torch.clamp(dw, -1.0, 1.0)
        w = torch.clamp(w + dw, -100.0, 100.0)
        del dw

    w_min, w_max = torch.min(w), torch.max(w)
    w_range = w_max - w_min
    if w_range.item() > 1e-8:
        w = (w - w_min) / w_range
    else:
        w = w - w_min

    return z, w
