"""
Multi-reward FKC formulation using Jacobian approach.
Handles arbitrary number of reward landscapes with adaptive weight updates.
"""

import torch
from typing import Tuple, Callable, List
from base import baseline_drift, diffusion_coefficient, baseline_score_function


def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    """derive a sub-generator deterministically from an existing generator"""
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen

def compute_jacobian(reward_grad_fns: List[Callable], x: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian matrix for multiple reward landscapes.
    
    Args:
        reward_grad_fns: List of gradient functions, one per reward landscape
        x: Particles (n_particles, dim)
    
    Returns:
        jacobian: (n_rewards, n_particles, dim)
    """
    # Stack gradients from all reward functions
    gradients = [grad_fn(x) for grad_fn in reward_grad_fns]
    # Stack along first dimension: (n_rewards, n_particles, dim)
    return torch.stack(gradients, dim=0)


def compute_combined_reward_gradient(jacobian: torch.Tensor, 
                                    individual_rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of combined reward: ∇(r₁r₂...rₙ) using product rule.
    
    Product rule: ∇(r₁r₂...rₙ) = (r₁r₂...rₙ) * Σᵢ(∇rᵢ/rᵢ)
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        individual_rewards: (n_rewards, n_particles)
    
    Returns:
        combined_grad: (n_particles, dim)
    """
    n_rewards, n_particles, dim = jacobian.shape
    
    # Product of all rewards: (n_particles,)
    reward_product = torch.prod(individual_rewards, dim=0)
    
    # For numerical stability, add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Compute Σᵢ(∇rᵢ/rᵢ): sum over reward dimension
    # individual_rewards: (n_rewards, n_particles) -> (n_rewards, n_particles, 1)
    # jacobian: (n_rewards, n_particles, dim)
    gradient_over_reward = jacobian / (individual_rewards[:, :, None] + epsilon)
    sum_grad_over_reward = torch.sum(gradient_over_reward, dim=0)  # (n_particles, dim)
    
    # Combined gradient: (r₁r₂...rₙ) * Σᵢ(∇rᵢ/rᵢ)
    combined_grad = reward_product[:, None] * sum_grad_over_reward
    
    return combined_grad


def compute_sum_of_dot_products(jacobian: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of dot products: <∇r₁, f> + <∇r₂, f> + ... + <∇rₙ, f>
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        f_x: drift term (n_particles, dim)
    
    Returns:
        sum_dots: (n_particles,)
    """
    # Compute dot product for each reward: (n_rewards, n_particles)
    dot_products = torch.sum(jacobian * f_x[None, :, :], dim=2)
    # Sum over rewards: (n_particles,)
    return torch.sum(dot_products, dim=0)


def compute_sum_of_score_interactions(jacobian: torch.Tensor, 
                                      score: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of score interactions: <∇r₁, ∇log q> + <∇r₂, ∇log q> + ... + <∇rₙ, ∇log q>
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        score: (n_particles, dim)
    
    Returns:
        sum_interactions: (n_particles,)
    """
    # Compute dot product for each reward: (n_rewards, n_particles)
    score_dots = torch.sum(jacobian * score[None, :, :], dim=2)
    # Sum over rewards: (n_particles,)
    return torch.sum(score_dots, dim=0)


def compute_adaptive_multi_reward_weights(jacobian: torch.Tensor, 
                                         individual_rewards: torch.Tensor, 
                                         f_x: torch.Tensor, 
                                         score: torch.Tensor,
                                         beta_t: float, 
                                         sigma_t: float, 
                                         dt_abs: float, 
                                         beta_dot: float) -> torch.Tensor:
    """
    Compute weight update for multi-reward FKC (excluding Term 3).
    
    Terms:
    1. β'(t) * r₁ * r₂ * ... * rₙ * dt
    2. -⟨β∇(r₁r₂...rₙ), f⟩ * dt
    4. Σᵢ⟨∇rᵢ, f⟩ * dt
    5. (σ²/2) * Σᵢ⟨∇rᵢ, ∇log q⟩ * dt
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        individual_rewards: (n_rewards, n_particles)
        f_x: drift term (n_particles, dim)
        score: (n_particles, dim)
        beta_t: reward coefficient
        sigma_t: diffusion coefficient
        dt_abs: absolute time step
        beta_dot: time derivative of beta
    
    Returns:
        dw: weight update (n_particles,)
    """
    # Term 1: Combined reward value r₁ * r₂ * ... * rₙ
    reward_product = torch.prod(individual_rewards, dim=0)  # (n_particles,)
    term1 = beta_dot * reward_product * dt_abs
    
    # Term 2: Combined reward gradient interaction -⟨β∇(r₁r₂...rₙ), f⟩
    combined_grad = compute_combined_reward_gradient(jacobian, individual_rewards)
    term2 = -torch.sum(beta_t * combined_grad * f_x, dim=1) * dt_abs
    
    # Term 4: Sum of dot products <∇r₁, f> + <∇r₂, f> + ... + <∇rₙ, f>
    term4 = compute_sum_of_dot_products(jacobian, f_x) * dt_abs
    
    # Term 5: Sum of score interactions with sigma factor
    term5 = (sigma_t**2 / 2) * compute_sum_of_score_interactions(jacobian, score) * dt_abs
    
    # Combine all terms
    dw = term1 + term2 + term4 + term5
    
    return dw


def feynman_kac_sde_step_multi_reward(x: torch.Tensor, 
                                      w: torch.Tensor, 
                                      t: float, 
                                      dt: float,
                                      reward_fns: List[Callable],
                                      reward_grad_fns: List[Callable],
                                      beta_t: float, 
                                      gamma_t: float,
                                      generator: torch.Generator | None, 
                                      network, 
                                      network_params,
                                      historical_particles: torch.Tensor = None,
                                      diversity_enabled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single FKC SDE step with multiple reward landscapes.
    
    Args:
        x: Particles (n_particles, dim)
        w: Weights (n_particles,)
        t: Current time
        dt: Time step (negative for backward)
        reward_fns: List of reward functions
        reward_grad_fns: List of reward gradient functions
        beta_t: Reward coefficient
        gamma_t: Diversity coefficient
        key: Random key
        network: Neural network dict
        network_params: Network parameters
        historical_particles: Historical particles for diversity
        diversity_enabled: Whether to use diversity terms
    
    Returns:
        x_new: Updated particles
        w_new: Updated weights
    """
    n_rewards = len(reward_fns)
    sigma_t = diffusion_coefficient(t)
    score = baseline_score_function(x, t)

    # Compute individual rewards without building autograd graph
    with torch.no_grad():
        individual_rewards_list = [reward_fn(x) for reward_fn in reward_fns]  # each: (n_particles,)
    individual_rewards = torch.stack(individual_rewards_list, dim=0)  # (n_rewards, n_particles)

    # Compute combined reward gradient via product rule in a streaming manner to avoid full Jacobian tensor
    # ∇(r₁r₂...rₙ) = (r₁r₂...rₙ) * Σᵢ(∇rᵢ / rᵢ)
    reward_product = torch.prod(individual_rewards, dim=0)  # (n_particles,)
    epsilon = 1e-8
    sum_grad_over_reward = torch.zeros_like(x)
    term4_accum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # Σ⟨∇rᵢ, f⟩
    term5_accum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # Σ⟨∇rᵢ, ∇log q⟩

    # Precompute baseline drift for weight terms
    f_x = baseline_drift(x, t)

    for i in range(n_rewards):
        grad_i = reward_grad_fns[i](x)  # (n_particles, dim)
        r_i = individual_rewards_list[i]  # (n_particles,)
        sum_grad_over_reward = sum_grad_over_reward + grad_i / (r_i[:, None] + epsilon)
        term4_accum = term4_accum + torch.sum(grad_i * f_x, dim=1)
        term5_accum = term5_accum + torch.sum(grad_i * score, dim=1)

    combined_grad = reward_product[:, None] * sum_grad_over_reward
    
    # Diversity gradient (if enabled)
    if diversity_enabled:
        from fkc import compute_diversity_loss_gradient_with_history
        diversity_grad = compute_diversity_loss_gradient_with_history(x, historical_particles)
    else:
        diversity_grad = torch.zeros_like(x)
    
    # SDE drift with combined reward gradient
    drift = (sigma_t**2 * (score + (beta_t / 2) * combined_grad)
             - baseline_drift(x, t)
             + (gamma_t * diversity_grad if diversity_enabled else 0.0))
    
    # Update particles
    sub_gen = _sub_generator(generator)
    noise = sigma_t * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=sub_gen)
    x_new = x + drift * dt + noise * torch.sqrt(torch.abs(torch.tensor(dt, device=x.device, dtype=x.dtype)))
    
    # Weight update using multi-reward formulation (streamed, no full Jacobian tensor)
    beta_dot = 1.0
    dt_abs = float(abs(dt))
    term1 = beta_dot * reward_product * dt_abs
    term2 = -torch.sum(beta_t * combined_grad * f_x, dim=1) * dt_abs
    term4 = term4_accum * dt_abs
    term5 = (sigma_t**2 / 2) * term5_accum * dt_abs
    dw = term1 + term2 + term4 + term5
    
    # Add diversity term to weights
    if diversity_enabled:
        term6 = gamma_t * torch.sum(diversity_grad * score, dim=1) * dt_abs
        dw = dw + term6
    
    # Clip and update weights
    dw = torch.clamp(dw, -1.0, 1.0)
    w_new = w + dw
    w_new = torch.clamp(w_new, -100.0, 100.0)
    
    return x_new, w_new


def run_fkc_simulation_multi_reward(particles: torch.Tensor, 
                                    weights: torch.Tensor,
                                    reward_fns: List[Callable],
                                    reward_grad_fns: List[Callable],
                                    beta_t: float, 
                                    gamma_t: float,
                                    n_steps: int, 
                                    generator: torch.Generator | None,
                                    network, 
                                    network_params,
                                    historical_particles: torch.Tensor = None,
                                    diversity_enabled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run full FKC simulation with multiple reward landscapes.
    
    Args:
        particles: Initial particles (n_particles, dim)
        weights: Initial weights (n_particles,)
        reward_fns: List of reward functions
        reward_grad_fns: List of reward gradient functions
        beta_t: Reward coefficient
        gamma_t: Diversity coefficient
        n_steps: Number of time steps
        key: Random key
        network: Neural network dict
        network_params: Network parameters
        historical_particles: Historical particles for diversity
        diversity_enabled: Whether to use diversity terms
    
    Returns:
        particles: Final particles
        weights: Final weights (normalized to [0,1])
    """
    dt = -1.0 / n_steps
    
    for i in range(n_steps):
        t = 1.0 + i * dt
        sub_gen = _sub_generator(generator)
        
        particles, weights = feynman_kac_sde_step_multi_reward(
            particles, weights, t, dt,
            reward_fns, reward_grad_fns,
            beta_t, gamma_t,
            sub_gen, network, network_params,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled
        )
        
        # Numerical stability checks
        if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
            weights = torch.where(torch.isnan(weights) | torch.isinf(weights), torch.ones_like(weights), weights)
        
        if torch.any(torch.isnan(particles)) or torch.any(torch.isinf(particles)):
            print(f"  WARNING: NaN particles detected at step {i}, clipping...")
            particles = torch.where(torch.isnan(particles) | torch.isinf(particles),
                                torch.clamp(particles, 0, 1), particles)
    
    # Normalize weights to [0, 1]
    w_min = torch.min(weights)
    w_max = torch.max(weights)
    
    if (w_max - w_min).item() > 1e-8:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = torch.ones_like(weights) * 0.5
    
    # Verify weights are in [0, 1]
    assert bool(torch.all(weights >= 0).item()) and bool(torch.all(weights <= 1).item()), "Weights must be in [0, 1]"
    
    return particles, weights

