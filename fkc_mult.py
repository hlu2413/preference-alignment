"""
Multi-reward FKC formulation using Jacobian approach.
Handles arbitrary number of reward landscapes with adaptive weight updates.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Tuple, Callable, List
from base import baseline_drift, diffusion_coefficient, baseline_score_function


def compute_jacobian(reward_grad_fns: List[Callable], x: jnp.ndarray) -> jnp.ndarray:
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
    return jnp.stack(gradients, axis=0)


def compute_combined_reward_gradient(jacobian: jnp.ndarray, 
                                    individual_rewards: jnp.ndarray) -> jnp.ndarray:
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
    reward_product = jnp.prod(individual_rewards, axis=0)
    
    # For numerical stability, add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Compute Σᵢ(∇rᵢ/rᵢ): sum over reward dimension
    # individual_rewards: (n_rewards, n_particles) -> (n_rewards, n_particles, 1)
    # jacobian: (n_rewards, n_particles, dim)
    gradient_over_reward = jacobian / (individual_rewards[:, :, None] + epsilon)
    sum_grad_over_reward = jnp.sum(gradient_over_reward, axis=0)  # (n_particles, dim)
    
    # Combined gradient: (r₁r₂...rₙ) * Σᵢ(∇rᵢ/rᵢ)
    combined_grad = reward_product[:, None] * sum_grad_over_reward
    
    return combined_grad


def compute_sum_of_dot_products(jacobian: jnp.ndarray, f_x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute sum of dot products: <∇r₁, f> + <∇r₂, f> + ... + <∇rₙ, f>
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        f_x: drift term (n_particles, dim)
    
    Returns:
        sum_dots: (n_particles,)
    """
    # Compute dot product for each reward: (n_rewards, n_particles)
    dot_products = jnp.sum(jacobian * f_x[None, :, :], axis=2)
    # Sum over rewards: (n_particles,)
    return jnp.sum(dot_products, axis=0)


def compute_sum_of_score_interactions(jacobian: jnp.ndarray, 
                                      score: jnp.ndarray) -> jnp.ndarray:
    """
    Compute sum of score interactions: <∇r₁, ∇log q> + <∇r₂, ∇log q> + ... + <∇rₙ, ∇log q>
    
    Args:
        jacobian: (n_rewards, n_particles, dim)
        score: (n_particles, dim)
    
    Returns:
        sum_interactions: (n_particles,)
    """
    # Compute dot product for each reward: (n_rewards, n_particles)
    score_dots = jnp.sum(jacobian * score[None, :, :], axis=2)
    # Sum over rewards: (n_particles,)
    return jnp.sum(score_dots, axis=0)


def compute_adaptive_multi_reward_weights(jacobian: jnp.ndarray, 
                                         individual_rewards: jnp.ndarray, 
                                         f_x: jnp.ndarray, 
                                         score: jnp.ndarray,
                                         beta_t: float, 
                                         sigma_t: float, 
                                         dt_abs: float, 
                                         beta_dot: float) -> jnp.ndarray:
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
    reward_product = jnp.prod(individual_rewards, axis=0)  # (n_particles,)
    term1 = beta_dot * reward_product * dt_abs
    
    # Term 2: Combined reward gradient interaction -⟨β∇(r₁r₂...rₙ), f⟩
    combined_grad = compute_combined_reward_gradient(jacobian, individual_rewards)
    term2 = -jnp.sum(beta_t * combined_grad * f_x, axis=1) * dt_abs
    
    # Term 4: Sum of dot products <∇r₁, f> + <∇r₂, f> + ... + <∇rₙ, f>
    term4 = compute_sum_of_dot_products(jacobian, f_x) * dt_abs
    
    # Term 5: Sum of score interactions with sigma factor
    term5 = (sigma_t**2 / 2) * compute_sum_of_score_interactions(jacobian, score) * dt_abs
    
    # Combine all terms
    dw = term1 + term2 + term4 + term5
    
    return dw


def feynman_kac_sde_step_multi_reward(x: jnp.ndarray, 
                                      w: jnp.ndarray, 
                                      t: float, 
                                      dt: float,
                                      reward_fns: List[Callable],
                                      reward_grad_fns: List[Callable],
                                      beta_t: float, 
                                      gamma_t: float,
                                      key: jnp.ndarray, 
                                      network, 
                                      network_params,
                                      historical_particles: jnp.ndarray = None,
                                      diversity_enabled: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    
    # Compute Jacobian: (n_rewards, n_particles, dim)
    jacobian = compute_jacobian(reward_grad_fns, x)
    
    # Compute individual rewards: (n_rewards, n_particles)
    individual_rewards = jnp.stack([reward_fn(x) for reward_fn in reward_fns], axis=0)
    
    # Combined reward gradient (average or use product rule)
    # For SDE drift, we use the combined gradient
    combined_grad = compute_combined_reward_gradient(jacobian, individual_rewards)
    
    # Diversity gradient (if enabled)
    if diversity_enabled:
        from fkc import compute_diversity_loss_gradient_with_history
        diversity_grad = compute_diversity_loss_gradient_with_history(x, historical_particles)
    else:
        diversity_grad = jnp.zeros_like(x)
    
    # SDE drift with combined reward gradient
    drift = (sigma_t**2 * (score + (beta_t / 2) * combined_grad)
             - baseline_drift(x, t)
             + diversity_enabled * gamma_t * diversity_grad)
    
    # Update particles
    noise = sigma_t * random.normal(key, x.shape)
    x_new = x + drift * dt + noise * jnp.sqrt(jnp.abs(dt))
    
    # Weight update using multi-reward formulation
    beta_dot = 1.0
    dt_abs = jnp.abs(dt)
    f_t = baseline_drift(x, t)
    
    # Compute weight update using adaptive multi-reward weights
    dw = compute_adaptive_multi_reward_weights(
        jacobian, individual_rewards, f_t, score,
        beta_t, sigma_t, dt_abs, beta_dot
    )
    
    # Add diversity term to weights
    if diversity_enabled:
        term6 = gamma_t * jnp.sum(diversity_grad * score, axis=1) * dt_abs
        dw = dw + term6
    
    # Clip and update weights
    dw = jnp.clip(dw, -1.0, 1.0)
    w_new = w + dw
    w_new = jnp.clip(w_new, -100.0, 100.0)
    
    return x_new, w_new


def run_fkc_simulation_multi_reward(particles: jnp.ndarray, 
                                    weights: jnp.ndarray,
                                    reward_fns: List[Callable],
                                    reward_grad_fns: List[Callable],
                                    beta_t: float, 
                                    gamma_t: float,
                                    n_steps: int, 
                                    key: jnp.ndarray,
                                    network, 
                                    network_params,
                                    historical_particles: jnp.ndarray = None,
                                    diversity_enabled: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        key, subkey = random.split(key)
        
        particles, weights = feynman_kac_sde_step_multi_reward(
            particles, weights, t, dt,
            reward_fns, reward_grad_fns,
            beta_t, gamma_t,
            subkey, network, network_params,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled
        )
        
        # Numerical stability checks
        if jnp.any(jnp.isnan(weights)) or jnp.any(jnp.isinf(weights)):
            weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 1.0, weights)
        
        if jnp.any(jnp.isnan(particles)) or jnp.any(jnp.isinf(particles)):
            print(f"  WARNING: NaN particles detected at step {i}, clipping...")
            particles = jnp.where(jnp.isnan(particles) | jnp.isinf(particles),
                                jnp.clip(particles, 0, 1), particles)
    
    # Normalize weights to [0, 1]
    w_min = jnp.min(weights)
    w_max = jnp.max(weights)
    
    if w_max - w_min > 1e-8:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = jnp.ones_like(weights) * 0.5
    
    # Verify weights are in [0, 1]
    assert jnp.all(weights >= 0) and jnp.all(weights <= 1), "Weights must be in [0, 1]"
    
    return particles, weights

