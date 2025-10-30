"""
Diversity mechanisms and FKC corrector functions.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Tuple
from base import baseline_drift, diffusion_coefficient, baseline_score_function

def compute_pairwise_distance_loss(x: jnp.ndarray, particle_idx: int) -> float:
    """
    Compute diversity loss for a single particle: L(x_i) = -∑_{j≠i} ||x_i - x_j||²
    
    Args:
        x: All particles (n_particles, 2)
        particle_idx: Index of the particle to compute loss for
    
    Returns:
        loss: Scalar loss value (negative sum of squared distances)
    """
    x_i = x[particle_idx]
    # Compute squared distances to all other particles
    distances_sq = jnp.sum((x - x_i[None, :])**2, axis=1)
    # Exclude self (distance to itself is 0)
    # mask = jnp.arange(len(x)) != particle_idx
    # Return negative sum (so gradient pushes away from clusters)
    #return jnp.sum(distances_sq * mask)
    return jnp.sum(distances_sq)


def compute_diversity_loss_gradient_with_history(x: jnp.ndarray, 
                                                   historical_particles: jnp.ndarray = None) -> jnp.ndarray:
    """
    UPDATED: Vectorized diversity loss gradient computation with historical particle consideration.
    
    Diversity loss for particle i: L(x_i) = -∑_{j≠i} ||x_i - x_j||²
    where j includes both current batch particles AND all historical observed particles.
    
    Gradient: ∇L(x_i) = -2∑_{j≠i} (x_i - x_j)
    
    NORMALIZATION: Divide by total number of particles to keep magnitude comparable
    to reward/score gradients.
    
    Args:
        x: Current batch particles (n_particles, 2)
        historical_particles: All previously observed particles (n_historical, 2) or None
    
    Returns:
        gradients: (n_particles, 2) normalized gradients for each particle
    """
    n_current = x.shape[0]
    
    # Compute gradient from current batch particles
    # differences[i, j] = x[i] - x[j]
    differences_current = x[:, None, :] - x[None, :, :]
    grad_current = 2 * jnp.sum(differences_current, axis=1)  # Sum over j dimension
    
    # Add contribution from historical particles if provided
    if historical_particles is not None and len(historical_particles) > 0:
        # For each current particle, compute distance to all historical particles
        # x: (n_current, 2), historical: (n_historical, 2)
        # differences_hist[i, j] = x[i] - historical[j]
        differences_hist = x[:, None, :] - historical_particles[None, :, :]
        grad_historical = 2 * jnp.sum(differences_hist, axis=1)  # Sum over historical particles
        
        # Total gradient combines both terms
        grad = grad_current + grad_historical
        
        # Normalize by total number of particles considered
        total_particles = n_current + len(historical_particles)
        grad = grad / total_particles
    else:
        # Only current batch particles
        grad = grad_current / n_current
    
    return grad


def gamma_schedule(step: int, total_steps: int, gamma_max: float = 0.05, gamma_min: float = 0.0) -> float:
    """
    CORRECTED: Diversity coefficient decreases across BUDGET STEPS, not diffusion time.
    
    Args:
        step: Current budget step (0, 1, 2, ...)
        total_steps: Total number of budget steps
        gamma_max: Maximum gamma at step 0 (high diversity/exploration)
        gamma_min: Minimum gamma at final step (low diversity/exploitation)
    
    Returns:
        gamma_t: Diversity coefficient for this step
    """
    if total_steps <= 1:
        return gamma_max
    
    progress = step / (total_steps - 1)  # 0.0 at start → 1.0 at end
    # Linear decay: high gamma early, low gamma late
    return gamma_max * (1 - progress) + gamma_min * progress


def feynman_kac_sde_step(x: jnp.ndarray, w: jnp.ndarray, t: float, dt: float, 
                         reward_grad_fn, beta_t: float, gamma_t: float, 
                         key: jnp.ndarray, network, network_params,
                         historical_particles: jnp.ndarray = None,
                         diversity_enabled: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    MODIFIED: Add diversity_enabled parameter to control diversity terms.
    
    Args:
        diversity_enabled: Binary variable (1=enable diversity, 0=disable diversity)
    """
    sigma_t = diffusion_coefficient(t)
    score = baseline_score_function(x, t)
    reward_grad = reward_grad_fn(x)
    
    # Compute diversity loss gradient WITH HISTORICAL PARTICLES
    diversity_grad = compute_diversity_loss_gradient_with_history(x, historical_particles)
    
    # SDE drift - diversity term controlled by binary variable
    drift = (sigma_t**2 * (score + (beta_t / 2) * reward_grad) 
             - baseline_drift(x, t)
             + diversity_enabled * gamma_t * diversity_grad)  # <-- MODIFIED
        
    noise = sigma_t * random.normal(key, x.shape)
    x_new = x + drift * dt + noise * jnp.sqrt(jnp.abs(dt))
    
    # Weight equation - diversity term controlled by binary variable
    predicted_reward = network['forward'](network_params, x)
    beta_dot = 1.0
    dt_abs = jnp.abs(dt)
    
    if x.ndim == 2:  # Batch of particles
        term1 = beta_dot * predicted_reward * dt_abs
        f_t = baseline_drift(x, t)
        term2 = -jnp.sum(beta_t * reward_grad * f_t, axis=1) * dt_abs
        term3 = jnp.sum(beta_t * reward_grad * (sigma_t**2 / 2) * score, axis=1) * dt_abs
        
        # Term 6: Diversity-Score alignment controlled by BOTH gamma_t AND diversity_enabled
        term6 = diversity_enabled * gamma_t * jnp.sum(diversity_grad * score, axis=1) * dt_abs  # <-- MODIFIED
        
        dw = term1 + term2 + term3 + term6
        
    else:  # Single particle
        term1 = beta_dot * predicted_reward * dt_abs
        f_t = baseline_drift(x, t)
        term2 = -jnp.dot(beta_t * reward_grad, f_t) * dt_abs
        term3 = jnp.dot(beta_t * reward_grad, (sigma_t**2 / 2) * score) * dt_abs
        
        term6 = diversity_enabled * gamma_t * jnp.dot(diversity_grad, score) * dt_abs  # <-- MODIFIED
        
        dw = term1 + term2 + term3 + term6
    
    # Clip and update weights
    dw = jnp.clip(dw, -1.0, 1.0)
    w_new = w + dw
    w_new = jnp.clip(w_new, -100.0, 100.0)
    
    return x_new, w_new


def run_fkc_simulation(particles: jnp.ndarray, weights: jnp.ndarray, 
                      reward_grad_fn, beta_t: float, gamma_t: float,
                      n_steps: int, key: jnp.ndarray,
                      network, network_params,
                      historical_particles: jnp.ndarray = None,
                      diversity_enabled: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    MODIFIED: Add diversity_enabled parameter.
    """
    dt = -1.0 / n_steps
    
    for i in range(n_steps):
        t = 1.0 + i * dt
        key, subkey = random.split(key)
        
        particles, weights = feynman_kac_sde_step(
            particles, weights, t, dt, 
            reward_grad_fn, beta_t, gamma_t,
            subkey, network, network_params,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled  # <-- MODIFIED
        )
        
        # Numerical stability checks
        if jnp.any(jnp.isnan(weights)) or jnp.any(jnp.isinf(weights)):
            weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 1.0, weights)
        
        if jnp.any(jnp.isnan(particles)) or jnp.any(jnp.isinf(particles)):
            print(f"  WARNING: NaN particles detected at step {i}, clipping...")
            particles = jnp.where(jnp.isnan(particles) | jnp.isinf(particles), 
                                jnp.clip(particles, 0, 1), particles)
    
    # ========== NORMALIZE WEIGHTS TO [0, 1] ==========
    w_min = jnp.min(weights)
    w_max = jnp.max(weights)
    
    if w_max - w_min > 1e-8:  # Avoid division by zero
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = jnp.ones_like(weights) * 0.5
    
    # Verify weights are in [0, 1]
    assert jnp.all(weights >= 0) and jnp.all(weights <= 1), "Weights must be in [0, 1]"
    
    return particles, weights
