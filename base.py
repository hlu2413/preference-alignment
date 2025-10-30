"""
Baseline diffusion model components.
"""

import jax
import jax.numpy as jnp
import jax.random as random

def baseline_score_function(x: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Pretrained score function for baseline diffusion.
    MODIFIED: Make this OPPOSITE to the four optima reward to simulate worst case.
    Base model has high probability in CENTER, while rewards are at CORNERS.
    """
    # Instead of 4 peaks at corners, put a single strong peak at center (0.5, 0.5)
    # This is OPPOSITE to the 4-optima reward (corners)
    mean = jnp.array([0.5, 0.5])
    sigma = 0.15  # Narrow peak at center
    
    # Handle both single particle and batch
    if x.ndim == 1:
        score = -(x - mean) / (sigma**2)
    else:  # Batch of particles
        score = -(x - mean[None, :]) / (sigma**2)
    
    return score

def baseline_drift(x: jnp.ndarray, t: float) -> jnp.ndarray:
    """Deterministic drift term f_t(x_t) for baseline diffusion."""
    # Return zeros with the same shape as input
    return jnp.zeros_like(x)

def diffusion_coefficient(t: float) -> float:
    """Time-dependent diffusion coefficient σ_t."""
    return 0.1 * (1 - t)  # Decreases from 0.1 to 0

def baseline_sde_step(x: jnp.ndarray, t: float, dt: float, key: jnp.ndarray) -> jnp.ndarray:
    """
    Single step of baseline SDE: dx_t = (-f_t(x_t) + σ_t²∇log q_t(x_t))dt + σ_t dW_t
    """
    sigma_t = diffusion_coefficient(t)
    score = baseline_score_function(x, t)
    drift = -baseline_drift(x, t) + sigma_t**2 * score
    noise = sigma_t * random.normal(key, x.shape)
    
    # Use absolute value of dt for noise scaling since we go backwards in time
    return x + drift * dt + noise * jnp.sqrt(jnp.abs(dt))  # FIXED: use abs(dt)
