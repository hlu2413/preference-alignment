"""
Baseline diffusion model components.
"""

import torch

def baseline_score_function(x: torch.Tensor, t: float) -> torch.Tensor:
    """
    Pretrained score function for baseline diffusion.
    MODIFIED: Make this OPPOSITE to the four optima reward to simulate worst case.
    Base model has high probability in CENTER, while rewards are at CORNERS.
    """
    # Instead of 4 peaks at corners, put a single strong peak at center (0.5, 0.5)
    # This is OPPOSITE to the 4-optima reward (corners)
    mean = torch.tensor([0.5, 0.5], device=x.device, dtype=x.dtype)
    sigma = 0.15  # Narrow peak at center
    
    # Handle both single particle and batch
    if x.ndim == 1:
        score = -(x - mean) / (sigma**2)
    else:  # Batch of particles
        score = -(x - mean[None, :]) / (sigma**2)
    
    return score

def baseline_drift(x: torch.Tensor, t: float) -> torch.Tensor:
    """Deterministic drift term f_t(x_t) for baseline diffusion."""
    # Return zeros with the same shape as input
    return torch.zeros_like(x)

def diffusion_coefficient(t: float) -> float:
    """Time-dependent diffusion coefficient σ_t."""
    return 0.1 * (1 - t)  # Decreases from 0.1 to 0

def baseline_sde_step(
    x: torch.Tensor,
    t: float,
    dt: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Single step of baseline SDE: dx_t = (-f_t(x_t) + σ_t²∇log q_t(x_t))dt + σ_t dW_t
    """
    sigma_t = diffusion_coefficient(t)
    score = baseline_score_function(x, t)
    drift = -baseline_drift(x, t) + sigma_t**2 * score
    noise = sigma_t * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    
    # Use absolute value of dt for noise scaling since we go backwards in time
    return x + drift * dt + noise * torch.sqrt(torch.abs(torch.tensor(dt, device=x.device, dtype=x.dtype)))  # FIXED: use abs(dt)
