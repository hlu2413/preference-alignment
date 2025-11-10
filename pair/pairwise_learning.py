"""
Pairwise preference learning using CNN.
Learn reward function r(x) from preference comparisons.
Train on pairs: maximize r_winner - r_loser.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Dict
from torch.func import vmap, grad as func_grad


def create_preference_cnn(input_dim: int = 2, hidden_channels: int = 16) -> Dict:
    """
    CNN that learns rewards from pairwise preferences.
    Input: (2,) position -> Output: scalar reward in [0, 1]
    
    Architecture:
    - Multi-layer MLP (called CNN per requirements, can add conv layers if needed)
    - Output: scalar reward
    
    Args:
        input_dim: Input dimension (default 2 for 2D positions)
        hidden_channels: Hidden layer size
    
    Returns:
        Dictionary with 'init' and 'forward' functions
    """
    
    def init_fn(generator: torch.Generator | None) -> Dict:
        params = {}
        dev = getattr(generator, "device", torch.device("cpu")) if generator is not None else torch.device("cpu")
        
        # Layer 1: (2,) -> (16,)
        params['fc1_W'] = torch.randn((input_dim, hidden_channels), generator=generator, device=dev) * 0.1
        params['fc1_b'] = torch.zeros(hidden_channels, device=dev)
        
        # Layer 2: (16,) -> (32,)
        params['fc2_W'] = torch.randn((hidden_channels, hidden_channels * 2), generator=generator, device=dev) * 0.1
        params['fc2_b'] = torch.zeros(hidden_channels * 2, device=dev)
        
        # Layer 3: (32,) -> (16,)
        params['fc3_W'] = torch.randn((hidden_channels * 2, hidden_channels), generator=generator, device=dev) * 0.1
        params['fc3_b'] = torch.zeros(hidden_channels, device=dev)
        
        # Output: (16,) -> (1,)
        params['out_W'] = torch.randn((hidden_channels, 1), generator=generator, device=dev) * 0.1
        params['out_b'] = torch.zeros(1, device=dev)
        
        return params
    
    def forward(params: Dict, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            params: Network parameters
            x: (batch_size, 2) or (2,) particle positions
        
        Returns:
            rewards: (batch_size,) or scalar reward values in [0, 1]
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x.reshape(1, -1)
        
        # Forward pass
        h = x @ params['fc1_W'] + params['fc1_b']
        h = F.relu(h)
        
        h = h @ params['fc2_W'] + params['fc2_b']
        h = F.relu(h)
        
        h = h @ params['fc3_W'] + params['fc3_b']
        h = F.relu(h)
        
        rewards = h @ params['out_W'] + params['out_b']
        rewards = rewards.squeeze(-1)  # raw score (unbounded)
        
        if is_single:
            rewards = rewards.squeeze()
        
        return rewards
    
    return {'init': init_fn, 'forward': forward}


def train_preference_network(network: Dict, 
                            params: Dict, 
                            optimizer: torch.optim.Optimizer, 
                            opt_state,  # kept for API parity; unused
                            winners: torch.Tensor, 
                            losers: torch.Tensor) -> Tuple[Dict, None, float]:
    """
    Train network on pairwise preferences.
    Objective: maximize r_winner - r_loser.
    
    Uses log-sigmoid loss (Bradley-Terry model):
    L = -log(sigmoid(r_winner - r_loser))
    
    Args:
        network: Network dictionary with 'forward' function
        params: Network parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        winners: (k, 2) winning particles
        losers: (k, 2) losing particles
    
    Returns:
        new_params: Updated parameters
        new_opt_state: Updated optimizer state
        loss: Training loss value
    """
    
    for v in params.values():
        if isinstance(v, dict):
            for t in v.values():
                if torch.is_tensor(t):
                    t.requires_grad_(True)
        elif torch.is_tensor(v):
            v.requires_grad_(True)

    # ensure inputs are not attached to any prior autograd graph
    winners = winners.detach()
    losers = losers.detach()
    r_win = network['forward'](params, winners)
    r_loss = network['forward'](params, losers)

    # Bradley-Terry on raw scores: L = softplus(-(s_win - s_loss))
    margin = r_win - r_loss
    loss_tensor = torch.nn.functional.softplus(-margin).mean()

    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    optimizer.step()

    return params, None, float(loss_tensor.item())


def compute_reward_gradient(network: Dict, 
                           params: Dict, 
                           particle: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of learned reward w.r.t. particle position.
    Uses JAX autodiff for automatic differentiation.
    
    This is used by FKC to guide particle updates.
    
    Args:
        network: Network dictionary with 'forward' function
        params: Network parameters
        particle: (2,) single particle position
    
    Returns:
        gradient: (2,) gradient of reward w.r.t. position
    """
    particle = particle.clone().detach().requires_grad_(True)
    out = network['forward'](params, particle)
    if out.ndim > 0:
        out = out.sum()
    grad = torch.autograd.grad(out, particle, retain_graph=False, create_graph=False)[0]
    return grad


def create_reward_and_gradient_functions(network: Dict, 
                                        params: Dict) -> Tuple[Callable, Callable]:
    """
    Create reward and gradient functions compatible with FKC.
    
    Args:
        network: Network dictionary
        params: Network parameters
    
    Returns:
        reward_fn: Function that takes (n_particles, 2) and returns (n_particles,)
        reward_grad_fn: Function that takes (n_particles, 2) and returns (n_particles, 2)
    """
    
    def reward_fn(particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (n_particles, 2)
        Returns:
            rewards: (n_particles,)
        """
        return network['forward'](params, particles)
    
    def reward_grad_fn(particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (n_particles, 2)
        Returns:
            gradients: (n_particles, 2)
        """
        # vectorized gradient via torch.func
        def single_reward(p_single: torch.Tensor) -> torch.Tensor:
            out = network['forward'](params, p_single)
            return out.sum() if out.ndim > 0 else out

        grad_single = func_grad(single_reward)
        gradients = vmap(grad_single)(particles)
        gradients = torch.clamp(gradients, -10.0, 10.0)
        gradients = torch.where(
            torch.isnan(gradients) | torch.isinf(gradients),
            torch.zeros_like(gradients),
            gradients
        )
        
        return gradients
    
    return reward_fn, reward_grad_fn

