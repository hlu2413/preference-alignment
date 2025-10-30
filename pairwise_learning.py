"""
Pairwise preference learning using CNN.
Learn reward function r(x) from preference comparisons.
Train on pairs: maximize r_winner - r_loser.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Tuple, Callable, Dict


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
    
    def init_fn(key: jnp.ndarray) -> Dict:
        params = {}
        
        key, *subkeys = random.split(key, 5)
        
        # Layer 1: (2,) -> (16,)
        params['fc1_W'] = random.normal(subkeys[0], (input_dim, hidden_channels)) * 0.1
        params['fc1_b'] = jnp.zeros(hidden_channels)
        
        # Layer 2: (16,) -> (32,)
        params['fc2_W'] = random.normal(subkeys[1], (hidden_channels, hidden_channels * 2)) * 0.1
        params['fc2_b'] = jnp.zeros(hidden_channels * 2)
        
        # Layer 3: (32,) -> (16,)
        params['fc3_W'] = random.normal(subkeys[2], (hidden_channels * 2, hidden_channels)) * 0.1
        params['fc3_b'] = jnp.zeros(hidden_channels)
        
        # Output: (16,) -> (1,)
        params['out_W'] = random.normal(subkeys[3], (hidden_channels, 1)) * 0.1
        params['out_b'] = jnp.zeros(1)
        
        return params
    
    def forward(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
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
        h = jnp.dot(x, params['fc1_W']) + params['fc1_b']
        h = jax.nn.relu(h)
        
        h = jnp.dot(h, params['fc2_W']) + params['fc2_b']
        h = jax.nn.relu(h)
        
        h = jnp.dot(h, params['fc3_W']) + params['fc3_b']
        h = jax.nn.relu(h)
        
        rewards = jnp.dot(h, params['out_W']) + params['out_b']
        rewards = jax.nn.sigmoid(rewards).squeeze(-1)  # [0, 1]
        
        if is_single:
            rewards = rewards.squeeze()
        
        return rewards
    
    return {'init': init_fn, 'forward': forward}


def train_preference_network(network: Dict, 
                            params: Dict, 
                            optimizer: optax.GradientTransformation, 
                            opt_state: optax.OptState,
                            winners: jnp.ndarray, 
                            losers: jnp.ndarray) -> Tuple[Dict, optax.OptState, float]:
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
    
    def loss_fn(params: Dict) -> float:
        r_win = network['forward'](params, winners)   # (k,)
        r_loss = network['forward'](params, losers)   # (k,)
        
        # We want r_win > r_loss
        # Margin between winner and loser rewards
        margin = r_win - r_loss  # Should be positive
        
        # Log-sigmoid loss (Bradley-Terry model)
        # P(winner > loser) = sigmoid(margin)
        # Loss = -log P(winner > loser) = -log sigmoid(margin)
        loss = -jnp.mean(jax.nn.log_sigmoid(margin * 10.0))  # Scale for stability
        
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss


def compute_reward_gradient(network: Dict, 
                           params: Dict, 
                           particle: jnp.ndarray) -> jnp.ndarray:
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
    def reward_fn(x: jnp.ndarray) -> float:
        return network['forward'](params, x)
    
    return jax.grad(reward_fn)(particle)


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
    
    def reward_fn(particles: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            particles: (n_particles, 2)
        Returns:
            rewards: (n_particles,)
        """
        return network['forward'](params, particles)
    
    def reward_grad_fn(particles: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            particles: (n_particles, 2)
        Returns:
            gradients: (n_particles, 2)
        """
        # Vectorized gradient computation
        grad_fn_single = lambda p: compute_reward_gradient(network, params, p)
        gradients = jax.vmap(grad_fn_single)(particles)
        
        # Numerical stability
        gradients = jnp.clip(gradients, -10.0, 10.0)
        gradients = jnp.where(
            jnp.isnan(gradients) | jnp.isinf(gradients),
            0.0,
            gradients
        )
        
        return gradients
    
    return reward_fn, reward_grad_fn

