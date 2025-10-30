"""
Reward landscape functions and their gradients.
"""

import jax
import jax.numpy as jnp

def create_reward_landscape(x: jnp.ndarray) -> jnp.ndarray:
    """
    Create a synthetic reward landscape with values in [0,1].
    This is the 'true' reward function that we're trying to learn.
    """
    # BULLETPROOF: Handle any input shape
    original_shape = x.shape
    
    # Ensure we have a 2D array (batch_size, 2)
    if x.ndim == 1:
        x = x.reshape(-1, 2)
        squeeze_output = True
    elif x.ndim == 2:
        squeeze_output = False
    else:
        # Flatten all but last dimension
        x = x.reshape(-1, x.shape[-1])
        squeeze_output = False
    
    
    # Initialize reward as JAX array with correct shape
    reward = jnp.zeros(x.shape[0])
    
    # Add multiple Gaussian peaks
    peaks = [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]
    for peak_x, peak_y in peaks:
        # Calculate distance for each particle
        dist = jnp.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        # Add to reward (both should have shape (batch_size,))
        reward = reward + jnp.exp(-dist**2 / 0.1)
    
    # Ensure [0,1] range
    reward = jnp.clip(reward, 0, 1)
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    return reward

def reward_landscape_gradient(x: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of reward landscape using JAX autodiff."""
    if x.ndim == 1:
        return jax.grad(create_reward_landscape)(x)
    else:
        # For batch of particles, use vmap
        grad_fn = jax.grad(create_reward_landscape)
        return jax.vmap(grad_fn)(x)

def create_four_optima_reward_landscape(x: jnp.ndarray) -> jnp.ndarray:
    """
    Create reward landscape with four global optima in four quadrants.
    Peaks at approximately (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    """
    # BULLETPROOF: Handle any input shape
    original_shape = x.shape
    
    # Ensure we have a 2D array (batch_size, 2)
    if x.ndim == 1:
        x = x.reshape(-1, 2)
        squeeze_output = True
    elif x.ndim == 2:
        squeeze_output = False
    else:
        # Flatten all but last dimension
        x = x.reshape(-1, x.shape[-1])
        squeeze_output = False
    
    
    # Initialize reward as JAX array with correct shape
    reward = jnp.zeros(x.shape[0])
    
    # Four equal peaks in four quadrants
    peaks = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for peak_x, peak_y in peaks:
        dist = jnp.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        reward = reward + jnp.exp(-dist**2 / 0.05)  # Narrower peaks
    
    reward = jnp.clip(reward, 0, 1)  # Just clip to [0,1]
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    
    return reward

def four_optima_reward_gradient(x: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of four optima reward landscape."""
    if x.ndim == 1:
        return jax.grad(create_four_optima_reward_landscape)(x)
    else:
        grad_fn = jax.grad(create_four_optima_reward_landscape)
        return jax.vmap(grad_fn)(x)

def create_three_mode_reward_landscape(x: jnp.ndarray) -> jnp.ndarray:
    """
    Create reward landscape with three modes:
    1. Lower left quadrant (0.25, 0.25) - coincides with lower left of four quadrants
    2. Center (0.5, 0.5) 
    3. Upper right quadrant (0.75, 0.75) - coincides with upper right of four quadrants
    """
    # BULLETPROOF: Handle any input shape
    original_shape = x.shape
    
    # Ensure we have a 2D array (batch_size, 2)
    if x.ndim == 1:
        x = x.reshape(-1, 2)
        squeeze_output = True
    elif x.ndim == 2:
        squeeze_output = False
    else:
        # Flatten all but last dimension
        x = x.reshape(-1, x.shape[-1])
        squeeze_output = False
    
    # Initialize reward as JAX array with correct shape
    reward = jnp.zeros(x.shape[0])
    
    # Three peaks: lower left, center, upper right
    peaks = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for peak_x, peak_y in peaks:
        dist = jnp.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        reward = reward + jnp.exp(-dist**2 / 0.03)  # Steeper peaks
    
    reward = jnp.clip(reward, 0, 1)  # Just clip to [0,1]
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    
    return reward

def three_mode_reward_gradient(x: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of three mode reward landscape."""
    if x.ndim == 1:
        return jax.grad(create_three_mode_reward_landscape)(x)
    else:
        grad_fn = jax.grad(create_three_mode_reward_landscape)
        return jax.vmap(grad_fn)(x)