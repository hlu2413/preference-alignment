"""
Neural network components for reward learning.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax

def create_cnn_reward_network():
    """
    Create a CNN for reward prediction that treats 2D coordinates as spatial features.
    """
    def init_fn(key):
        """Initialize CNN parameters."""
        params = {}
        
        # Feature expansion layer: 2 -> 32
        key, subkey = random.split(key)
        params['expand_W'] = random.normal(subkey, (2, 32)) * 0.1
        params['expand_b'] = jnp.zeros(32)
        
        # Conv layers (we'll simulate with dense layers on feature representations)
        key, subkey = random.split(key)
        params['conv1_W'] = random.normal(subkey, (32, 64)) * 0.1
        params['conv1_b'] = jnp.zeros(64)
        
        key, subkey = random.split(key)
        params['conv2_W'] = random.normal(subkey, (64, 64)) * 0.1
        params['conv2_b'] = jnp.zeros(64)
        
        # Fully connected layers
        key, subkey = random.split(key)
        params['fc1_W'] = random.normal(subkey, (64, 32)) * 0.1
        params['fc1_b'] = jnp.zeros(32)
        
        key, subkey = random.split(key)
        params['fc2_W'] = random.normal(subkey, (32, 1)) * 0.1
        params['fc2_b'] = jnp.zeros(1)
        
        return params
    
    def forward(params, x):
        """Forward pass through CNN."""
        # Handle both single particle and batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Feature expansion
        h = jnp.dot(x, params['expand_W']) + params['expand_b']
        h = jax.nn.relu(h)
        
        # Conv-like layer 1
        h = jnp.dot(h, params['conv1_W']) + params['conv1_b']
        h = jax.nn.relu(h)
        
        # Conv-like layer 2
        h = jnp.dot(h, params['conv2_W']) + params['conv2_b']
        h = jax.nn.relu(h)
        
        # Fully connected layer 1
        h = jnp.dot(h, params['fc1_W']) + params['fc1_b']
        h = jax.nn.relu(h)
        
        # Output layer: predict actual reward value [0,1]
        h = jnp.dot(h, params['fc2_W']) + params['fc2_b']
        h = jax.nn.sigmoid(h)  # Output in [0,1]
        
        
        # CRITICAL FIX: Ensure output is 1D for batch inputs
        if h.ndim > 1 and h.shape[-1] == 1:
            h = h.squeeze(-1)  # Remove the last dimension if it's 1
        
        if squeeze_output:
            h = h.squeeze()
        
        
        return h
    
    return {'init': init_fn, 'forward': forward}

def update_network(network, params, optimizer, opt_state, x_batch, y_batch):
    """Update neural network parameters using observed rewards."""
    def loss_fn(params):
        predictions = network['forward'](params, x_batch)
        
        # Ensure shapes match
        if predictions.shape != y_batch.shape:
            if predictions.ndim > y_batch.ndim:
                # Remove extra dimensions
                while predictions.ndim > y_batch.ndim:
                    predictions = predictions.squeeze(-1)
            elif predictions.ndim < y_batch.ndim:
                # Add missing dimensions
                while predictions.ndim < y_batch.ndim:
                    predictions = predictions.unsqueeze(-1)
        
        # MSE loss on actual reward values [0,1]
        return jnp.mean((predictions - y_batch)**2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

def reward_network_gradient(network, params, x):
    """Compute gradient of reward network using JAX autodiff with numerical stability."""
    # For vectorized inputs, we need to compute gradients for each particle
    if x.ndim == 2:  # Batch of particles
        # Use vmap to vectorize the gradient computation
        grad_fn = jax.grad(lambda single_x: network['forward'](params, single_x))
        grad = jax.vmap(grad_fn)(x)
    else:  # Single particle
        grad = jax.grad(lambda single_x: network['forward'](params, single_x))(x)
    
    # Add numerical stability: clip gradients to prevent extreme values
    grad = jnp.clip(grad, -10.0, 10.0)
    
    # Check for NaN values and replace with zeros
    grad = jnp.where(jnp.isnan(grad) | jnp.isinf(grad), 0.0, grad)
    
    return grad
