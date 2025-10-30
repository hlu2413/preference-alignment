"""
Multi-user reward learning network.
Transformer feature extractor + Multi-head MLP architecture.
Learns rewards from positions, gradients computed via autodiff.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Tuple, List, Callable, Dict


def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, 
               eps: float = 1e-5) -> jnp.ndarray:
    """Layer normalization"""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def init_transformer_block(key: jnp.ndarray, d_model: int, n_heads: int) -> Dict:
    """Initialize parameters for a single Transformer block"""
    params = {}
    
    # Multi-head attention
    key, *subkeys = random.split(key, 5)
    params['Q_W'] = random.normal(subkeys[0], (d_model, d_model)) * 0.1
    params['K_W'] = random.normal(subkeys[1], (d_model, d_model)) * 0.1
    params['V_W'] = random.normal(subkeys[2], (d_model, d_model)) * 0.1
    params['O_W'] = random.normal(subkeys[3], (d_model, d_model)) * 0.1
    
    # Layer norm parameters
    params['norm1_scale'] = jnp.ones(d_model)
    params['norm1_bias'] = jnp.zeros(d_model)
    params['norm2_scale'] = jnp.ones(d_model)
    params['norm2_bias'] = jnp.zeros(d_model)
    
    # Feed-forward network
    key, *subkeys = random.split(key, 3)
    params['ffn_W1'] = random.normal(subkeys[0], (d_model, d_model * 4)) * 0.1
    params['ffn_b1'] = jnp.zeros(d_model * 4)
    params['ffn_W2'] = random.normal(subkeys[1], (d_model * 4, d_model)) * 0.1
    params['ffn_b2'] = jnp.zeros(d_model)
    
    return params


def transformer_block_forward(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass through Transformer block
    
    Args:
        params: Block parameters
        x: (batch_size, n_tokens, d_model)
    
    Returns:
        output: (batch_size, n_tokens, d_model)
    """
    d_model = x.shape[-1]
    
    # Multi-head self-attention
    Q = jnp.dot(x, params['Q_W'])
    K = jnp.dot(x, params['K_W'])
    V = jnp.dot(x, params['V_W'])
    
    # Scaled dot-product attention
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / jnp.sqrt(d_model)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(attn_weights, V)
    attn_out = jnp.dot(attn_out, params['O_W'])
    
    # Residual connection + layer norm
    x = x + attn_out
    x = layer_norm(x, params['norm1_scale'], params['norm1_bias'])
    
    # Feed-forward network
    ffn_out = jnp.dot(x, params['ffn_W1']) + params['ffn_b1']
    ffn_out = jax.nn.relu(ffn_out)
    ffn_out = jnp.dot(ffn_out, params['ffn_W2']) + params['ffn_b2']
    
    # Residual connection + layer norm
    x = x + ffn_out
    x = layer_norm(x, params['norm2_scale'], params['norm2_bias'])
    
    return x


def create_transformer_feature_extractor(d_model: int = 64, 
                                         n_heads: int = 4, 
                                         n_layers: int = 3) -> Dict:
    """
    Create Transformer feature extractor
    Input: (batch_size, 2) positions
    Output: (batch_size, d_model) shared features
    """
    
    def init_fn(key: jnp.ndarray) -> Dict:
        params = {}
        
        # Position embedding: (2,) -> (d_model,)
        key, subkey = random.split(key)
        params['pos_embed_W'] = random.normal(subkey, (2, d_model)) * 0.1
        params['pos_embed_b'] = jnp.zeros(d_model)
        
        # Transformer blocks
        params['transformer_blocks'] = []
        for i in range(n_layers):
            key, subkey = random.split(key)
            block_params = init_transformer_block(subkey, d_model, n_heads)
            params['transformer_blocks'].append(block_params)
        
        # Output projection
        key, subkey = random.split(key)
        params['output_W'] = random.normal(subkey, (d_model, d_model)) * 0.1
        params['output_b'] = jnp.zeros(d_model)
        
        return params
    
    def forward(params: Dict, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            particles: (batch_size, 2)
        
        Returns:
            shared_features: (batch_size, d_model)
        """
        # Embed positions
        h = jnp.dot(particles, params['pos_embed_W']) + params['pos_embed_b']
        
        # Reshape to (batch_size, 1, d_model) for Transformer
        h = h.reshape(particles.shape[0], 1, -1)
        
        # Pass through Transformer blocks
        for block_params in params['transformer_blocks']:
            h = transformer_block_forward(block_params, h)
        
        # Pool: (batch_size, 1, d_model) -> (batch_size, d_model)
        shared_features = jnp.mean(h, axis=1)
        
        # Output projection
        shared_features = jnp.dot(shared_features, params['output_W']) + params['output_b']
        
        return shared_features
    
    return {'init': init_fn, 'forward': forward}


def create_reward_head(d_shared: int = 64) -> Dict:
    """
    Create simple MLP head for single user reward prediction
    Input: (batch_size, d_shared)
    Output: (batch_size,) reward values in [0, 1]
    """
    
    def init_fn(key: jnp.ndarray) -> Dict:
        params = {}
        
        key, *subkeys = random.split(key, 4)
        
        # 3-layer MLP
        params['fc1_W'] = random.normal(subkeys[0], (d_shared, 32)) * 0.1
        params['fc1_b'] = jnp.zeros(32)
        
        params['fc2_W'] = random.normal(subkeys[1], (32, 16)) * 0.1
        params['fc2_b'] = jnp.zeros(16)
        
        params['fc3_W'] = random.normal(subkeys[2], (16, 1)) * 0.1
        params['fc3_b'] = jnp.zeros(1)
        
        return params
    
    def forward(params: Dict, shared_features: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            shared_features: (batch_size, d_shared)
        
        Returns:
            reward: (batch_size,)
        """
        h = jnp.dot(shared_features, params['fc1_W']) + params['fc1_b']
        h = jax.nn.relu(h)
        
        h = jnp.dot(h, params['fc2_W']) + params['fc2_b']
        h = jax.nn.relu(h)
        
        reward = jnp.dot(h, params['fc3_W']) + params['fc3_b']
        reward = jax.nn.sigmoid(reward).squeeze(-1)
        
        return reward
    
    return {'init': init_fn, 'forward': forward}


def create_multi_user_reward_network(n_users: int, 
                                     d_model: int = 64,
                                     n_heads: int = 4,
                                     n_layers: int = 3) -> Dict:
    """
    Create complete multi-user reward network
    Architecture: Transformer feature extractor + n MLP heads
    
    Input: (batch_size, 2) positions
    Output: (batch_size, n_users) rewards
    """
    
    def init_fn(key: jnp.ndarray) -> Dict:
        params = {}
        
        # Transformer feature extractor
        key, subkey = random.split(key)
        feature_extractor = create_transformer_feature_extractor(
            d_model, n_heads, n_layers
        )
        params['feature_extractor'] = feature_extractor['init'](subkey)
        
        # One head per user
        params['heads'] = []
        for i in range(n_users):
            key, subkey = random.split(key)
            head = create_reward_head(d_shared=d_model)
            params['heads'].append(head['init'](subkey))
        
        return params
    
    def forward(params: Dict, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            particles: (batch_size, 2)
        
        Returns:
            rewards: (batch_size, n_users)
        """
        # Extract shared features
        feature_extractor = create_transformer_feature_extractor(
            d_model, n_heads, n_layers
        )
        shared_features = feature_extractor['forward'](
            params['feature_extractor'], particles
        )
        
        # Predict reward for each user
        head_network = create_reward_head(d_model)
        rewards_list = []
        
        for i in range(n_users):
            reward_i = head_network['forward'](params['heads'][i], shared_features)
            rewards_list.append(reward_i)
        
        rewards = jnp.stack(rewards_list, axis=1)
        
        return rewards
    
    return {'init': init_fn, 'forward': forward}


def train_multi_user_network(network: Dict, 
                             params: Dict, 
                             optimizer: optax.GradientTransformation, 
                             opt_state: optax.OptState,
                             particles: jnp.ndarray, 
                             true_rewards: jnp.ndarray) -> Tuple[Dict, optax.OptState, float]:
    """
    Train network on observed data
    
    Args:
        network: Network dictionary
        params: Network parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        particles: (batch_size, 2) observed positions
        true_rewards: (batch_size, n_users) observed rewards
    
    Returns:
        new_params: Updated parameters
        new_opt_state: Updated optimizer state
        loss: Training loss
    """
    
    def loss_fn(params: Dict) -> float:
        pred_rewards = network['forward'](params, particles)
        return jnp.mean((pred_rewards - true_rewards) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss


def compute_single_user_gradient(network: Dict, 
                                 params: Dict, 
                                 particle: jnp.ndarray, 
                                 user_idx: int,
                                 n_users: int,
                                 d_model: int,
                                 n_heads: int,
                                 n_layers: int) -> jnp.ndarray:
    """
    Compute gradient of reward w.r.t. position for a single particle and user
    Uses autodiff
    
    Args:
        network: Network dictionary
        params: Network parameters
        particle: (2,) single particle position
        user_idx: User index
        n_users: Total number of users
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
    
    Returns:
        gradient: (2,) gradient vector
    """
    
    def single_reward(particle_input: jnp.ndarray) -> float:
        """Reward for single particle"""
        # Reshape to batch
        particle_batch = particle_input.reshape(1, 2)
        
        # Get shared features
        feature_extractor = create_transformer_feature_extractor(
            d_model, n_heads, n_layers
        )
        shared_features = feature_extractor['forward'](
            params['feature_extractor'], particle_batch
        )
        
        # Get reward from specific head
        head = create_reward_head(d_model)
        reward = head['forward'](params['heads'][user_idx], shared_features)
        
        return reward[0]
    
    # Compute gradient using JAX autodiff
    gradient = jax.grad(single_reward)(particle)
    
    return gradient


def create_reward_and_gradient_functions(network: Dict, 
                                        params: Dict, 
                                        n_users: int,
                                        d_model: int = 64,
                                        n_heads: int = 4,
                                        n_layers: int = 3) -> Tuple[List[Callable], List[Callable]]:
    """
    Create reward and gradient functions compatible with FKC
    
    Args:
        network: Network dictionary
        params: Network parameters
        n_users: Number of users
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
    
    Returns:
        reward_fns: List of reward functions, one per user
        reward_grad_fns: List of gradient functions, one per user
    """
    
    # Create reward functions
    reward_fns = []
    for i in range(n_users):
        def make_reward_fn(user_idx: int) -> Callable:
            def reward_fn(particles: jnp.ndarray) -> jnp.ndarray:
                """
                Args:
                    particles: (n_particles, 2)
                Returns:
                    rewards: (n_particles,)
                """
                all_rewards = network['forward'](params, particles)
                return all_rewards[:, user_idx]
            return reward_fn
        reward_fns.append(make_reward_fn(i))
    
    # Create gradient functions using autodiff
    reward_grad_fns = []
    for i in range(n_users):
        def make_grad_fn(user_idx: int) -> Callable:
            def grad_fn(particles: jnp.ndarray) -> jnp.ndarray:
                """
                Args:
                    particles: (n_particles, 2)
                Returns:
                    gradients: (n_particles, 2)
                """
                # Vectorized gradient computation
                grad_fn_single = lambda p: compute_single_user_gradient(
                    network, params, p, user_idx, n_users, d_model, n_heads, n_layers
                )
                gradients = jax.vmap(grad_fn_single)(particles)
                
                # Numerical stability
                gradients = jnp.clip(gradients, -10.0, 10.0)
                gradients = jnp.where(
                    jnp.isnan(gradients) | jnp.isinf(gradients),
                    0.0,
                    gradients
                )
                
                return gradients
            
            return grad_fn
        
        reward_grad_fns.append(make_grad_fn(i))
    
    return reward_fns, reward_grad_fns

