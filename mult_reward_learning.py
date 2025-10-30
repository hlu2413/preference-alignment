"""
Multi-user reward learning network.
Transformer feature extractor + Multi-head MLP architecture.
Learns rewards from positions, gradients computed via autodiff.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Callable, Dict
from torch.func import vmap, grad as func_grad


def layer_norm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, 
               eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization"""
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return scale * (x - mean) / torch.sqrt(var + eps) + bias


def init_transformer_block(generator: torch.Generator | None, d_model: int, n_heads: int) -> Dict:
    """Initialize parameters for a single Transformer block"""
    params = {}
    
    # Multi-head attention
    dev = getattr(generator, "device", torch.device("cpu")) if generator is not None else torch.device("cpu")
    params['Q_W'] = torch.randn((d_model, d_model), generator=generator, device=dev) * 0.1
    params['K_W'] = torch.randn((d_model, d_model), generator=generator, device=dev) * 0.1
    params['V_W'] = torch.randn((d_model, d_model), generator=generator, device=dev) * 0.1
    params['O_W'] = torch.randn((d_model, d_model), generator=generator, device=dev) * 0.1
    
    # Layer norm parameters
    params['norm1_scale'] = torch.ones(d_model, device=dev)
    params['norm1_bias'] = torch.zeros(d_model, device=dev)
    params['norm2_scale'] = torch.ones(d_model, device=dev)
    params['norm2_bias'] = torch.zeros(d_model, device=dev)
    
    # Feed-forward network
    params['ffn_W1'] = torch.randn((d_model, d_model * 4), generator=generator, device=dev) * 0.1
    params['ffn_b1'] = torch.zeros(d_model * 4, device=dev)
    params['ffn_W2'] = torch.randn((d_model * 4, d_model), generator=generator, device=dev) * 0.1
    params['ffn_b2'] = torch.zeros(d_model, device=dev)
    
    return params


def transformer_block_forward(params: Dict, x: torch.Tensor) -> torch.Tensor:
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
    Q = x @ params['Q_W']
    K = x @ params['K_W']
    V = x @ params['V_W']
    
    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(float(d_model)))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, V)
    attn_out = attn_out @ params['O_W']
    
    # Residual connection + layer norm
    x = x + attn_out
    x = layer_norm(x, params['norm1_scale'], params['norm1_bias'])
    
    # Feed-forward network
    ffn_out = x @ params['ffn_W1'] + params['ffn_b1']
    ffn_out = F.relu(ffn_out)
    ffn_out = ffn_out @ params['ffn_W2'] + params['ffn_b2']
    
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
    
    def init_fn(generator: torch.Generator | None) -> Dict:
        params = {}
        dev = getattr(generator, "device", torch.device("cpu")) if generator is not None else torch.device("cpu")
        
        # Position embedding: (2,) -> (d_model,)
        params['pos_embed_W'] = torch.randn((2, d_model), generator=generator, device=dev) * 0.1
        params['pos_embed_b'] = torch.zeros(d_model, device=dev)
        
        # Transformer blocks
        params['transformer_blocks'] = []
        for i in range(n_layers):
            block_params = init_transformer_block(generator, d_model, n_heads)
            params['transformer_blocks'].append(block_params)
        
        # Output projection
        params['output_W'] = torch.randn((d_model, d_model), generator=generator, device=dev) * 0.1
        params['output_b'] = torch.zeros(d_model, device=dev)
        
        return params
    
    def forward(params: Dict, particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (batch_size, 2)
        
        Returns:
            shared_features: (batch_size, d_model)
        """
        # Embed positions
        h = particles @ params['pos_embed_W'] + params['pos_embed_b']
        
        # Reshape to (batch_size, 1, d_model) for Transformer
        h = h.reshape(particles.shape[0], 1, -1)
        
        # Pass through Transformer blocks
        for block_params in params['transformer_blocks']:
            h = transformer_block_forward(block_params, h)
        
        # Pool: (batch_size, 1, d_model) -> (batch_size, d_model)
        shared_features = torch.mean(h, dim=1)
        
        # Output projection
        shared_features = shared_features @ params['output_W'] + params['output_b']
        
        return shared_features
    
    return {'init': init_fn, 'forward': forward}


def create_reward_head(d_shared: int = 64) -> Dict:
    """
    Create simple MLP head for single user reward prediction
    Input: (batch_size, d_shared)
    Output: (batch_size,) reward values in [0, 1]
    """
    
    def init_fn(generator: torch.Generator | None) -> Dict:
        params = {}
        dev = getattr(generator, "device", torch.device("cpu")) if generator is not None else torch.device("cpu")
        
        # 3-layer MLP
        params['fc1_W'] = torch.randn((d_shared, 32), generator=generator, device=dev) * 0.1
        params['fc1_b'] = torch.zeros(32, device=dev)
        
        params['fc2_W'] = torch.randn((32, 16), generator=generator, device=dev) * 0.1
        params['fc2_b'] = torch.zeros(16, device=dev)
        
        params['fc3_W'] = torch.randn((16, 1), generator=generator, device=dev) * 0.1
        params['fc3_b'] = torch.zeros(1, device=dev)
        
        return params
    
    def forward(params: Dict, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features: (batch_size, d_shared)
        
        Returns:
            reward: (batch_size,)
        """
        h = shared_features @ params['fc1_W'] + params['fc1_b']
        h = F.relu(h)
        
        h = h @ params['fc2_W'] + params['fc2_b']
        h = F.relu(h)
        
        reward = h @ params['fc3_W'] + params['fc3_b']
        reward = torch.sigmoid(reward).squeeze(-1)
        
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
    
    def init_fn(generator: torch.Generator | None) -> Dict:
        params = {}
        
        # Transformer feature extractor
        feature_extractor = create_transformer_feature_extractor(
            d_model, n_heads, n_layers
        )
        params['feature_extractor'] = feature_extractor['init'](generator)
        
        # One head per user
        params['heads'] = []
        for i in range(n_users):
            head = create_reward_head(d_shared=d_model)
            params['heads'].append(head['init'](generator))
        
        return params
    
    def forward(params: Dict, particles: torch.Tensor) -> torch.Tensor:
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
        
        rewards = torch.stack(rewards_list, dim=1)
        
        return rewards
    
    return {'init': init_fn, 'forward': forward}


def _collect_param_tensors(params: Dict) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for v in params.values():
        if isinstance(v, dict):
            tensors.extend(_collect_param_tensors(v))
        elif isinstance(v, list):
            for item in v:
                tensors.extend(_collect_param_tensors(item) if isinstance(item, dict) else ([item] if torch.is_tensor(item) else []))
        else:
            if torch.is_tensor(v):
                tensors.append(v)
    return tensors


def train_multi_user_network(network: Dict, 
                             params: Dict, 
                             optimizer: torch.optim.Optimizer, 
                             opt_state,  # kept for API parity; unused
                             particles: torch.Tensor, 
                             true_rewards: torch.Tensor) -> Tuple[Dict, None, float]:
    """
    Train network on observed data
    
    Args:
        network: Network dictionary
        params: Network parameters
        optimizer: torch optimizer (expects params' tensors with requires_grad)
        opt_state: unused (kept for compatibility)
        particles: (batch_size, 2) observed positions
        true_rewards: (batch_size, n_users) observed rewards
    
    Returns:
        new_params: Updated parameters
        new_opt_state: None (optimizer holds state internally)
        loss: Training loss
    """
    # clear stale grads before new backward
    for t in _collect_param_tensors(params):
        if t.grad is not None:
            t.grad = None

    optimizer.zero_grad(set_to_none=True)
    # ensure training data is not part of any autograd graph
    particles = particles.detach()
    true_rewards = true_rewards.detach()
    pred_rewards = network['forward'](params, particles)
    loss_tensor = torch.mean((pred_rewards - true_rewards) ** 2)
    loss_tensor.backward()
    optimizer.step()

    return params, None, float(loss_tensor.item())


def compute_single_user_gradient(network: Dict, 
                                 params: Dict, 
                                 particle: torch.Tensor, 
                                 user_idx: int,
                                 n_users: int,
                                 d_model: int,
                                 n_heads: int,
                                 n_layers: int) -> torch.Tensor:
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
    
    def single_reward(particle_input: torch.Tensor) -> torch.Tensor:
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
    
    particle = particle.clone().detach().requires_grad_(True)
    out = single_reward(particle)
    grad = torch.autograd.grad(out, particle, retain_graph=False, create_graph=False)[0]
    
    return grad


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
            def reward_fn(particles: torch.Tensor) -> torch.Tensor:
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
    
    # Create gradient functions using autodiff (chunked to reduce memory)
    reward_grad_fns = []
    for i in range(n_users):
        def make_grad_fn(user_idx: int) -> Callable:
            def grad_fn(particles: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    particles: (n_particles, 2)
                Returns:
                    gradients: (n_particles, 2)
                """
                # Compute gradients in chunks to avoid high peak memory from vmap/create_graph
                n = particles.shape[0]
                grads = torch.empty_like(particles)
                chunk_size = 256
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    x_chunk = particles[start:end].clone().detach().requires_grad_(True)
                    # Forward for this user on the chunk
                    feature_extractor = create_transformer_feature_extractor(
                        d_model, n_heads, n_layers
                    )
                    shared_features = feature_extractor['forward'](
                        params['feature_extractor'], x_chunk
                    )
                    head = create_reward_head(d_model)
                    reward_chunk = head['forward'](params['heads'][user_idx], shared_features)
                    # Sum to get scalar and take grad w.r.t. inputs
                    out = reward_chunk.sum()
                    grad_chunk = torch.autograd.grad(out, x_chunk, retain_graph=False, create_graph=False)[0]
                    grads[start:end] = grad_chunk

                grads = torch.clamp(grads, -10.0, 10.0)
                grads = torch.where(
                    torch.isnan(grads) | torch.isinf(grads),
                    torch.zeros_like(grads),
                    grads
                )
                return grads
            
            return grad_fn
        
        reward_grad_fns.append(make_grad_fn(i))
    
    return reward_fns, reward_grad_fns

