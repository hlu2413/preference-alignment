"""
Neural network components for reward learning.
"""

import torch
import torch.nn.functional as F
from torch.func import vmap, grad as func_grad

def create_cnn_reward_network():
    """
    Create a CNN for reward prediction that treats 2D coordinates as spatial features.
    """
    def init_fn(generator: torch.Generator | None):
        """Initialize CNN parameters."""
        params = {}
        dev = getattr(generator, "device", torch.device("cpu")) if generator is not None else torch.device("cpu")
        
        # Feature expansion layer: 2 -> 32
        params['expand_W'] = torch.randn((2, 32), generator=generator, device=dev) * 0.1
        params['expand_b'] = torch.zeros(32, device=dev)
        
        # Conv layers (we'll simulate with dense layers on feature representations)
        params['conv1_W'] = torch.randn((32, 64), generator=generator, device=dev) * 0.1
        params['conv1_b'] = torch.zeros(64, device=dev)
        
        params['conv2_W'] = torch.randn((64, 64), generator=generator, device=dev) * 0.1
        params['conv2_b'] = torch.zeros(64, device=dev)
        
        # Fully connected layers
        params['fc1_W'] = torch.randn((64, 32), generator=generator, device=dev) * 0.1
        params['fc1_b'] = torch.zeros(32, device=dev)
        
        params['fc2_W'] = torch.randn((32, 1), generator=generator, device=dev) * 0.1
        params['fc2_b'] = torch.zeros(1, device=dev)
        
        return params
    
    def forward(params, x: torch.Tensor):
        """Forward pass through CNN."""
        # Handle both single particle and batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Feature expansion
        h = x @ params['expand_W'] + params['expand_b']
        h = F.relu(h)
        
        # Conv-like layer 1
        h = h @ params['conv1_W'] + params['conv1_b']
        h = F.relu(h)
        
        # Conv-like layer 2
        h = h @ params['conv2_W'] + params['conv2_b']
        h = F.relu(h)
        
        # Fully connected layer 1
        h = h @ params['fc1_W'] + params['fc1_b']
        h = F.relu(h)
        
        # Output layer: predict actual reward value [0,1]
        h = h @ params['fc2_W'] + params['fc2_b']
        h = torch.sigmoid(h)  # Output in [0,1]
        
        
        # CRITICAL FIX: Ensure output is 1D for batch inputs
        if h.ndim > 1 and h.shape[-1] == 1:
            h = h.squeeze(-1)  # Remove the last dimension if it's 1
        
        if squeeze_output:
            h = h.squeeze()
        
        
        return h
    
    return {'init': init_fn, 'forward': forward}

def _collect_param_tensors(params):
    tensors = []
    for v in params.values():
        if isinstance(v, dict):
            tensors.extend(_collect_param_tensors(v))
        else:
            if torch.is_tensor(v):
                tensors.append(v)
    return tensors

def update_network(network, params, optimizer, opt_state, x_batch, y_batch):
    """Update neural network parameters using observed rewards (PyTorch)."""
    # clear stale grads before new backward
    for t in _collect_param_tensors(params):
        if t.grad is not None:
            t.grad = None

    # ensure training data is not part of any autograd graph
    x_batch = x_batch.detach()
    y_batch = y_batch.detach()
    predictions = network['forward'](params, x_batch)
    
    # Ensure shapes match
    if predictions.shape != y_batch.shape:
        if predictions.ndim > y_batch.ndim:
            while predictions.ndim > y_batch.ndim:
                predictions = predictions.squeeze(-1)
        elif predictions.ndim < y_batch.ndim:
            while predictions.ndim < y_batch.ndim:
                predictions = predictions.unsqueeze(-1)
    
    loss_tensor = torch.mean((predictions - y_batch) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    optimizer.step()
    
    return params, None, float(loss_tensor.item())

def reward_network_gradient(network, params, x: torch.Tensor):
    """Compute gradient of reward network using PyTorch autograd with numerical stability."""
    def f(single_x: torch.Tensor) -> torch.Tensor:
        out = network['forward'](params, single_x)
        return out.sum() if out.ndim > 0 else out

    grad_f = func_grad(f)
    if x.ndim == 2:
        grads = vmap(grad_f)(x)
    else:
        grads = grad_f(x)
    grads = torch.clamp(grads, -10.0, 10.0)
    grads = torch.where(torch.isnan(grads) | torch.isinf(grads), torch.zeros_like(grads), grads)
    return grads
