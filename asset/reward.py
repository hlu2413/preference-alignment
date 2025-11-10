"""
Reward landscape functions and their gradients.
"""

import torch

def create_reward_landscape(x: torch.Tensor) -> torch.Tensor:
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
    
    
    # Initialize reward tensor with correct shape
    reward = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
    # Add multiple Gaussian peaks
    peaks = [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]
    for peak_x, peak_y in peaks:
        # Calculate distance for each particle
        dist = torch.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        # Add to reward (both should have shape (batch_size,))
        reward = reward + torch.exp(-dist**2 / 0.1)
    
    # Ensure [0,1] range
    reward = torch.clamp(reward, 0, 1)
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    return reward

def reward_landscape_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of reward landscape using PyTorch autograd."""
    def grad_single(single_x: torch.Tensor) -> torch.Tensor:
        single_x = single_x.clone().detach().requires_grad_(True)
        out = create_reward_landscape(single_x)
        if out.ndim > 0:
            out = out.sum()
        grad = torch.autograd.grad(out, single_x, retain_graph=False, create_graph=False)[0]
        return grad
    if x.ndim == 1:
        return grad_single(x)
    grads = [grad_single(xi) for xi in x]
    return torch.stack(grads, dim=0)

def create_four_optima_reward_landscape(x: torch.Tensor) -> torch.Tensor:
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
    
    
    # Initialize reward tensor with correct shape
    reward = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
    # Four equal peaks in four quadrants
    peaks = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for peak_x, peak_y in peaks:
        dist = torch.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        reward = reward + torch.exp(-dist**2 / 0.05)  # Narrower peaks
    
    reward = torch.clamp(reward, 0, 1)  # Just clip to [0,1]
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    
    return reward

def four_optima_reward_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of four optima reward landscape using PyTorch autograd."""
    def grad_single(single_x: torch.Tensor) -> torch.Tensor:
        single_x = single_x.clone().detach().requires_grad_(True)
        out = create_four_optima_reward_landscape(single_x)
        if out.ndim > 0:
            out = out.sum()
        grad = torch.autograd.grad(out, single_x, retain_graph=False, create_graph=False)[0]
        return grad
    if x.ndim == 1:
        return grad_single(x)
    grads = [grad_single(xi) for xi in x]
    return torch.stack(grads, dim=0)

def create_three_mode_reward_landscape(x: torch.Tensor) -> torch.Tensor:
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
    
    # Initialize reward tensor with correct shape
    reward = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
    # Three peaks: lower left, center, upper right
    peaks = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for peak_x, peak_y in peaks:
        dist = torch.sqrt((x[:, 0] - peak_x)**2 + (x[:, 1] - peak_y)**2)
        reward = reward + torch.exp(-dist**2 / 0.03)  # Steeper peaks
    
    reward = torch.clamp(reward, 0, 1)  # Just clip to [0,1]
    
    # Reshape back to original shape if needed
    if squeeze_output and original_shape == (2,):
        reward = reward.squeeze()
    elif original_shape != x.shape:
        reward = reward.reshape(original_shape[:-1])
    
    return reward

def three_mode_reward_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of three mode reward landscape using PyTorch autograd."""
    def grad_single(single_x: torch.Tensor) -> torch.Tensor:
        single_x = single_x.clone().detach().requires_grad_(True)
        out = create_three_mode_reward_landscape(single_x)
        if out.ndim > 0:
            out = out.sum()
        grad = torch.autograd.grad(out, single_x, retain_graph=False, create_graph=False)[0]
        return grad
    if x.ndim == 1:
        return grad_single(x)
    grads = [grad_single(xi) for xi in x]
    return torch.stack(grads, dim=0)