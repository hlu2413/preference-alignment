"""
Main FKC diffusion algorithm with budget constraints.
"""

import torch
from typing import Tuple, Dict, List
from reward import create_four_optima_reward_landscape, four_optima_reward_gradient
from base import baseline_sde_step
from fkc import gamma_schedule, run_fkc_simulation
from nn import create_cnn_reward_network, update_network, reward_network_gradient
from visualization import visualize_base_model_distribution, visualize_step, visualize_gamma_and_diversity

def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen

def _collect_param_tensors(params):
    tensors = []
    for v in params.values():
        if isinstance(v, dict):
            tensors.extend(_collect_param_tensors(v))
        else:
            if torch.is_tensor(v):
                tensors.append(v)
    return tensors

def budget_constrained_diffusion(k_observe: int, B: int, 
                                n_particles: int = 200,
                                n_steps: int = 100, 
                                reward_fn = None,
                                network = None,
                                network_params = None,
                                opt_state = None):
    """
    MODIFIED: Add diversity control mechanism based on neural network convergence.
    """
    if reward_fn is None:
        raise ValueError("Must provide a reward_fn")
    
    if k_observe > n_particles:
        raise ValueError("k_observe must be <= n_particles")
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Network setup
    if network is None:
        network = create_cnn_reward_network()
        sub_gen = _sub_generator(generator)
        network_params = network['init'](sub_gen)
        # move params to device
        for t in _collect_param_tensors(network_params):
            t.data = t.data.to(device)
            t.requires_grad_(True)
        optimizer = torch.optim.Adam(_collect_param_tensors(network_params), lr=0.01)
        opt_state = None
        print("  Created new neural network")
    else:
        if opt_state is None:
            optimizer = torch.optim.Adam(_collect_param_tensors(network_params), lr=0.01)
            opt_state = None
        print("  Using pre-trained neural network (continuing training)")
    
    # Storage for results
    all_particles = []
    all_weights = []
    all_observed_particles = []
    all_observed_rewards = []
    success_rates = []
    gamma_history = []
    
    # Track ALL historical observed particles AND their rewards
    historical_particles_list = []
    historical_rewards_list = []
    
    # NEW: Add diversity control variables
    diversity_enabled = True  # Binary variable: 1=enable diversity, 0=disable
    convergence_threshold = 0.03  # Threshold for network convergence
    convergence_check_particles = 10  # Number of random particles to check
    
    # Estimate total steps
    total_steps = B // k_observe
    step_count = 0  # <-- FIXED: Initialize step_count here
    
    # === COLD START (Step 0) ===
    print(f"Step {step_count} [COLD START], Budget remaining: {B}")
    
    sub_gen = _sub_generator(generator)
    particles = torch.rand((k_observe, 2), generator=sub_gen, device=device)
    
    dt = -1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 + i * dt
        sub_gen = _sub_generator(generator)
        particles = baseline_sde_step(particles, t, dt, sub_gen)
    
    particles = torch.clamp(particles, 0, 1)
    observed_particles = particles
    observed_rewards = reward_fn(observed_particles)
    
    # Add to historical data
    historical_particles_list.append(observed_particles)
    historical_rewards_list.append(observed_rewards)
    
    # Train on all historical data (just this batch for now)
    all_historical_particles = torch.cat(historical_particles_list, dim=0)
    all_historical_rewards = torch.cat(historical_rewards_list, dim=0)
    
    print(f"  Training CNN on {len(all_historical_particles)} observations")
    for epoch in range(50):
        network_params, opt_state, loss = update_network(
            network, network_params, optimizer, opt_state,
            all_historical_particles, all_historical_rewards
        )
    print(f"    Loss: {loss:.4f}")
    
    success_rate = torch.mean((observed_rewards > 0.7).float())
    success_rates.append(float(success_rate.item()))
    
    all_particles.append(particles)
    all_weights.append(torch.ones(len(particles), device=device))
    all_observed_particles.append(observed_particles)
    all_observed_rewards.append(observed_rewards)
    
    B -= k_observe
    step_count += 1
    
    visualize_step(
        step_count,
        particles.detach().cpu().numpy(),
        torch.ones(len(particles)).detach().cpu().numpy(),
        observed_particles.detach().cpu().numpy(),
        observed_rewards.detach().cpu().numpy(),
        [float(s.item()) if torch.is_tensor(s) else float(s) for s in success_rates],
        network, network_params, reward_fn
    )
    
    # === MAIN LOOP (Step 1 onwards) ===
    while B > 0:
        print(f"Step {step_count}, Budget remaining: {B}")
        
        # NEW: Check for neural network convergence
        if diversity_enabled and len(historical_particles_list) > 0:
            # Randomly sample particles from previous step
            sub_gen = _sub_generator(generator)
            perm = torch.randperm(len(historical_particles_list[-1]), generator=sub_gen, device=device)
            check_indices = perm[:convergence_check_particles]
            check_particles = historical_particles_list[-1][check_indices]
            
            # Get old and new reward estimates
            old_rewards = historical_rewards_list[-1][check_indices]
            new_rewards = network['forward'](network_params, check_particles)
            
            # Check convergence
            reward_change = torch.mean(torch.abs(new_rewards - old_rewards))
            print(f"  Convergence check: {float(reward_change):.4f}")
            
            if reward_change < convergence_threshold:
                diversity_enabled = False
                print(f"  *** DIVERSITY DISABLED: Converged ***")
        
        # Compute gamma for this budget step (only used if diversity_enabled=True)
        current_gamma = gamma_schedule(step_count - 1, total_steps, gamma_max=0.05, gamma_min=0.0)
        print(f"  Gamma: {current_gamma:.3f} | Diversity: {diversity_enabled}")
        
        # Concatenate all historical particles (for diversity)
        historical_particles = torch.cat(historical_particles_list, dim=0)
        print(f"  Historical particles: {len(historical_particles)}")
        
        # Generate particles
        sub_gen = _sub_generator(generator)
        particles = torch.rand((n_particles, 2), generator=sub_gen, device=device)
        weights = torch.zeros(n_particles, device=device)
        
        def reward_grad_fn(x):
            return reward_network_gradient(network, network_params, x)
        
        # Run FKC with diversity control
        particles, weights = run_fkc_simulation(
            particles, weights, 
            reward_grad_fn,
            beta_t=1.0,
            gamma_t=current_gamma,
            n_steps=n_steps,
            generator=sub_gen,
            network=network,
            network_params=network_params,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled  # <-- MODIFIED
        )
        
        # Store gamma
        gamma_history.append(current_gamma)
        
        # Selection
        sorted_indices = torch.argsort(weights)
        observe_indices = sorted_indices[-k_observe:]
        
        selected_weights = weights[observe_indices]
        print(f"    Weight range: [{float(torch.min(weights)):.2f}, {float(torch.max(weights)):.2f}] | Selected: [{float(torch.min(selected_weights)):.2f}, {float(torch.max(selected_weights)):.2f}]")
        
        # Observe true rewards
        observed_particles = particles[observe_indices]
        observed_rewards = reward_fn(observed_particles)
        
        # Add to historical data
        historical_particles_list.append(observed_particles)
        historical_rewards_list.append(observed_rewards)
        
        success_rate = torch.mean((observed_rewards > 0.7).float())
        success_rates.append(float(success_rate.item()))
        
        # CRITICAL FIX: Train on ALL historical observations, not just current batch
        all_historical_particles = torch.cat(historical_particles_list, dim=0)
        all_historical_rewards = torch.cat(historical_rewards_list, dim=0)
        
        print(f"  Training CNN on {len(all_historical_particles)} observations...")
        for epoch in range(50):
            network_params, opt_state, loss = update_network(
                network, network_params, optimizer, opt_state,
                all_historical_particles, all_historical_rewards
            )
        print(f"    Loss: {loss:.4f}")
        
        # Store results
        all_particles.append(particles)
        all_weights.append(weights)
        all_observed_particles.append(observed_particles)
        all_observed_rewards.append(observed_rewards)
        
        # Update budget
        n_to_observe = min(k_observe, B)
        B -= n_to_observe
        step_count += 1
        
        # Visualize
        visualize_step(
            step_count,
            particles.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            observed_particles.detach().cpu().numpy(),
            observed_rewards.detach().cpu().numpy(),
            [float(s.item()) if torch.is_tensor(s) else float(s) for s in success_rates],
            network, network_params, reward_fn
        )
    
    # Return results
    return (all_particles, all_weights, all_observed_particles, all_observed_rewards, 
            success_rates, network, network_params, opt_state, gamma_history)