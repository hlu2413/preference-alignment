"""
Multi-user reward learning with budget-constrained FKC diffusion.
"""

import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Callable

from reward import (
    create_four_optima_reward_landscape,
    four_optima_reward_gradient,
    create_three_mode_reward_landscape,
    three_mode_reward_gradient,
    create_reward_landscape,
    reward_landscape_gradient
)
from base import baseline_sde_step
from fkc import gamma_schedule
from fkc_mult import run_fkc_simulation_multi_reward
from mult_reward_learning import (
    create_multi_user_reward_network,
    train_multi_user_network,
    create_reward_and_gradient_functions
)
from visuzalization_mult import (
    visualize_multi_user_step,
    visualize_multi_user_gamma_and_success,
    visualize_multi_user_final_results
)

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
        elif isinstance(v, list):
            for item in v:
                tensors.extend(_collect_param_tensors(item) if isinstance(item, dict) else ([item] if torch.is_tensor(item) else []))
        else:
            if torch.is_tensor(v):
                tensors.append(v)
    return tensors

def multi_user_budget_constrained_diffusion(
    k_observe: int,
    B: int,
    n_users: int,
    n_particles: int,
    n_steps: int,
    true_reward_fns: List[Callable],
    true_reward_grad_fns: List[Callable],
    d_model: int,
    n_heads: int,
    n_layers: int,
    learning_rate: float,
    convergence_threshold: float,
    convergence_check_particles: int,
    gamma_max: float,
    gamma_min: float
):
    """
    Multi-user budget-constrained diffusion with active learning
    
    Args:
        k_observe: Number of particles to observe at each step
        B: Total budget (number of observations allowed)
        n_users: Number of users with different reward functions
        n_particles: Number of particles to generate
        n_steps: Number of diffusion time steps
        true_reward_fns: List of true reward functions
        true_reward_grad_fns: List of true reward gradient functions
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
        learning_rate: Network learning rate
        convergence_threshold: Threshold for convergence check
        convergence_check_particles: Number of particles to check convergence
        gamma_max: Maximum diversity coefficient
        gamma_min: Minimum diversity coefficient
    """
    
    if k_observe > n_particles:
        raise ValueError("k_observe must be <= n_particles")
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    # Create multi-user network
    print("Initializing multi-user reward network...")
    network = create_multi_user_reward_network(
        n_users=n_users,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    sub_gen = _sub_generator(generator)
    network_params = network['init'](sub_gen)
    for t in _collect_param_tensors(network_params):
        t.data = t.data.to(device)
        t.requires_grad_(True)
    
    optimizer = torch.optim.Adam(_collect_param_tensors(network_params), lr=learning_rate)
    opt_state = None
    print(f"  Created network for {n_users} users")
    
    # Storage for results
    all_particles = []
    all_weights = []
    all_observed_particles = []
    all_observed_rewards = []
    success_rates_per_user_history = []
    success_rates_overall_history = []
    gamma_history = []
    
    # Track historical observations
    historical_particles_list = []
    historical_rewards_list = []
    
    # Diversity control
    diversity_enabled = True
    user_convergence_status = torch.zeros(n_users, dtype=torch.bool, device=device)
    
    # Estimate total steps
    total_steps = B // k_observe
    step_count = 0
    
    # Cold start
    print(f"\n{'='*60}")
    print(f"Step {step_count} [COLD START], Budget remaining: {B}")
    print(f"{'='*60}")
    
    sub_gen = _sub_generator(generator)
    particles = torch.rand((k_observe, 2), generator=sub_gen, device=device)
    
    # Run baseline diffusion
    dt = -1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 + i * dt
        sub_gen = _sub_generator(generator)
        particles = baseline_sde_step(particles, t, dt, sub_gen)
    
    particles = torch.clamp(particles, 0, 1)
    observed_particles = particles
    
    # Observe rewards from all users (no grad tracking)
    observed_rewards_list = []
    with torch.no_grad():
        for i in range(n_users):
            rewards_i = true_reward_fns[i](observed_particles)
            observed_rewards_list.append(rewards_i)
        observed_rewards = torch.stack(observed_rewards_list, dim=1)
    
    # Add to historical data
    historical_particles_list.append(observed_particles)
    historical_rewards_list.append(observed_rewards)
    
    # Train on initial observations
    all_historical_particles = torch.cat(historical_particles_list, dim=0)
    all_historical_rewards = torch.cat(historical_rewards_list, dim=0)
    
    print(f"  Training network on {len(all_historical_particles)} observations")
    for epoch in range(50):
        network_params, opt_state, loss = train_multi_user_network(
            network, network_params, optimizer, opt_state,
            all_historical_particles, all_historical_rewards
        )
    print(f"    Final loss: {loss:.4f}")
    
    # Calculate success rates (no grad tracking)
    with torch.no_grad():
        per_user_success = torch.stack([
            torch.mean((observed_rewards[:, i] > 0.7).float())
            for i in range(n_users)
        ])
        # Overall success: concatenate all user rewards and compute success rate
        overall_success = torch.mean((observed_rewards > 0.7).float())
    
    success_rates_per_user_history.append(per_user_success)
    success_rates_overall_history.append(float(overall_success))
    
    # Store results
    all_particles.append(particles)
    all_weights.append(torch.ones(len(particles), device=device))
    all_observed_particles.append(observed_particles)
    all_observed_rewards.append(observed_rewards)
    
    # Visualize
    success_rates_per_user_history_py = [
        [
            (x.detach().cpu().item() if hasattr(x, 'detach') else float(x))
            for x in step
        ]
        for step in success_rates_per_user_history
    ]
    visualize_multi_user_step(
        step_count,
        particles.detach().cpu().numpy(),
        torch.ones(len(particles)).detach().cpu().numpy(),
        observed_particles.detach().cpu().numpy(),
        observed_rewards.detach().cpu().numpy(),
        success_rates_per_user_history_py, success_rates_overall_history,
        network, network_params, true_reward_fns, n_users
    )
    
    B -= k_observe
    step_count += 1
    
    # Main loop
    while B > 0:
        print(f"\n{'='*60}")
        print(f"Step {step_count}, Budget remaining: {B}")
        print(f"{'='*60}")
        
        # Check convergence for each user
        if diversity_enabled and len(historical_particles_list) > 0:
            sub_gen = _sub_generator(generator)
            perm = torch.randperm(len(historical_particles_list[-1]), generator=sub_gen, device=device)
            check_indices = perm[:convergence_check_particles]
            check_particles = historical_particles_list[-1][check_indices]
            
            old_rewards = historical_rewards_list[-1][check_indices]
            with torch.no_grad():
                new_rewards = network['forward'](network_params, check_particles)
            
            print(f"  Convergence check:")
            for i in range(n_users):
                change_i = torch.mean(torch.abs(new_rewards[:, i] - old_rewards[:, i]))
                user_converged = (change_i < convergence_threshold)
                user_convergence_status[i] = user_converged
                status = "converged" if bool(user_converged.item()) else "exploring"
                print(f"    User {i + 1}: {float(change_i):.4f} [{status}]")
            
            if bool(torch.all(user_convergence_status).item()):
                diversity_enabled = False
                print(f"  *** DIVERSITY DISABLED: All users converged ***")
            else:
                n_still_exploring = int((~user_convergence_status).sum().item())
                print(f"  Diversity enabled: {n_still_exploring}/{n_users} users still exploring")
        
        # Compute gamma
        current_gamma = gamma_schedule(step_count - 1, total_steps, gamma_max, gamma_min)
        print(f"  Gamma: {current_gamma:.3f} | Diversity: {diversity_enabled}")
        
        # Concatenate historical particles
        historical_particles = torch.cat(historical_particles_list, dim=0)
        print(f"  Historical particles: {len(historical_particles)}")
        
        # Generate particles
        sub_gen = _sub_generator(generator)
        particles = torch.rand((n_particles, 2), generator=sub_gen, device=device)
        weights = torch.zeros(n_particles, device=device)
        
        # Create network-based reward and gradient functions
        network_reward_fns, network_grad_fns = create_reward_and_gradient_functions(
            network, network_params, n_users, d_model, n_heads, n_layers
        )
        
        # Run multi-reward FKC
        particles, weights = run_fkc_simulation_multi_reward(
            particles, weights,
            reward_fns=network_reward_fns,
            reward_grad_fns=network_grad_fns,
            beta_t=1.0,
            gamma_t=current_gamma,
            n_steps=n_steps,
            generator=sub_gen,
            network=None,
            network_params=None,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled
        )
        
        gamma_history.append(current_gamma)
        
        # Select top-k particles
        sorted_indices = torch.argsort(weights)
        observe_indices = sorted_indices[-k_observe:]
        
        selected_weights = weights[observe_indices]
        print(f"    Weight range: [{float(torch.min(weights)):.2f}, {float(torch.max(weights)):.2f}]")
        print(f"    Selected weights: [{float(torch.min(selected_weights)):.2f}, {float(torch.max(selected_weights)):.2f}]")
        
        # Observe rewards from all users (no grad tracking)
        observed_particles = particles[observe_indices]
        observed_rewards_list = []
        with torch.no_grad():
            for i in range(n_users):
                rewards_i = true_reward_fns[i](observed_particles)
                observed_rewards_list.append(rewards_i)
            observed_rewards = torch.stack(observed_rewards_list, dim=1)
        
        # Add to historical data
        historical_particles_list.append(observed_particles)
        historical_rewards_list.append(observed_rewards)
        
        # Calculate success rates (no grad tracking)
        with torch.no_grad():
            per_user_success = torch.stack([
                torch.mean((observed_rewards[:, i] > 0.7).float())
                for i in range(n_users)
            ])
            # Overall success: concatenate all user rewards and compute success rate
            overall_success = torch.mean((observed_rewards > 0.7).float())
        
        success_rates_per_user_history.append(per_user_success)
        success_rates_overall_history.append(float(overall_success))
        
        # Train on all historical data
        all_historical_particles = torch.cat(historical_particles_list, dim=0)
        all_historical_rewards = torch.cat(historical_rewards_list, dim=0)
        
        print(f"  Training network on {len(all_historical_particles)} observations...")
        for epoch in range(50):
            network_params, opt_state, loss = train_multi_user_network(
                network, network_params, optimizer, opt_state,
                all_historical_particles, all_historical_rewards
            )
        print(f"    Final loss: {loss:.4f}")
        
        # Store results
        all_particles.append(particles)
        all_weights.append(weights)
        all_observed_particles.append(observed_particles)
        all_observed_rewards.append(observed_rewards)
        
        # Visualize
        success_rates_per_user_history_py = [
            [
                (x.detach().cpu().item() if hasattr(x, 'detach') else float(x))
                for x in step
            ]
            for step in success_rates_per_user_history
        ]
        visualize_multi_user_step(
            step_count,
            particles.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            observed_particles.detach().cpu().numpy(),
            observed_rewards.detach().cpu().numpy(),
            success_rates_per_user_history_py, success_rates_overall_history,
            network, network_params, true_reward_fns, n_users
        )
        
        # Update budget
        n_to_observe = min(k_observe, B)
        B -= n_to_observe
        step_count += 1
    
    # Final summary visualizations
    success_rates_per_user_history_py = [
        [
            (x.detach().cpu().item() if hasattr(x, 'detach') else float(x))
            for x in step
        ]
        for step in success_rates_per_user_history
    ]
    visualize_multi_user_gamma_and_success(
        gamma_history,
        success_rates_per_user_history_py,
        success_rates_overall_history,
        [w.detach().cpu().numpy() for w in all_weights],
        k_observe,
        n_users
    )
    
    visualize_multi_user_final_results(
        success_rates_per_user_history_py,
        success_rates_overall_history,
        gamma_history,
        n_users
    )
    
    return (
        all_particles,
        all_weights,
        all_observed_particles,
        all_observed_rewards,
        success_rates_per_user_history,
        success_rates_overall_history,
        network,
        network_params,
        opt_state,
        gamma_history
    )
