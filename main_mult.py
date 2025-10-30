"""
Multi-user reward learning with budget-constrained FKC diffusion.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
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
    key = random.PRNGKey(42)
    
    # Create multi-user network
    print("Initializing multi-user reward network...")
    network = create_multi_user_reward_network(
        n_users=n_users,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    key, subkey = random.split(key)
    network_params = network['init'](subkey)
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(network_params)
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
    user_convergence_status = jnp.zeros(n_users, dtype=bool)
    
    # Estimate total steps
    total_steps = B // k_observe
    step_count = 0
    
    # Cold start
    print(f"\n{'='*60}")
    print(f"Step {step_count} [COLD START], Budget remaining: {B}")
    print(f"{'='*60}")
    
    key, subkey = random.split(key)
    particles = random.uniform(subkey, (k_observe, 2))
    
    # Run baseline diffusion
    dt = -1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 + i * dt
        key, subkey = random.split(key)
        particles = baseline_sde_step(particles, t, dt, subkey)
    
    particles = jnp.clip(particles, 0, 1)
    observed_particles = particles
    
    # Observe rewards from all users
    observed_rewards_list = []
    for i in range(n_users):
        rewards_i = true_reward_fns[i](observed_particles)
        observed_rewards_list.append(rewards_i)
    observed_rewards = jnp.stack(observed_rewards_list, axis=1)
    
    # Add to historical data
    historical_particles_list.append(observed_particles)
    historical_rewards_list.append(observed_rewards)
    
    # Train on initial observations
    all_historical_particles = jnp.concatenate(historical_particles_list, axis=0)
    all_historical_rewards = jnp.concatenate(historical_rewards_list, axis=0)
    
    print(f"  Training network on {len(all_historical_particles)} observations")
    for epoch in range(50):
        network_params, opt_state, loss = train_multi_user_network(
            network, network_params, optimizer, opt_state,
            all_historical_particles, all_historical_rewards
        )
    print(f"    Final loss: {loss:.4f}")
    
    # Calculate success rates
    per_user_success = jnp.array([
        jnp.mean((observed_rewards[:, i] > 0.7).astype(float))
        for i in range(n_users)
    ])
    # Overall success: concatenate all user rewards and compute success rate
    overall_success = jnp.mean((observed_rewards > 0.7).astype(float))
    
    success_rates_per_user_history.append(per_user_success)
    success_rates_overall_history.append(float(overall_success))
    
    # Store results
    all_particles.append(particles)
    all_weights.append(jnp.ones(len(particles)))
    all_observed_particles.append(observed_particles)
    all_observed_rewards.append(observed_rewards)
    
    # Visualize
    visualize_multi_user_step(
        step_count, particles, jnp.ones(len(particles)),
        observed_particles, observed_rewards,
        success_rates_per_user_history, success_rates_overall_history,
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
            key, subkey = random.split(key)
            check_indices = random.choice(
                subkey, len(historical_particles_list[-1]),
                (convergence_check_particles,), replace=False
            )
            check_particles = historical_particles_list[-1][check_indices]
            
            old_rewards = historical_rewards_list[-1][check_indices]
            new_rewards = network['forward'](network_params, check_particles)
            
            print(f"  Convergence check:")
            for i in range(n_users):
                change_i = jnp.mean(jnp.abs(new_rewards[:, i] - old_rewards[:, i]))
                user_converged = change_i < convergence_threshold
                user_convergence_status = user_convergence_status.at[i].set(user_converged)
                status = "converged" if user_converged else "exploring"
                print(f"    User {i + 1}: {change_i:.4f} [{status}]")
            
            if jnp.all(user_convergence_status):
                diversity_enabled = False
                print(f"  *** DIVERSITY DISABLED: All users converged ***")
            else:
                n_still_exploring = jnp.sum(~user_convergence_status)
                print(f"  Diversity enabled: {n_still_exploring}/{n_users} users still exploring")
        
        # Compute gamma
        current_gamma = gamma_schedule(step_count - 1, total_steps, gamma_max, gamma_min)
        print(f"  Gamma: {current_gamma:.3f} | Diversity: {diversity_enabled}")
        
        # Concatenate historical particles
        historical_particles = jnp.concatenate(historical_particles_list, axis=0)
        print(f"  Historical particles: {len(historical_particles)}")
        
        # Generate particles
        key, subkey = random.split(key)
        particles = random.uniform(subkey, (n_particles, 2))
        weights = jnp.zeros(n_particles)
        
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
            key=subkey,
            network=None,
            network_params=None,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled
        )
        
        gamma_history.append(current_gamma)
        
        # Select top-k particles
        sorted_indices = jnp.argsort(weights)
        observe_indices = sorted_indices[-k_observe:]
        
        selected_weights = weights[observe_indices]
        print(f"    Weight range: [{jnp.min(weights):.2f}, {jnp.max(weights):.2f}]")
        print(f"    Selected weights: [{jnp.min(selected_weights):.2f}, {jnp.max(selected_weights):.2f}]")
        
        # Observe rewards from all users
        observed_particles = particles[observe_indices]
        observed_rewards_list = []
        for i in range(n_users):
            rewards_i = true_reward_fns[i](observed_particles)
            observed_rewards_list.append(rewards_i)
        observed_rewards = jnp.stack(observed_rewards_list, axis=1)
        
        # Add to historical data
        historical_particles_list.append(observed_particles)
        historical_rewards_list.append(observed_rewards)
        
        # Calculate success rates
        per_user_success = jnp.array([
            jnp.mean((observed_rewards[:, i] > 0.7).astype(float))
            for i in range(n_users)
        ])
        # Overall success: concatenate all user rewards and compute success rate
        overall_success = jnp.mean((observed_rewards > 0.7).astype(float))
        
        success_rates_per_user_history.append(per_user_success)
        success_rates_overall_history.append(float(overall_success))
        
        # Train on all historical data
        all_historical_particles = jnp.concatenate(historical_particles_list, axis=0)
        all_historical_rewards = jnp.concatenate(historical_rewards_list, axis=0)
        
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
        visualize_multi_user_step(
            step_count, particles, weights,
            observed_particles, observed_rewards,
            success_rates_per_user_history, success_rates_overall_history,
            network, network_params, true_reward_fns, n_users
        )
        
        # Update budget
        n_to_observe = min(k_observe, B)
        B -= n_to_observe
        step_count += 1
    
    # Final summary visualizations
    visualize_multi_user_gamma_and_success(
        gamma_history,
        success_rates_per_user_history,
        success_rates_overall_history,
        all_weights,
        k_observe,
        n_users
    )
    
    visualize_multi_user_final_results(
        success_rates_per_user_history,
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
