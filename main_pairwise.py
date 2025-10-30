"""
Main FKC diffusion algorithm with pairwise preference learning.
Uses pairwise comparisons instead of direct reward observations.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Tuple, Dict, List
from reward import create_four_optima_reward_landscape, four_optima_reward_gradient
from base import baseline_sde_step
from fkc import gamma_schedule, run_fkc_simulation
from pairwise_learning import (
    create_preference_cnn, 
    train_preference_network,
    create_reward_and_gradient_functions
)
from visualization import visualize_base_model_distribution, visualize_step, visualize_gamma_and_diversity


def create_pairwise_comparisons(particles: jnp.ndarray, 
                                weights: jnp.ndarray, 
                                k_observe: int, 
                                key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select top-k particles and create k random pairs.
    Ensures no self-pairing.
    
    Args:
        particles: (n_particles, 2) all particles
        weights: (n_particles,) particle weights
        k_observe: number of pairs to create
        key: random key
    
    Returns:
        pairs: (k, 2, 2) - k pairs of 2D particles
        key: updated random key
    """
    # Get top k particles by weight
    top_k_indices = jnp.argsort(weights)[-k_observe:]
    top_k_particles = particles[top_k_indices]
    
    # Create k pairs by random sampling with no self-pairing
    pairs_list = []
    
    for i in range(k_observe):
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Sample two different indices
        idx1 = random.choice(subkey1, k_observe)
        idx2 = random.choice(subkey2, k_observe)
        
        # Ensure no self-pairing
        attempts = 0
        while idx2 == idx1 and attempts < 100:
            key, subkey2 = random.split(key)
            idx2 = random.choice(subkey2, k_observe)
            attempts += 1
        
        pairs_list.append([top_k_particles[idx1], top_k_particles[idx2]])
    
    pairs = jnp.array(pairs_list)  # (k, 2, 2)
    return pairs, key


def observe_preferences(pairs: jnp.ndarray, 
                       true_reward_fn) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Determine winner and loser for each pair based on true rewards.
    
    Args:
        pairs: (k, 2, 2) - k pairs of 2D particles
        true_reward_fn: ground truth reward function
    
    Returns:
        winners: (k, 2) winning particles
        losers: (k, 2) losing particles
    """
    # Compute true rewards for both particles in each pair
    rewards_0 = true_reward_fn(pairs[:, 0, :])  # (k,) rewards for first particle
    rewards_1 = true_reward_fn(pairs[:, 1, :])  # (k,) rewards for second particle
    
    # Determine winners: 1 if second particle wins, 0 if first wins
    winner_is_1 = (rewards_1 > rewards_0).astype(float)
    
    # Extract winners and losers based on comparison
    winners = jnp.where(winner_is_1[:, None], pairs[:, 1, :], pairs[:, 0, :])
    losers = jnp.where(winner_is_1[:, None], pairs[:, 0, :], pairs[:, 1, :])
    
    return winners, losers


def budget_constrained_diffusion_pairwise(k_observe: int, 
                                         B: int, 
                                         n_particles: int,
                                         n_steps: int, 
                                         reward_fn,
                                         network = None,
                                         network_params = None,
                                         opt_state = None):
    """
    Budget-constrained diffusion with pairwise preference learning.
    
    Main differences from standard version:
    - Uses pairwise comparisons instead of direct reward observations
    - Trains on winner/loser pairs using Bradley-Terry loss
    - Creates k pairs from top-k particles each step
    
    Args:
        k_observe: Number of pairs to create per step
        B: Total budget (number of observations)
        n_particles: Number of particles for FKC
        n_steps: Number of FKC time steps
        reward_fn: Ground truth reward function
        network: Optional pre-trained network
        network_params: Optional pre-trained parameters
        opt_state: Optional optimizer state
    
    Returns:
        Tuple of results including particles, weights, pairs, success rates, etc.
    """
    if reward_fn is None:
        raise ValueError("Must provide a reward_fn")
    
    if k_observe > n_particles:
        raise ValueError("k_observe must be <= n_particles")
    
    # Initialize
    key = random.PRNGKey(42)
    
    # Network setup - use preference CNN
    if network is None:
        network = create_preference_cnn(input_dim=2, hidden_channels=16)
        key, subkey = random.split(key)
        network_params = network['init'](subkey)
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(network_params)
        print("  Created new preference learning network")
    else:
        optimizer = optax.adam(learning_rate=0.01)
        if opt_state is None:
            opt_state = optimizer.init(network_params)
        print("  Using pre-trained preference network (continuing training)")
    
    # Storage for results
    all_particles = []
    all_weights = []
    all_pairs = []  # Store pairs instead of observed particles
    all_winners = []
    all_losers = []
    success_rates = []
    gamma_history = []
    
    # Track ALL historical winners and losers
    historical_winners_list = []
    historical_losers_list = []
    
    # Also track learned rewards at observation time for convergence checking
    historical_winner_learned_rewards_list = []
    
    # Diversity control variables
    diversity_enabled = True
    convergence_threshold = 0.03
    convergence_check_particles = 10
    
    # Estimate total steps
    total_steps = B // (k_observe * 2)  # Each pair uses 2 observations
    step_count = 0
    
    # === COLD START (Step 0) ===
    print(f"Step {step_count} [COLD START], Budget remaining: {B}")
    
    key, subkey = random.split(key)
    particles = random.uniform(subkey, (k_observe, 2))
    
    # Run baseline diffusion
    dt = -1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 + i * dt
        key, subkey = random.split(key)
        particles = baseline_sde_step(particles, t, dt, subkey)
    
    particles = jnp.clip(particles, 0, 1)
    
    # Create initial pairs from cold start particles
    # Since we don't have weights yet, create random pairs
    pairs = []
    for i in range(k_observe):
        key, subkey1, subkey2 = random.split(key, 3)
        idx1 = random.choice(subkey1, k_observe)
        idx2 = random.choice(subkey2, k_observe)
        
        # Ensure no self-pairing
        attempts = 0
        while idx2 == idx1 and attempts < 100:
            key, subkey2 = random.split(key)
            idx2 = random.choice(subkey2, k_observe)
            attempts += 1
        
        pairs.append([particles[idx1], particles[idx2]])
    
    pairs = jnp.array(pairs)  # (k, 2, 2)
    
    # Observe preferences
    winners, losers = observe_preferences(pairs, reward_fn)
    
    # Add to historical data
    historical_winners_list.append(winners)
    historical_losers_list.append(losers)
    
    # Train on all historical data
    all_historical_winners = jnp.concatenate(historical_winners_list, axis=0)
    all_historical_losers = jnp.concatenate(historical_losers_list, axis=0)
    
    print(f"  Training preference network on {len(all_historical_winners)} pairs")
    for epoch in range(50):
        network_params, opt_state, loss = train_preference_network(
            network, network_params, optimizer, opt_state,
            all_historical_winners, all_historical_losers
        )
    print(f"    Loss: {loss:.4f}")
    
    # Compute success rate based on learned rewards
    winner_rewards = network['forward'](network_params, winners)
    loser_rewards = network['forward'](network_params, losers)
    # Success: winner has reward > 0.7 OR winner > loser
    success_rate = jnp.mean((winner_rewards > loser_rewards).astype(float))
    success_rates.append(success_rate)
    
    # Store learned rewards for convergence check
    historical_winner_learned_rewards_list.append(winner_rewards)
    
    all_particles.append(particles)
    all_weights.append(jnp.ones(len(particles)))
    all_pairs.append(pairs)
    all_winners.append(winners)
    all_losers.append(losers)
    
    B -= k_observe * 2  # Each pair uses 2 observations
    step_count += 1
    
    # Visualization (using winners as "observed particles" for compatibility)
    visualize_step(step_count, particles, jnp.ones(len(particles)), winners, 
                  winner_rewards, success_rates, network, network_params, reward_fn)
    
    # === MAIN LOOP (Step 1 onwards) ===
    while B > 0:
        print(f"Step {step_count}, Budget remaining: {B}")
        
        # Check for neural network convergence
        if diversity_enabled and len(historical_winners_list) > 0:
            # Randomly sample particles from previous step
            key, subkey = random.split(key)
            check_indices = random.choice(subkey, len(historical_winners_list[-1]), 
                                       (convergence_check_particles,), replace=False)
            check_particles = historical_winners_list[-1][check_indices]
            
            # Get old and new reward estimates
            old_rewards = historical_winner_learned_rewards_list[-1][check_indices]
            new_rewards = network['forward'](network_params, check_particles)
            
            # Check convergence
            reward_change = jnp.mean(jnp.abs(new_rewards - old_rewards))
            print(f"  Convergence check: {reward_change:.4f}")
            
            if reward_change < convergence_threshold:
                diversity_enabled = False
                print(f"  *** DIVERSITY DISABLED: Converged ***")
        
        # Compute gamma for this step
        current_gamma = gamma_schedule(step_count - 1, total_steps, gamma_max=0.05, gamma_min=0.0)
        print(f"  Gamma: {current_gamma:.3f} | Diversity: {diversity_enabled}")
        
        # Concatenate all historical winners for diversity computation
        historical_particles = jnp.concatenate(historical_winners_list, axis=0)
        print(f"  Historical particles: {len(historical_particles)}")
        
        # Generate particles
        key, subkey = random.split(key)
        particles = random.uniform(subkey, (n_particles, 2))
        weights = jnp.zeros(n_particles)
        
        # Create reward and gradient functions from learned network
        reward_fn_learned, reward_grad_fn = create_reward_and_gradient_functions(
            network, network_params
        )
        
        # Run FKC with learned reward
        particles, weights = run_fkc_simulation(
            particles, weights, 
            reward_grad_fn,
            beta_t=1.0,
            gamma_t=current_gamma,
            n_steps=n_steps,
            key=subkey,
            network=network,
            network_params=network_params,
            historical_particles=historical_particles,
            diversity_enabled=diversity_enabled
        )
        
        # Store gamma
        gamma_history.append(current_gamma)
        
        # Create pairwise comparisons from top-k particles
        pairs, key = create_pairwise_comparisons(particles, weights, k_observe, key)
        
        selected_weights_indices = jnp.argsort(weights)[-k_observe:]
        selected_weights = weights[selected_weights_indices]
        print(f"    Weight range: [{jnp.min(weights):.2f}, {jnp.max(weights):.2f}] | Selected: [{jnp.min(selected_weights):.2f}, {jnp.max(selected_weights):.2f}]")
        
        # Observe preferences (ground truth)
        winners, losers = observe_preferences(pairs, reward_fn)
        
        # Add to historical data
        historical_winners_list.append(winners)
        historical_losers_list.append(losers)
        
        # Train on ALL historical preferences
        all_historical_winners = jnp.concatenate(historical_winners_list, axis=0)
        all_historical_losers = jnp.concatenate(historical_losers_list, axis=0)
        
        print(f"  Training preference network on {len(all_historical_winners)} pairs...")
        for epoch in range(50):
            network_params, opt_state, loss = train_preference_network(
                network, network_params, optimizer, opt_state,
                all_historical_winners, all_historical_losers
            )
        print(f"    Loss: {loss:.4f}")
        
        # Compute success rate based on learned rewards
        winner_rewards_learned = network['forward'](network_params, winners)
        loser_rewards_learned = network['forward'](network_params, losers)
        success_rate = jnp.mean((winner_rewards_learned > loser_rewards_learned).astype(float))
        success_rates.append(success_rate)
        
        # Store learned rewards for convergence check
        historical_winner_learned_rewards_list.append(winner_rewards_learned)
        
        # Store results
        all_particles.append(particles)
        all_weights.append(weights)
        all_pairs.append(pairs)
        all_winners.append(winners)
        all_losers.append(losers)
        
        # Update budget
        n_to_observe = min(k_observe * 2, B)  # Each pair uses 2 observations
        B -= n_to_observe
        step_count += 1
        
        # Visualize (using winners as proxy for observed particles)
        visualize_step(step_count, particles, weights, winners, 
                      winner_rewards_learned, success_rates, network, network_params, reward_fn)
    
    # Return results
    return (all_particles, all_weights, all_pairs, all_winners, all_losers,
            success_rates, network, network_params, opt_state, gamma_history)

