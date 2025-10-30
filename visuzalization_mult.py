import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Callable
from base import baseline_score_function

def visualize_multi_user_step(
    step: int,
    particles: jnp.ndarray,
    weights: jnp.ndarray,
    observed_particles: jnp.ndarray,
    observed_rewards: jnp.ndarray,
    success_rates_per_user_history: List[jnp.ndarray],
    success_rates_overall_history: List[float],
    network: Dict,
    network_params: Dict,
    true_reward_fns: List[Callable],
    n_users: int
):
    """
    Visualize multi-user results
    Layout: 2 rows x (n_users + 1) columns
    Row 1: Learned rewards for each user + combined
    Row 2: Actual rewards for each user + combined
    """
    fig, axes = plt.subplots(2, n_users + 1, figsize=(6 * (n_users + 1), 12))
    
    # Create grid for landscapes
    x_grid = jnp.linspace(0, 1, 100)
    y_grid = jnp.linspace(0, 1, 100)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Get learned rewards for all users
    learned_rewards_all = network['forward'](network_params, grid_points)
    
    # Get actual rewards for all users
    actual_rewards_all = jnp.stack([
        true_reward_fns[i](grid_points) for i in range(n_users)
    ], axis=1)
    
    # Row 1: Learned Rewards
    for user_idx in range(n_users):
        ax = axes[0, user_idx]
        
        # Plot learned reward landscape
        learned_r = learned_rewards_all[:, user_idx].reshape(X.shape)
        im1 = ax.contourf(X, Y, learned_r, levels=20, cmap='viridis', alpha=0.4)
        plt.colorbar(im1, ax=ax, label='Learned Reward', shrink=0.8)
        
        # Plot all particles colored by weight
        valid_mask = ~jnp.isnan(particles).any(axis=1)
        valid_particles = particles[valid_mask]
        valid_weights = weights[valid_mask]
        
        if len(valid_particles) > 0:
            scatter1 = ax.scatter(
                valid_particles[:, 0], valid_particles[:, 1],
                c=valid_weights, cmap='plasma', s=30, alpha=0.6,
                vmin=jnp.min(valid_weights), vmax=jnp.max(valid_weights)
            )
            plt.colorbar(scatter1, ax=ax, label='Weight', shrink=0.8, pad=0.1)
        
        # Circle selected particles
        if len(observed_particles) > 0:
            ax.scatter(
                observed_particles[:, 0], observed_particles[:, 1],
                s=150, facecolors='none', edgecolors='red', linewidths=2.5
            )
        
        ax.set_title(f'User {user_idx + 1}: Learned Reward', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    # Combined learned (average across users - concatenated view)
    ax = axes[0, n_users]
    combined_learned = jnp.mean(learned_rewards_all, axis=1).reshape(X.shape)
    im = ax.contourf(X, Y, combined_learned, levels=20, cmap='viridis', alpha=0.4)
    plt.colorbar(im, ax=ax, label='Avg Learned Reward', shrink=0.8)
    
    if len(valid_particles) > 0:
        scatter = ax.scatter(
            valid_particles[:, 0], valid_particles[:, 1],
            c=valid_weights, cmap='plasma', s=30, alpha=0.6,
            vmin=jnp.min(valid_weights), vmax=jnp.max(valid_weights)
        )
        plt.colorbar(scatter, ax=ax, label='Weight', shrink=0.8, pad=0.1)
    
    if len(observed_particles) > 0:
        ax.scatter(
            observed_particles[:, 0], observed_particles[:, 1],
            s=150, facecolors='none', edgecolors='red', linewidths=2.5
        )
    
    ax.set_title('Combined: Avg Learned Reward\n(All Users)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Row 2: Actual Rewards
    for user_idx in range(n_users):
        ax = axes[1, user_idx]
        
        # Plot actual reward landscape
        actual_r = actual_rewards_all[:, user_idx].reshape(X.shape)
        im2 = ax.contourf(X, Y, actual_r, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(im2, ax=ax, label='Actual Reward', shrink=0.8)
        
        # Plot observed particles colored by their actual reward
        if len(observed_particles) > 0:
            obs_rewards_i = observed_rewards[:, user_idx]
            scatter2 = ax.scatter(
                observed_particles[:, 0], observed_particles[:, 1],
                c=obs_rewards_i, cmap='RdYlGn', s=100, alpha=0.9,
                vmin=0, vmax=1, edgecolors='black', linewidths=1.5
            )
            plt.colorbar(scatter2, ax=ax, label='Observed Reward', shrink=0.8, pad=0.1)
        
        ax.set_title(f'User {user_idx + 1}: Actual Reward', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    # Combined actual (average across users - concatenated view)
    ax = axes[1, n_users]
    combined_actual = jnp.mean(actual_rewards_all, axis=1).reshape(X.shape)
    im = ax.contourf(X, Y, combined_actual, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Avg Actual Reward', shrink=0.8)
    
    # Color by average reward across all users
    if len(observed_particles) > 0:
        avg_obs_rewards = jnp.mean(observed_rewards, axis=1)
        scatter = ax.scatter(
            observed_particles[:, 0], observed_particles[:, 1],
            c=avg_obs_rewards, cmap='RdYlGn', s=100, alpha=0.9,
            vmin=0, vmax=1, edgecolors='black', linewidths=1.5
        )
        plt.colorbar(scatter, ax=ax, label='Avg Observed Reward', shrink=0.8, pad=0.1)
    
    ax.set_title('Combined: Avg Actual Reward\n(All Users)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    plt.suptitle(f'Step {step}: Multi-User Reward Learning', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n  Step {step} Statistics:")
    print(f"    Per-user success rates: {success_rates_per_user_history[-1]}")
    print(f"    Overall success rate: {success_rates_overall_history[-1]:.3f}")
    if len(observed_particles) > 0:
        mean_rewards = jnp.mean(observed_rewards, axis=0)
        print(f"    Mean rewards per user: {mean_rewards}")


def visualize_multi_user_gamma_and_success(
    gamma_history: List[float],
    success_rates_per_user_history: List[jnp.ndarray],
    success_rates_overall_history: List[float],
    all_weights: List[jnp.ndarray],
    k_observe: int,
    n_users: int
):
    """
    Visualize gamma schedule, success rate evolution, and weight distribution
    Similar to visualize_gamma_and_diversity but for multi-user case
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    steps = range(1, len(gamma_history) + 1)
    
    # Plot 1: Gamma schedule over time
    axes[0, 0].plot(steps, gamma_history, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Gamma Schedule (Diversity Coefficient)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Gamma', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, max(gamma_history) * 1.1 if max(gamma_history) > 0 else 0.1)
    
    # Plot 2: Per-user success rates
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    for user_idx in range(n_users):
        rates = [rates_array[user_idx] for rates_array in success_rates_per_user_history]
        color = colors[user_idx % len(colors)]
        axes[0, 1].plot(range(len(rates)), rates, marker='o', linewidth=2,
                       label=f'User {user_idx + 1}', color=color)
    
    axes[0, 1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
    axes[0, 1].set_title('Per-User Success Rates Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Success Rate', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Overall success rate
    axes[1, 0].plot(range(len(success_rates_overall_history)), success_rates_overall_history,
                   'b-o', linewidth=2, markersize=6, label='Overall (Concatenated)')
    axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
    axes[1, 0].set_title('Overall Success Rate\n(All User-Particle Pairs)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Success Rate', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Weight distribution evolution (box plot)
    weight_data = [w for w in all_weights[1:]]  # Skip cold start
    if len(weight_data) > 0:
        axes[1, 1].boxplot(weight_data, positions=steps)
        axes[1, 1].set_title('Weight Distribution Evolution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Step', fontsize=12)
        axes[1, 1].set_ylabel('Weight Value', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "=" * 50)
    print("GAMMA AND SUCCESS RATE SUMMARY")
    print("=" * 50)
    print(f"Steps: {len(gamma_history)} | Gamma: {min(gamma_history):.3f}→{max(gamma_history):.3f}")
    print(f"Selection: top-{k_observe} particles | Final gamma: {gamma_history[-1]:.3f}")
    print(f"Final per-user success: {success_rates_per_user_history[-1]}")
    print(f"Final overall success: {success_rates_overall_history[-1]:.3f}")
    print("=" * 50)


def visualize_multi_user_final_results(
    success_rates_per_user_history: List[jnp.ndarray],
    success_rates_overall_history: List[float],
    gamma_history: List[float],
    n_users: int
):
    """
    Final summary plots: compact 3-panel view
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    steps = range(len(success_rates_overall_history))
    
    # Plot 1: Per-user success rates
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    for user_idx in range(n_users):
        rates = [rates_array[user_idx] for rates_array in success_rates_per_user_history]
        color = colors[user_idx % len(colors)]
        axes[0].plot(steps, rates, marker='o', linewidth=2, 
                    label=f'User {user_idx + 1}', color=color)
    
    axes[0].set_title('Per-User Success Rates', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Success Rate', fontsize=12)
    axes[0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Overall success rate (concatenated)
    axes[1].plot(steps, success_rates_overall_history, 'b-o', linewidth=2, markersize=6)
    axes[1].set_title('Overall Success Rate\n(All User-Particle Pairs)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Success Rate', fontsize=12)
    axes[1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Plot 3: Gamma schedule
    if len(gamma_history) > 0:
        axes[2].plot(range(1, len(gamma_history) + 1), gamma_history, 
                    'g-o', linewidth=2, markersize=6)
        axes[2].set_title('Gamma Schedule\n(Diversity Coefficient)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Step', fontsize=12)
        axes[2].set_ylabel('Gamma', fontsize=12)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total steps: {len(success_rates_overall_history)}")
    print(f"\nPer-User Success Rates:")
    for user_idx in range(n_users):
        print(f"  User {user_idx + 1}: {success_rates_per_user_history[-1][user_idx]:.3f}")
    print(f"\nOverall (concatenated across all users): {success_rates_overall_history[-1]:.3f}")
    print("=" * 60)
