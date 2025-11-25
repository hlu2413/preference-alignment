import os
import random
from datetime import datetime
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage
from typing import Dict, Tuple, Optional, List

from imagegen_r.sd15_pipeline import SD15LatentModel
from imagegen_r.openclip_proxy import OpenCLIPPreferenceProxy
from imagegen_r.surrogate import create_surrogate_and_grad
from imagegen_r.fkc import run_fkc_simulation_image, gamma_schedule, beta_schedule
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weight_visualization import visualize_weight_distribution


def _sub_generator(generator: torch.Generator | None) -> torch.Generator | None:
    if generator is None:
        return None
    device = getattr(generator, "device", "cpu")
    new_gen = torch.Generator(device=device)
    seed_tensor = torch.randint(0, 2**31 - 1, (1,), device=device, generator=generator)
    new_gen.manual_seed(int(seed_tensor.item()))
    return new_gen


def run_experiment(
    k_observe: int,
    B: int,
    n_particles: int,
    n_steps: int,
    latent_shape: Tuple[int, int, int] = (4, 64, 64),
    device: Optional[torch.device] = None,
    temperature: float = 10.0,
    fruit_users: Optional[Dict[str, List[str]]] = None,
    prompt_for_seed: str = "a photo of food",
    output_root: str = "imagegen_r/results",
    seed: Optional[int] = None
) -> Dict:
    if device is None:
        device = torch.device('cuda')

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    sd = SD15LatentModel(device=device, num_inference_steps=n_steps)
    score_fn = sd.get_score_function(prompt_for_seed)
    sd.pipe.scheduler.set_timesteps(n_steps, device=device)
    sd.sigmas = sd.pipe.scheduler.sigmas.to(device)

    clip_proxy = OpenCLIPPreferenceProxy(device=device)
    if fruit_users is None:
        fruit_users = {
            'user_fruit': ["a photo of a fruit", "a photo of fruits", "fruit", "fruits"],
        }
    user_prompt_bank = clip_proxy.build_user_prompt_bank(fruit_users)

    learned_reward_model, reward_grad_fn = create_surrogate_and_grad(latent_shape, device)
    optimizer = optim.Adam(learned_reward_model.parameters(), lr=1e-3)

    # Prepare output directory: imagegen/results/<datetime>
    run_dir = os.path.join(output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    historical_latents: List[torch.Tensor] = []
    historical_scores: List[torch.Tensor] = []
    success_rates: List[float] = []
    gamma_history: List[float] = []
    beta_history: List[float] = []
    step_count = 0

    sub_gen = _sub_generator(generator)
    # Use prompt-conditioned SD15 generation to obtain clean initial latents
    z0 = sd.generate_latents_from_prompt(prompt_for_seed, batch_size=k_observe, generator=sub_gen)
    with torch.no_grad():
        images0 = sd.decode_latents(z0).cpu()
    user_scores0 = clip_proxy.score_images(images0, user_prompt_bank, temperature)
    union0 = torch.stack(list(user_scores0.values()), dim=1).max(dim=1).values

    historical_latents.append(z0.detach().cpu())
    historical_scores.append(union0.detach().cpu())

    for _ in range(200):
        optimizer.zero_grad()
        preds = learned_reward_model(z0.detach())
        loss = torch.mean((preds - union0.detach()) ** 2)
        loss.backward()
        optimizer.step()

    success_rates.append(float((union0 > 0.7).float().mean().item()))

    del union0
    del z0

    # Save step 1 images into run_dir/step_001
    step_dir = os.path.join(run_dir, f"step_{step_count + 1:03d}")
    os.makedirs(step_dir, exist_ok=True)
    to_pil = ToPILImage()
    for i, img in enumerate(images0):
        to_pil(img.cpu()).save(os.path.join(step_dir, f"selected_{i:03d}.png"))
    # Save a grid for quick view
    grid = vutils.make_grid(images0, nrow=min(8, images0.shape[0]))
    vutils.save_image(grid, os.path.join(step_dir, "grid.png"))
    del images0

    B -= k_observe
    step_count += 1
    total_steps = B // k_observe
    diversity_enabled = True
    fkc_steps = 5

    while B > 0:
        remaining_steps = total_steps - (step_count - 1)
        use_fkc = remaining_steps <= fkc_steps
        
        step_dir = os.path.join(run_dir, f"step_{step_count + 1:03d}")
        os.makedirs(step_dir, exist_ok=True)
        
        if use_fkc:
            fkc_step_idx = fkc_steps - remaining_steps
            current_gamma = gamma_schedule(fkc_step_idx, fkc_steps, gamma_max=0.05, gamma_min=0.0)
            current_beta = beta_schedule(fkc_step_idx, fkc_steps, beta_min=0.5, beta_max=2.0)
        hist_latents_gpu = torch.cat([lat.to(device) for lat in historical_latents], dim=0)

        sub_gen = _sub_generator(generator)
        z = sd.sample_latents((n_particles,) + latent_shape, sub_gen)
        w = torch.zeros(n_particles, device=device)

        def grad_fn(z_batch: torch.Tensor) -> torch.Tensor:
            z_batch = z_batch.clone().requires_grad_(True)
            reward = learned_reward_model(z_batch)
            reward_sum = reward.sum() if reward.ndim > 0 else reward
            return torch.autograd.grad(reward_sum, z_batch, retain_graph=False, create_graph=False)[0]

        def reward_fn(z_batch: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return learned_reward_model(z_batch)

        z, w = run_fkc_simulation_image(
            z, w, grad_fn, reward_fn, beta_t=current_beta, gamma_t=current_gamma,
            n_steps=n_steps, sd_model=sd, generator=sub_gen, score_fn=score_fn,
            historical_particles=hist_latents_gpu, diversity_enabled=diversity_enabled
        )
        del hist_latents_gpu

        gamma_history.append(current_gamma)
        beta_history.append(current_beta)

        sorted_idx = torch.argsort(w)
        observe_idx = sorted_idx[-k_observe:]
        selected_z = z[observe_idx]
            
            visualize_weight_distribution(w, step_dir, step_count + 1, k_observe)
        else:
            sub_gen = _sub_generator(generator)
            selected_z = sd.generate_latents_from_prompt(prompt_for_seed, batch_size=k_observe, generator=sub_gen)
            gamma_history.append(0.0)
            beta_history.append(0.0)

        with torch.no_grad():
            selected_images = sd.decode_latents(selected_z).cpu()
        user_scores = clip_proxy.score_images(selected_images, user_prompt_bank, temperature)
        union_scores = torch.stack(list(user_scores.values()), dim=1).max(dim=1).values

        historical_latents.append(selected_z.detach().cpu())
        historical_scores.append(union_scores.detach().cpu())
        del selected_z

        all_hist_z = torch.cat([lat.to(device) for lat in historical_latents], dim=0)
        all_hist_scores = torch.cat(historical_scores, dim=0).to(device)

        for _ in range(200):
            optimizer.zero_grad()
            preds = learned_reward_model(all_hist_z)
            loss = torch.mean((preds - all_hist_scores) ** 2)
            loss.backward()
            optimizer.step()
        del all_hist_z
        del all_hist_scores

        success_rates.append(float((union_scores > 0.7).float().mean().item()))
        del union_scores

        # Save per-step images
        for i, img in enumerate(selected_images):
            to_pil(img.cpu()).save(os.path.join(step_dir, f"selected_{i:03d}.png"))
        
        # Save grid - for FKC steps, show all candidates; for base model steps, show selected only
        if use_fkc:
        with torch.no_grad():
            all_images = sd.decode_latents(z).cpu()
        grid = vutils.make_grid(all_images, nrow=min(8, all_images.shape[0]))
        vutils.save_image(grid, os.path.join(step_dir, "grid.png"))
        del all_images
        del z
        else:
            grid = vutils.make_grid(selected_images, nrow=min(8, selected_images.shape[0]))
            vutils.save_image(grid, os.path.join(step_dir, "grid.png"))
        
        del selected_images

        B -= k_observe
        step_count += 1

    # plotting removed for minimal core functionality

    return {
        'success_rates': success_rates,
        'gamma_history': gamma_history,
        'beta_history': beta_history,
        'learned_reward_model': learned_reward_model,
        'output_dir': run_dir,
    }


if __name__ == "__main__":
    device = torch.device('cuda')
    run_experiment(
        k_observe=8,
        B=80,
        n_particles=32,
        n_steps=25,
        latent_shape=(4, 64, 64),
        device=device,
        temperature=10.0,
        prompt_for_seed="a photo of food",
        seed=None
    )

