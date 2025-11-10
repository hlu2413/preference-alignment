import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from typing import Tuple, List
from torchvision import transforms as T


VAE_SCALE_FACTOR = 0.18215


class SD15LatentModel:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: torch.device | None = None, num_inference_steps: int = 50):
        if device is None:
            device = torch.device('cuda')
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        self.pipe = self.pipe.to(device)
        self.unet_dtype = next(self.pipe.unet.parameters()).dtype
        self.pipe.safety_checker = None
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # set a default schedule and cache timesteps/sigmas for score queries
        self.num_inference_steps = int(num_inference_steps)
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.sigmas = self.pipe.scheduler.sigmas.to(self.device)

    def sample_latents(self, batch_shape: Tuple[int, int, int, int], generator: torch.Generator | None) -> torch.Tensor:
        b, c, h, w = batch_shape
        init_sigma = self.pipe.scheduler.init_noise_sigma
        latents = torch.randn((b, c, h, w), generator=generator, device=self.device)
        return latents * init_sigma

    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        images = images.to(device=self.device, dtype=vae_dtype)
        images = (images * 2.0) - 1.0
        posterior = self.pipe.vae.encode(images).latent_dist
        latents = posterior.sample() * VAE_SCALE_FACTOR
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        images = self.pipe.vae.decode(latents / VAE_SCALE_FACTOR).sample
        images = (images + 1.0) / 2.0
        return torch.clamp(images, 0.0, 1.0)

    # --- New: prompt helpers and UNet-based score ---
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.pipe.tokenizer(
            [prompt],
            padding='max_length',
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            text_embeds = self.pipe.text_encoder(tokens.input_ids.to(self.device))[0]
        return text_embeds.to(device=self.device, dtype=self.unet_dtype)

    def _t_to_index(self, t: float) -> int:
        # map t in [0,1] (1=noisiest, 0=clean) to scheduler index
        n = len(self.pipe.scheduler.timesteps) - 1 if len(self.pipe.scheduler.timesteps) > 1 else 1
        idx = int(round((1.0 - float(t)) * n))
        return max(0, min(n, idx))

    def _score_unet(self, z: torch.Tensor, t: float, prompt_embeds: torch.Tensor) -> torch.Tensor:
        idx = self._t_to_index(t)
        timesteps = self.pipe.scheduler.timesteps.to(self.device)
        t_step = timesteps[idx]
        # scale input as in generation
        latent_model_input = self.pipe.scheduler.scale_model_input(
            z.to(device=self.device, dtype=self.unet_dtype), t_step
        )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.unet_dtype)
        if prompt_embeds.shape[0] != latent_model_input.shape[0]:
            prompt_embeds = prompt_embeds.repeat(latent_model_input.shape[0], 1, 1)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                latent_model_input, t_step, encoder_hidden_states=prompt_embeds
            ).sample
        # convert to score ∇_z log p_t(z) ≈ -ε/σ_t
        sigma_t = self.sigmas[idx] if idx < len(self.sigmas) else self.sigmas[-1]
        # prevent div-by-zero when near t=0
        sigma_t = torch.clamp(sigma_t, min=1e-5)
        return -noise_pred / sigma_t

    def get_score_function(self, prompt: str | None = None):
        prompt = prompt or ""
        prompt_embeds = self._encode_prompt(prompt)
        def score_fn(z: torch.Tensor, t: float) -> torch.Tensor:
            return self._score_unet(z, t, prompt_embeds)
        return score_fn

    def generate_latents_from_prompt(self, prompt: str, batch_size: int, generator: torch.Generator | None = None, num_inference_steps: int | None = None) -> torch.Tensor:
        steps = int(num_inference_steps) if num_inference_steps is not None else self.num_inference_steps
        prompts: List[str] = [prompt] * batch_size
        with torch.no_grad():
            result = self.pipe(prompts, num_inference_steps=steps, generator=generator)
            pil_images = result.images
        to_tensor = T.ToTensor()
        images = torch.stack([to_tensor(img) for img in pil_images]).to(self.device)
        # encode back to SD15 latent space for downstream optimization
        latents = self.encode_images_to_latents(images)
        return latents


