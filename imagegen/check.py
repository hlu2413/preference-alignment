from diffusers import StableDiffusionPipeline
StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
print("pipeline ok")