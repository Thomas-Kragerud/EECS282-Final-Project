from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from matplotlib import pyplot as plt

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.unet.load_attn_procs("model")

i = 0
while True:
    prompt = input("Input prompt: ")
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save(f"outputimages/output_{i}.png")
    print(f"Saved output image as outputimages/output_{i}.png")
    i += 1