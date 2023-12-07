from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from collections import defaultdict

from matplotlib import pyplot as plt

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.unet.load_attn_procs("model")
image_type = defaultdict(int)
i = 0
while True:
    prompt = input("Input prompt: ")
    image_type_name = input("give a image type: ").lower()
    image_type[image_type_name] += 1
    image = pipe(prompt, num_inference_steps=25).images[0]
    file_name = f'{image_type_name}_{image_type[image_type_name]}_in:{prompt}'
    image.save(f"outputimages/{file_name}.png")
    print(f"Saved output image as outputimages/{file_name}.png")
