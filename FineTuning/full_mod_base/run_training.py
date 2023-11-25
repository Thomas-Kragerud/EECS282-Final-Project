import os
import subprocess

def main():

    # path to training file
    train_file_path = "diffusers/examples/dreambooth/train_dreambooth.py"
    #train_file_path = "/Users/thomas/eecs282/FinalProject/diffusers/examples/dreambooth/train_dreambooth.py"


    # Set environment variables
    os.environ['MODEL_NAME'] = "CompVis/stable-diffusion-v1-4"
    os.environ['INSTANCE_DIR'] = "dog_images"
    os.environ['OUTPUT_DIR'] = "out"

    # Construct the command
    command = f"""
    accelerate launch {train_file_path} \
      --pretrained_model_name_or_path={os.environ['MODEL_NAME']} \
      --instance_data_dir={os.environ['INSTANCE_DIR']} \
      --output_dir={os.environ['OUTPUT_DIR']} \
      --instance_prompt="a photo of sks dog" \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --learning_rate=5e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=400 \
      --push_to_hub
    """

    # Execute the command
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    print("test2")
    main()
