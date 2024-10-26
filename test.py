import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import os

def load_fine_tuned_model(
    model_name="CompVis/stable-diffusion-v1-4",
    fine_tuned_unet_path="fine_tuned_model/unet",
    tokenizer_path=None,  # If you fine-tuned the tokenizer
    text_encoder_path=None  # If you fine-tuned the text encoder
):
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU.")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path else model_name, subfolder="tokenizer"
    )

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        text_encoder_path if text_encoder_path else model_name, subfolder="text_encoder"
    ).to(device)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)

    # Load fine-tuned UNet
    unet = UNet2DConditionModel.from_pretrained(fine_tuned_unet_path).to(device)

    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    return tokenizer, text_encoder, vae, unet, scheduler, device

def create_pipeline(tokenizer, text_encoder, vae, unet, scheduler, device):
    # Disable safety checker for simplicity
    safety_checker = None

    # Create the pipeline
    pipeline = StableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=None,
    ).to(device)

    # Enable attention slicing to reduce memory usage (optional)
    pipeline.enable_attention_slicing()

    return pipeline

def generate_images(pipeline, prompts, num_inference_steps=50, guidance_scale=7.5):
    images = []
    for prompt in prompts:
        # Generate image
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        images.append((prompt, image))
    return images

import os

def display_and_save_images(images, output_dir="generated_images"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, (prompt, image) in enumerate(images):
        # Display the image (optional)
        # image.show(title=prompt)  # Uncomment this line if running locally

        # Save the image
        image_path = os.path.join(output_dir, f"{prompt}.png")
        image.save(image_path)
        print(f"Saved image for prompt '{prompt}' at {image_path}")

if __name__ == "__main__":
    # Load the fine-tuned model
    tokenizer, text_encoder, vae, unet, scheduler, device = load_fine_tuned_model()

    # Create the pipeline
    pipeline = create_pipeline(tokenizer, text_encoder, vae, unet, scheduler, device)

    # Define prompts to test the model
    prompts = [
        "love"
    ]

    # Generate images
    images = generate_images(pipeline, prompts)

    # Display and save the images
    display_and_save_images(images)
