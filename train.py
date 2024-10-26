import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import os


def fine_tune_stable_diffusion(
        model_name="CompVis/stable-diffusion-v1-4",
        dataset_path="kanji_dataset",
        output_dir="fine_tuned_model",
        num_train_epochs=5,
        train_batch_size=2,  # Reduced batch size due to memory constraints
        learning_rate=7e-7,
        intermediate_images_dir="intermediate_images",
        test_prompts=["beautiful"],  # Test prompts for intermediate visualization
):
    # Detect if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU.")

    # Load the pretrained model components
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Disable gradient computation for VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # Preprocess images and text
    def preprocess(examples):
        images = []
        for image_path in examples["image"]:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512))  # Resize images to 512x512
            image = transforms.ToTensor()(image)
            images.append(image)
        examples["pixel_values"] = images

        # Tokenize the captions
        inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        examples["input_ids"] = inputs.input_ids.squeeze(1)
        return examples

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True, remove_columns=["image", "text"])

    # Set the dataset format to PyTorch tensors
    dataset.set_format(type="torch", columns=["pixel_values", "input_ids"])

    # Create DataLoaders
    train_dataloader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset["test"], batch_size=train_batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Create the output directory for intermediate results
    os.makedirs(intermediate_images_dir, exist_ok=True)

    # Helper function to generate and save images for intermediate checkpoints
    def generate_intermediate_images(epoch, step, prompts):
        unet.eval()  # Set the model to evaluation mode
        pipeline = StableDiffusionPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()

        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                image = pipeline(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image_path = os.path.join(intermediate_images_dir, f"epoch_{epoch + 1}_step_{step}_prompt_{i}.png")
            image.save(image_path)
            print(f"Saved intermediate image for prompt '{prompt}' at epoch {epoch + 1}, step {step}: {image_path}")

    # Training loop
    for epoch in range(num_train_epochs):
        unet.train()
        total_steps_in_epoch = len(train_dataloader)
        steps_per_20_percent = max(1, total_steps_in_epoch // 5)  # Calculate 20% of the steps

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor as per Stable Diffusion

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to the latents (using the scheduler)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = F.mse_loss(model_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

            # Generate images every 20% of the epoch
            if step % steps_per_20_percent == 0 and step > 0:
                generate_intermediate_images(epoch, step, test_prompts)

        # Save a checkpoint after each epoch
        unet.save_pretrained(f"{output_dir}/unet_epoch_{epoch + 1}")

    # Save the final fine-tuned model components
    unet.save_pretrained(f"{output_dir}/unet")

    print("Fine-tuning completed and model saved.")


if __name__ == "__main__":
    fine_tune_stable_diffusion()