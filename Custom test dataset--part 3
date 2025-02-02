import os
import pandas as pd
from PIL import Image
from transformers import pipeline
from google.colab import drive
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
input_image_folder = '/content/drive/My Drive/abcd/coco_images/train'
output_dir = '/content/drive/My Drive/abcd/ai_image/ai_train'
output_csv_path = '/content/drive/My Drive/abcd/ai_train_image.csv'
prompts_csv_path = '/content/drive/My Drive/abcd/train_prompts.csv'
images_csv_path = '/content/drive/My Drive/abcd/train_images.csv'

# Load CSV files
df_prompts = pd.read_csv(prompts_csv_path)
df_images = pd.read_csv(images_csv_path)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Load the model
model_id = "stabilityai/stable-diffusion-2-1"  # Change to your preferred version
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None  # Disable safety checker if not required
).to(device)


# Function to load images
def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Initialize output CSV with headers if not already present
if not os.path.exists(output_csv_path):
    with open(output_csv_path, 'w') as f:
        f.write("sl_no,id,file_name\n")

# Process images 1 to 200
start_index = 400
end_index = 600

for index in range(start_index, end_index):
    row = df_prompts.iloc[index]
    prompt = row['prompt']
    image_id = row['id']

    # Get the corresponding file_name from the train_images.csv
    matching_row = df_images[df_images['id'] == image_id]
    if matching_row.empty:
        print(f"No matching image found for ID {image_id}. Skipping...")
        continue

    file_name = matching_row['file_name'].values[0]

    # Load the input image without adding '.jpg'
    input_image_path = os.path.join(input_image_folder, file_name)
    if not os.path.exists(input_image_path):
        print(f"Image {input_image_path} not found. Skipping...")
        continue

    # Load the image using the function
    input_image = load_image(input_image_path)
    if input_image is None:
        continue

    # Run the Stable Diffusion img2img pipeline
    result = pipe(
        prompt=prompt,
        image=input_image,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=50
    )

    # Get the generated image
    enhanced_image = result.images[0]

    # Save the generated image with the same filename
    output_image_path = os.path.join(output_dir, file_name)
    enhanced_image.save(output_image_path, format='JPEG')

    # Append to the ai_train_image CSV
    with open(output_csv_path, 'a') as f:
        f.write(f"{index+1},{image_id},{file_name}\n")

    print(f"Image {index+1} with ID {image_id} processed and saved as {file_name}")

print("All images processed and CSV updated successfully.")
