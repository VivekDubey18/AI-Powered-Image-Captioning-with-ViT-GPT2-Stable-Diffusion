!pip install gradio transformers torch datasets pillow

import gradio as gr
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
from PIL import Image

# Define paths (make sure your Drive is mounted in Colab)
output_model_dir = "/content/drive/MyDrive/abcd/ai_trained_I-P_model/model"

# Load the ViT-GPT2 model, tokenizer, and image processor
model = VisionEncoderDecoderModel.from_pretrained(output_model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_model_dir)
image_processor = ViTImageProcessor.from_pretrained(output_model_dir)

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token

# Ensure the model is on the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the prediction function
def generate_caption(image):
    # Preprocess the input image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    # Generate the caption using the model
    output_ids = model.generate(pixel_values, max_length=32, num_beams=5, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with ViT-GPT2",
    description="Upload an image to generate a caption using a fine-tuned ViT-GPT2 model."
)

# Launch the Gradio app
interface.launch(share=True)
