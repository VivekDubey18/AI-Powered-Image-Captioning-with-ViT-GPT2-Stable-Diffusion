import torch
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor

# Define the path to your saved checkpoint directory
checkpoint_dir = "/content/drive/MyDrive/abcd/trained_I-P_model/checkpoint-600 s=0.1355"

# Load the model from the checkpoint
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)

# Load the tokenizer from the original Hugging Face model (not from the checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the image processor from the original Hugging Face model
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Save the final model, tokenizer, and processor to a new directory
final_model_dir = "/content/drive/MyDrive/abcd/trained_I-P_model/model"
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
image_processor.save_pretrained(final_model_dir)

print(f"Model, tokenizer, and processor successfully saved to {final_model_dir}")
