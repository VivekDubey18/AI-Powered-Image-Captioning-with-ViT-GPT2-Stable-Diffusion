!pip install datasets # coco im and ai model
import os
import torch
import pandas as pd
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

# Set up the paths for the trained model and new test dataset
images_dir = "/content/drive/MyDrive/abcd/coco_images/test"
test_images_file = "/content/drive/MyDrive/abcd/test_images.csv"
test_prompt_file = "/content/drive/MyDrive/abcd/test_prompts.csv"
output_model_dir = "/content/drive/MyDrive/abcd/ai_trained_I-P_model/model"

# Load the trained model, image processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained(output_model_dir)
image_processor = ViTImageProcessor.from_pretrained(output_model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_model_dir)
tokenizer.pad_token = tokenizer.eos_token

# Function to load the test dataset
def load_data(images_file, prompts_file, images_dir):
    image_df = pd.read_csv(images_file)
    prompt_df = pd.read_csv(prompts_file)
    df = pd.merge(image_df, prompt_df, on='id')

    def load_image_and_caption(row):
        image_path = os.path.join(images_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")
        caption = row["prompt"]
        return {"image": image, "caption": caption}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(load_image_and_caption)
    return dataset

# Preprocess the test dataset
def preprocess_data(example):
    inputs = image_processor(images=example["image"], return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]

    # Tokenize captions
    tokenized_caption = tokenizer(example["caption"], return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    labels = tokenized_caption.input_ids.squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100  # Set pad tokens to -100

    return {"pixel_values": pixel_values, "labels": labels, "caption": example["caption"]}

# Load and preprocess the test dataset
test_dataset = load_data(test_images_file, test_prompt_file, images_dir)
test_dataset = test_dataset.map(preprocess_data, remove_columns=["image", "caption"])
test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# Define the compute_metrics function for BLEU score evaluation
def compute_metrics(pred):
    # Extract the predicted token IDs
    logits = pred.predictions[0]
    predicted_ids = logits.argmax(-1)

    # Decode the predicted captions
    predicted_captions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    # Extract the true labels
    labels = pred.label_ids
    labels[labels == -100] = tokenizer.pad_token_id  # Replace -100 with pad token ID
    true_captions = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU scores
    bleu_scores = []
    for ref, pred in zip(true_captions, predicted_captions):
        ref_tokens = ref.split()  # Reference caption as list of tokens
        pred_tokens = pred.split()  # Predicted caption as list of tokens
        bleu_score = sentence_bleu([ref_tokens], pred_tokens)
        bleu_scores.append(bleu_score)

    # Calculate the average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return {"bleu_score": avg_bleu_score}

# Define evaluation arguments
eval_args = TrainingArguments(
    per_device_eval_batch_size=4,
    output_dir="./results",
    logging_dir="./logs",
    report_to="none",
    remove_unused_columns=False,
)

# Initialize the Trainer for evaluation
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the test dataset
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"BLEU Score on Test Dataset: {eval_results['eval_bleu_score']:.4f}")
