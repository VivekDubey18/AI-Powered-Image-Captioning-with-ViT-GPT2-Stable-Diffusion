!pip install datasets


import os
import torch
import pandas as pd
from transformers import ViTImageProcessor, GPT2Tokenizer, VisionEncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Load the ViT-GPT2 model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the image processor and tokenizer
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Ensure that the tokenizer uses pad tokens
tokenizer.pad_token = tokenizer.eos_token

# Define paths
images_dir = "/content/drive/MyDrive/abcd/ai_image/ai_train"
test_images_dir = "/content/drive/MyDrive/abcd/ai_image/ai_test"
train_images_file = "/content/drive/MyDrive/abcd/ai_train_image.csv"
train_prompt_file = "/content/drive/MyDrive/abcd/train_prompts.csv"
test_images_file = "/content/drive/MyDrive/abcd/ai_test_image.csv"
test_prompt_file = "/content/drive/MyDrive/abcd/test_prompts.csv"
output_model_dir = "/content/drive/MyDrive/abcd/ai_trained_I-P_model"

# Load the dataset from the CSV files
def load_data(images_file, prompts_file, images_dir):
    image_df = pd.read_csv(images_file)
    prompt_df = pd.read_csv(prompts_file)
    df = pd.merge(image_df, prompt_df, on='id')

    # Prepare the dataset in Hugging Face format
    def load_image_and_caption(row):
        image_path = os.path.join(images_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")
        caption = row["prompt"]
        return {"image": image, "caption": caption}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(load_image_and_caption)

    return dataset

# Preprocess the dataset
def preprocess_data(example):
    inputs = image_processor(images=example["image"], return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]

    tokenized_caption = tokenizer(example["caption"], return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    labels = tokenized_caption.input_ids.squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels, "caption": example["caption"]}

# Define compute metrics function for BLEU score
def compute_metrics(pred):
    # Extract the predicted token IDs
    logits = pred.predictions[0]
    predicted_ids = logits.argmax(-1)

    # Decode the predicted captions
    predicted_captions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    # Extract the true labels
    labels = pred.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    true_captions = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    bleu_scores = []
    for ref, pred in zip(true_captions, predicted_captions):
        ref_tokens = ref.split()  # Reference caption as list of tokens
        pred_tokens = pred.split()  # Predicted caption as list of tokens
        bleu_score = sentence_bleu([ref_tokens], pred_tokens)
        bleu_scores.append(bleu_score)

    # Calculate the average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return {"bleu_score": avg_bleu_score}

# Load and preprocess the training dataset
train_dataset = load_data(train_images_file, train_prompt_file, images_dir)
train_dataset = train_dataset.map(preprocess_data, remove_columns=["image", "caption"])

# Load and preprocess the test dataset
test_dataset = load_data(test_images_file, test_prompt_file, test_images_dir)
test_dataset = test_dataset.map(preprocess_data, remove_columns=["image", "caption"])

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs",
    save_total_limit=10,
    evaluation_strategy="epoch",
    save_steps=200,
    logging_steps=200,
    remove_unused_columns=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
