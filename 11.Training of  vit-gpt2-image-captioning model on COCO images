# Install dependencies
!pip install datasets transformers pillow nltk torchvision

import os
import torch
import pandas as pd
from transformers import ViTImageProcessor, GPT2Tokenizer, VisionEncoderDecoderModel, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from torchvision import transforms

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Load the ViT-GPT2 model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the image processor and tokenizer
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer.pad_token = tokenizer.eos_token

# Define paths
images_dir = "/content/drive/MyDrive/abcd/coco_images/train"
test_images_dir = "/content/drive/MyDrive/abcd/coco_images/test"
train_images_file = "/content/drive/MyDrive/abcd/train_images.csv"
train_prompt_file = "/content/drive/MyDrive/abcd/train_prompts.csv"
test_images_file = "/content/drive/MyDrive/abcd/test_images.csv"
test_prompt_file = "/content/drive/MyDrive/abcd/test_prompts.csv"
output_model_dir = "/content/drive/MyDrive/abcd/trained_I-P_model"

# Define data augmentation transformations
augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
])

# Load the dataset from the CSV files
def load_data(images_file, prompts_file, images_dir, augment=False):
    image_df = pd.read_csv(images_file)
    prompt_df = pd.read_csv(prompts_file)
    df = pd.merge(image_df, prompt_df, on='id')

    def load_image_and_caption(row):
        image_path = os.path.join(images_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Apply augmentations if enabled
        if augment:
            image = augmentation_pipeline(image)

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

# Define compute metrics function for BLEU score with beam search decoding
def compute_metrics(pred):
    logits = pred.predictions[0]
    predicted_ids = logits.argmax(-1)
    predicted_captions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    labels = pred.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    true_captions = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_scores = [
        sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(true_captions, predicted_captions)
    ]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return {"bleu_score": avg_bleu_score}

# Load and preprocess the training dataset with augmentation
train_dataset = load_data(train_images_file, train_prompt_file, images_dir, augment=True)
train_dataset = train_dataset.map(preprocess_data, remove_columns=["image", "caption"])

# Load and preprocess the test dataset without augmentation
test_dataset = load_data(test_images_file, test_prompt_file, test_images_dir, augment=False)
test_dataset = test_dataset.map(preprocess_data, remove_columns=["image", "caption"])

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    per_device_train_batch_size=4,
    num_train_epochs=15,  # Increased epochs for thorough training
    learning_rate=5e-6,   # Lower learning rate for stable training
    logging_dir="./logs",
    save_total_limit=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Match evaluation and save strategies
    logging_steps=200,
    remove_unused_columns=False,
    load_best_model_at_end=True,  # Load the best model based on eval BLEU
)

# Curriculum learning callback to track BLEU improvements
class CurriculumLearningCallback(TrainerCallback):
    def __init__(self):
        self.current_bleu_score = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        new_bleu_score = metrics.get("eval_bleu_score", 0)
        if new_bleu_score > self.current_bleu_score:
            self.current_bleu_score = new_bleu_score
            print(f"New BLEU Score: {self.current_bleu_score}")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[CurriculumLearningCallback()],
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
