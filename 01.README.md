# 🚀 AI-Powered Image Captioning with ViT-GPT2 & Stable Diffusion  



An advanced **image captioning system** leveraging **ViT-GPT2** and **Stable Diffusion 2.1** to generate accurate text descriptions for images.  
This project includes **dataset creation, model training, BLEU score evaluation, and a Gradio-based interactive demo**.

---

## 📌 Table of Contents  
- [Overview](#-overview)  
- [Features](#-features)  
- [Dataset Preparation](#-dataset-preparation)
- [Dataset Structure](#-dataset-structure)  
- [Model Training](#-model-training)  
- [Evaluation](#-evaluation)  
- [Results](#-results)  

---

## 🔍 Overview  
This project focuses on **image caption generation** using a **ViT-GPT2 transformer-based model** trained on both:  
✅ **COCO dataset** (real-world images)  
✅ **AI-generated dataset** (created using Stable Diffusion 2.1)  

The model is evaluated using **BLEU Score metrics**, and the final solution is deployed via a **Gradio-based UI** for real-time inference.

---

## 🚀 Features  
✔️ **Custom AI dataset creation** using Stable Diffusion  
✔️ **ViT-GPT2 fine-tuning** on AI & COCO images  
✔️ **BLEU Score evaluation** to compare model accuracy  
✔️ **Interactive Gradio frontend** for image captioning  
✔️ **Google Drive integration** for dataset storage  

---

## 📂 Dataset Preparation  
1. **Link Google Drive with Google Colab**  
2. **Download COCO annotation files**  
3. **Split dataset into Train & Test sets**  
4. **Generate AI images using Stable Diffusion**  
5. **Prepare dataset for training (AI & COCO)**  

---

## 📊 **Dataset Structure**

The project uses a custom dataset divided into several directories for training and testing. The dataset is structured as follows:
 ```
dataset
│
├── coco_images/
│   ├── train/
│   │   └── [Images for Training]
│   ├── test/
│       └── [Images for Testing]
│
├── prompts/
│ ├── train/
│ │ └── [prompts for Training]
│ ├── test/
│      └── [prompts for Testing]
│
├── csv_files/
│   ├── train_images.csv
│   ├── test_images.csv
│   ├── train_prompts.csv
│   └── test_prompts.csv
```
---

## 🎯 Model Training  
✅ Train **ViT-GPT2** on AI-generated images  
✅ Train **ViT-GPT2** on COCO dataset  
✅ Save & load model checkpoints  

---

## 📊 Evaluation  
The model is evaluated using **BLEU Score comparisons**:  
- AI-trained model on AI-generated images  
- AI-trained model on COCO images  
- COCO-trained model on COCO images  
- COCO-trained model on AI-generated images

---


## 📸 Results  

| Model               | BLEU Score(COCO) | BLEU Score (AI) |  
|---------------------|------------------|-----------------|  
| **ViT-GPT2 (COCO)** | 0.135            | 0.105           |  
| **ViT-GPT2 (AI)**   | 0.088            | 0.094           |  



