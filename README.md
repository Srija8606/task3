# 🐶🐱 Cat vs Dog Image Classifier

**Incremental PCA + SVC-based image classification pipeline using memory replay**

---

## 📌 Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset & Structure](#dataset--structure)  
- [Setup Instructions](#setup-instructions)  
- [Usage](#usage)  
- [Model Training Details](#model-training-details)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [Dependencies](#dependencies)  
- [License](#license)  
- [Authors](#authors)

---

## Project Overview

A Python pipeline that classifies grayscale cat and dog images using:

1. IncrementalPCA for feature reduction  
2. Batch-based memory replay (store recent samples)  
3. Training an SVC model (`sklearn.svm.SVC`) on PCA-transformed features  

The model is periodically retrained from memory instead of online learning (`SGDClassifier`), optimizing accuracy and stability.

---

## Features

- ✅ **Incremental PCA** to reduce dimensionality on large image batches  
- 📦 **Memory replay** to manage a fixed-capacity training buffer  
- 🐍 **SVC** model for final classification  
- 🔄 **Batch training workflow**, with validation after each batch  
- 💾 **PCA + model checkpointing** using `joblib`

---

## Dataset & Structure

```text
train/                     # Image directory
 ├── cat*.jpg
 └── dog*.jpg
memory_data.npz           # Stored memory of past batches
pca_incremental.joblib    # PCA pipeline file
svc_catsdogs.joblib       # Trained SVC model
README.md                 # This file
requirements.txt          # Dependencies
