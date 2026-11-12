# 🏛️ Landmark Recognition Using Deep Learning

A Deep Learning-based Landmark Recognition System built using **PyTorch and ResNet50** to identify landmarks from images across 51 classes using the **Google Landmark Dataset v2**.

The project implements an end-to-end image classification pipeline, including dataset acquisition, exploratory data analysis, preprocessing, class balancing, data augmentation, transfer learning, model training, evaluation, error analysis, and deployment through a Streamlit web application.

---

## 📌 Project Overview

Landmark recognition is a computer vision task that involves identifying geographical locations or landmarks from their visual characteristics.

This project uses a pretrained ResNet50 architecture with transfer learning to classify uploaded images into one of 51 landmark categories.

The primary objective is to develop a classification pipeline that can:

- Identify landmarks from input images.
- Handle class imbalance in the dataset.
- Improve generalization through data augmentation.
- Evaluate model performance using multiple classification metrics.
- Analyze misclassification patterns.
- Provide an interactive interface for real-world image testing.

## 📂 Dataset

**Dataset:** Google Landmark Dataset v2 (GLDv2)

**Selected Classes:** 51 landmark categories.

The dataset contains landmark images associated with their respective category labels.

### Dataset Preparation

The following steps were performed:

1. Loaded the prepared dataset using Hugging Face Datasets.
2. Inspected dataset features and sample images.
3. Analyzed the distribution of images across landmark categories.
4. Identified class imbalance.
5. Created training and validation splits.
6. Balanced the training dataset by selecting 500 images per class.

### Class Balancing Strategy

The original dataset contains an unequal number of images across different landmark categories.

To address this:

- Target images per class: **500**
- Classes with more than 500 images: Randomly sampled 500 images.
- Classes with fewer than 500 images: Random sampling with replacement was used to reach 500 samples.

This produces a balanced training dataset of:

**51 × 500 = 25,500 training samples.**

Note: Sampling with replacement can introduce repeated original images. Random transformations are applied during training to generate different augmented views.

## ⚙️ Data Preprocessing

### 1. Dataset Splitting

The original training split was divided into:

- Training: 90%
- Validation: 10%

Stratified sampling was used to preserve class proportions.

The original test split was kept separate for final model evaluation.

### 2. Image Augmentation

The following transformations were applied to training images:

| Transformation | Configuration |
|---|---|
| Random Resized Crop | 224 × 224, scale 0.85–1.0 |
| Random Horizontal Flip | Probability: 0.3 |
| Random Rotation | ±10 degrees |
| Color Jitter | Brightness, contrast, saturation and hue |
| ToTensor | Convert image to tensor |
| Normalization | ImageNet mean and standard deviation |

### 3. Validation and Testing Transformations

Validation and test images undergo:

1. Resize to 256 pixels.
2. Center crop to 224 × 224.
3. Convert to tensor.
4. ImageNet normalization.

### 4. Label Encoding

Each landmark category is converted into an integer label.

Two mappings are maintained:

- `category_to_label`: Landmark category to numerical label.
- `label_to_category`: Numerical label to landmark category.

These mappings are saved in the model checkpoint for inference.

## 🧠 Model Architecture — ResNet50

**Architecture:** Residual Network 50 (ResNet50)

**Approach:** Transfer Learning – Feature Extraction

The project uses ResNet50 pretrained on ImageNet through Torchvision.

### Transfer Learning Implementation

The original ResNet50 architecture contains a fully connected classification layer designed for 1,000 ImageNet classes.

For this project, the final classification layer was replaced with a new fully connected layer containing 51 output neurons.

```python
model = models.resnet50(
    weights=ResNet50_Weights.DEFAULT
)

model.fc = nn.Linear(
    in_features=2048,
    out_features=51
)
```

### Feature Extraction Strategy

The pretrained convolutional backbone is frozen, and only the newly initialized classification layer is trained.

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True
```

This allows the model to reuse pretrained visual features while learning landmark-specific classification boundaries.

### Model Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | ResNet50 |
| Pretrained Weights | ImageNet |
| Learning Strategy | Feature Extraction |
| Number of Classes | 51 |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Maximum Epochs | 20 |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Mode | Minimize Validation Loss |
| Scheduler Factor | 0.1 |
| Scheduler Patience | 2 |

### Training Pipeline

The training process consists of:

1. Forward propagation.
2. Loss calculation using CrossEntropyLoss.
3. Backpropagation.
4. Classification layer parameter updates.
5. Training accuracy and loss calculation.
6. Validation after each epoch.
7. Learning rate adjustment based on validation loss.
8. Saving the best-performing model.
9. Saving the latest checkpoint for training resumption.

### Checkpoint Management

Two checkpoints are maintained:

- `feature_extraction_latest.pth` – Stores the latest training state, optimizer, scheduler, training history and label mappings.
- `best_feature_extraction.pth` – Stores the model with the highest validation accuracy.

This also allows training to resume from a previously saved checkpoint.

## 📊 Model Evaluation

The trained model is evaluated using the original test split.

The evaluation pipeline includes:

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Per-class classification report

### Confusion Matrix

A confusion matrix is generated to understand the classification performance across all 51 landmark categories.

It helps identify:

- Correctly classified images.
- Frequently confused landmark categories.
- Classes with poor classification performance.

### Misclassification Analysis

The project also investigates the most frequent classification errors.

The analysis includes:

- Identifying actual and predicted landmark pairs.
- Ranking misclassification pairs by error count.
- Visualizing sample images from selected confusing categories.
- Investigating possible reasons for visual similarities between categories.

**Case Study: Corktown vs St. Lawrence, Toronto**

The evaluation identified significant confusion between these two categories.

| Actual Class | Predicted Class | Errors |
|---|---|---:|
| Corktown | St. Lawrence | 68 |
| St. Lawrence | Corktown | 40 |

This highlights the challenges of distinguishing visually similar geographical categories.

## 🚀 Streamlit Application

An interactive web application was developed using Streamlit to test the trained model.

### Application Features

- Upload landmark images.
- Supports JPG, JPEG, PNG and WEBP formats.
- Automatic image preprocessing.
- GPU/CPU device selection.
- Landmark classification.
- Top-5 prediction results.
- Prediction probabilities displayed as percentages.
- Visual probability bars.

### Inference Pipeline

1. User uploads an image.
2. Image is converted to RGB.
3. Image preprocessing is performed.
4. Trained ResNet50 checkpoint is loaded.
5. Model generates classification logits.
6. Softmax converts logits into probabilities.
7. Top-5 predictions are extracted.
8. Predicted landmark categories and probabilities are displayed.

The application uses Streamlit resource caching to avoid repeatedly loading the model.

**Note:** Softmax probabilities represent relative model confidence and are not guaranteed to be calibrated probabilities or indicators of prediction correctness.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet50 and image transformations |
| Hugging Face Datasets | Dataset loading and management |
| NumPy | Numerical operations |
| Pandas | Evaluation data handling |
| Scikit-learn | Dataset splitting and classification metrics |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Pillow | Image processing |
| Streamlit | Web application |
| tqdm | Training and evaluation progress tracking |

---

## 📈 Results

| Evaluation Metric | Result |
|---|---|
| Test Accuracy | To be updated |
| Precision | To be updated |
| Recall | To be updated |
| F1-score | To be updated |
| Best Validation Accuracy | To be updated |

*Replace the placeholders with the actual results obtained from the evaluation notebook.*

## 🔍 Key Learnings

Through this project, the following concepts were implemented and explored:

- Exploratory Data Analysis for image datasets.
- Handling class imbalance through sampling strategies.
- Image preprocessing and augmentation.
- Transfer learning using pretrained CNN architectures.
- Feature extraction using ResNet50.
- Freezing pretrained network layers.
- Training custom classification heads.
- Learning rate scheduling.
- Model checkpointing and training resumption.
- Multi-class classification evaluation.
- Confusion matrix interpretation.
- Misclassification analysis.
- Deploying deep learning models using Streamlit.

## 🔮 Future Improvements

Potential directions for improving the project include:

- Fine-tuning selected ResNet50 backbone layers.
- Experimenting with EfficientNet, ConvNeXt and MobileNet architectures.
- Exploring stronger data augmentation strategies.
- Evaluating model generalization across unseen landmark images.
- Implementing Grad-CAM for visual model interpretability.
- Optimizing inference speed and model size.
- Exploring geographical metadata and location-aware landmark recognition.

---