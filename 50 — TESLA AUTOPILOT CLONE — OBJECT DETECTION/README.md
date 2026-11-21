# 🚗 Tesla Autopilot Clone — Object Detection

An end-to-end computer vision project exploring real-time road object detection using **YOLO11n (You Only Look Once)** and the **BDD100K driving dataset**.

The project investigates how pretrained object detection models can be adapted to autonomous-driving environments through transfer learning, fine-tuning, and selective classification-head training.

---

## 📌 Project Overview

Autonomous driving systems need to identify and locate surrounding objects accurately to understand their driving environment.

This project implements and compares YOLO11n object detection models trained and evaluated using the BDD100K dataset.

The primary objectives are:

- Understand and implement the YOLO11 object detection architecture.
- Adapt a COCO-pretrained YOLO11n model to BDD100K classes.
- Perform full model fine-tuning.
- Experiment with classification-output-only training.
- Compare pretrained and fine-tuned model performance.
- Evaluate detection performance using Precision, Recall, mAP50, and mAP50-95.
- Perform qualitative and quantitative comparisons on validation images.

## 🎯 Project Objectives

1. Implement an object detection pipeline using YOLO11n.
2. Adapt the detection head from 80 COCO classes to 10 BDD100K classes.
3. Investigate transfer learning for autonomous-driving object detection.
4. Compare full fine-tuning against selective classification-output training.
5. Evaluate performance on the BDD100K validation dataset.
6. Analyze detection results through visualizations and evaluation metrics.

---

## 🧠 Model Architecture

**YOLO11n (Nano)** is used as the primary object detection architecture.

The model consists of three major components:

| Component | Responsibility |
|---|---|
| Backbone | Extracts visual features from input images |
| Neck | Combines features at different scales |
| Detection Head | Predicts bounding boxes and object classes |

### Detection Head Adaptation

The original COCO-pretrained YOLO11n model predicts 80 object classes.

For this project, its classification outputs are adapted to the 10 BDD100K classes.

The YOLO11n detection head contains:

- `cv2` — Bounding-box regression branches.
- `cv3` — Classification branches.
- `dfl` — Distribution Focal Loss component.

The final classification convolutions at all three detection scales are replaced with 10-class output layers.

---

## 🚦 Dataset — BDD100K

The project uses the **Berkeley DeepDrive 100K (BDD100K)** dataset, designed for autonomous-driving research.

The dataset contains real-world driving scenes with challenging conditions such as:

- Different lighting conditions.
- Urban and highway environments.
- Multiple road users.
- Traffic signs and traffic lights.
- Different weather and visibility conditions.

### Object Classes

The project uses 10 object classes:

| Class ID | Object Class |
|---:|---|
| 0 | Car |
| 1 | Traffic Sign |
| 2 | Traffic Light |
| 3 | Person |
| 4 | Truck |
| 5 | Bus |
| 6 | Bike |
| 7 | Rider |
| 8 | Motor |
| 9 | Train |

### Dataset Configuration

- Annotation format: YOLO
- Bounding boxes: Normalized coordinates
- Dataset configuration: `data.yaml`
- Validation manifest: `val.txt`
- Validation images: 10,000

---

## ⚙️ Experiments

Two training approaches are investigated.

### Experiment 1 — Full Fine-tuning

A COCO-pretrained YOLO11n model is adapted to the BDD100K dataset.

The model is fine-tuned for the target object detection task.

**Checkpoint:**

`outputs/yolo11n_bdd100k/weights/best.pt`

### Experiment 2 — Classification-Output-Only Training

This experiment investigates whether adapting only the final classification output layers can improve class prediction for autonomous-driving objects.

The procedure:

1. Load COCO-pretrained YOLO11n.
2. Replace the three final classification output convolutions.
3. Change the output classes from 80 to 10.
4. Freeze all model parameters.
5. Unfreeze only the three final classification output convolutions.
6. Train the selected parameters on BDD100K.

**Trainable parameters:**

| Parameter | Value |
|---|---:|
| Total parameters | 2,607,070 |
| Trainable parameters | 2,430 |
| Frozen parameters | 2,604,640 |
| Trainable percentage | 0.0932% |

Only six parameter tensors are trainable: the weights and biases of the three final classification convolutions.

This experiment uses a custom Ultralytics trainer to preserve selective parameter freezing during training.

**Checkpoint:**

`outputs/yolo11n_bdd100k_classification/weights/best.pt`

---

## 📊 Evaluation Methodology

Both models are evaluated using the BDD100K validation dataset.

### Detection Metrics

| Metric | Description |
|---|---|
| Precision | Proportion of predicted detections that are correct |
| Recall | Proportion of ground-truth objects successfully detected |
| mAP50 | Mean Average Precision at IoU 0.50 |
| mAP50-95 | Mean Average Precision across IoU thresholds 0.50–0.95 |

### Per-Class Evaluation

Performance is also analyzed separately for all 10 object classes.

This helps identify which road objects are detected effectively and which classes remain challenging.

### Qualitative Comparison

A fixed random seed is used to select 10 validation images for visual comparison.

The visual analysis includes:

- Original image.
- Ground-truth bounding boxes.
- Pretrained YOLO11n predictions.
- Fine-tuned YOLO11n predictions.

### Object Count Analysis

An additional count-based comparison is performed across all 10,000 validation images.

Metrics include:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Exact Count Accuracy.
- Bias.
- Count Precision.
- Count Recall.
- Count F1-score.

**Note:** Count-based metrics measure agreement in the number of objects per image. They do not replace IoU-based detection evaluation or mAP.

---

## 🛠️ Technology Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Ultralytics | YOLO11 implementation and training |
| YOLO11n | Object detection architecture |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pandas | Evaluation and data analysis |
| Matplotlib | Result visualization |
| Jupyter Notebook | Experiment development |
| CUDA | GPU acceleration |
| Git & GitHub | Version control |

---

## 📁 Project Structure

```text
50 — TESLA AUTOPILOT CLONE — OBJECT DETECTION/
│
├── dataset/
│   └── processed/
│       ├── images/
│       ├── labels/
│       ├── manifests/
│       │   └── val.txt
│       └── data.yaml
│
├── notebooks/
│   ├── 01_*.ipynb
│   ├── 02_*.ipynb
│   ├── 05_classification_output_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── outputs/
│   ├── yolo11n_bdd100k/
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   │
│   └── yolo11n_bdd100k_classification/
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
├── evaluation_outputs/
│   ├── visual_comparisons/
│   ├── finetuned_per_class_metrics.csv
│   ├── finetuned_overall_metrics.csv
│   ├── pretrained_per_class_metrics.csv
│   └── pretrained_overall_metrics.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above describes the main project components; optional files and intermediate outputs may vary.*

---



## 🚀 Future Improvements

- Optimize inference speed for real-time autonomous-driving applications.
- Experiment with larger YOLO architectures.
- Investigate class imbalance and rare-object detection.
- Improve detection of small objects such as traffic lights and traffic signs.
- Explore model quantization and deployment optimization.
- Integrate object tracking for temporal scene understanding.
- Extend the project towards lane detection and traffic-scene perception.
- Deploy the model through a lightweight inference application.

---
