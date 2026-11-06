# Face ID – Facial Recognition and Anti-Spoofing System

A deep learning-based facial recognition and authentication system built using **RetinaFace, MiniFASNetV2, and MobileFaceNet with ArcFace**.

The system detects faces through a webcam, checks whether the detected face is real or spoofed, extracts facial embeddings, and verifies the identity using cosine similarity.

---

## 1. Project Overview

This project implements a face authentication pipeline that combines face detection, anti-spoofing, and facial recognition.

The objective is to develop a lightweight facial authentication system capable of:

- Detecting faces from webcam input.
- Identifying potential spoofing attempts.
- Generating facial feature embeddings.
- Comparing a detected face against registered identities.
- Authenticating or rejecting a person based on similarity scores.
- Registering and managing multiple identities.

### System Pipeline

```text
             Webcam Input
                  |
                  v
           Face Detection
              RetinaFace
                  |
                  v
          Anti-Spoofing Check
             MiniFASNetV2
                  |
                  v
          Facial Embedding
             MobileFaceNet
                  |
                  v
         Cosine Similarity
                  |
                  v
        Identity Verification
                  |
           +------+------+
           |             |
           v             v
      Authenticated    Rejected
```

---

## 2. Technologies and Models

| Component | Technology / Model | Purpose |
|---|---|---|
| Programming Language | Python | Main development |
| Deep Learning | PyTorch | Model training and inference |
| Face Detection | RetinaFace | Face detection |
| Anti-Spoofing | MiniFASNetV2 | Real vs. fake face classification |
| Face Recognition | MobileFaceNet | Facial feature extraction |
| Loss Function | ArcFace | Discriminative identity learning |
| Similarity Metric | Cosine Similarity | Identity verification |
| GUI | Tkinter | Desktop application |
| Data Processing | NumPy, Pandas | Data handling |
| Computer Vision | OpenCV | Webcam and image processing |

---

## 3. Face Recognition Model

### MobileFaceNet

MobileFaceNet is a lightweight convolutional neural network designed for efficient facial feature extraction.

The model generates a **128-dimensional facial embedding** for every detected face.

These embeddings represent facial characteristics in a numerical feature space.

### ArcFace

ArcFace (Additive Angular Margin Loss) is used during training to improve the discriminative ability of facial embeddings.

Its objective is to:

- Minimize angular distance between embeddings of the same identity.
- Maximize angular separation between different identities.
- Improve the distinction between different facial identities.

### Model Configuration

| Parameter | Value |
|---|---:|
| Backbone | MobileFaceNet |
| Embedding Dimension | 128 |
| Training Identities | 5,103 |
| ArcFace Scale | 64 |
| ArcFace Margin | 0.5 |
| Training Epochs | 20 |
| Approximate Parameters | 1 Million |
| Input Image Size | 112 × 112 |
| Input Channels | 3 (RGB) |

---

## 4. Dataset Preparation

### CelebA Dataset

The CelebA dataset was used for training and evaluating the face recognition model.

The dataset contains facial images with identity annotations.

| Dataset Split | Images | Identities |
|---|---:|---:|
| Training | 131,442 | 5,103 |
| Validation | 16,280 | 629 |
| Testing | 16,027 | 616 |
| **Total** | **163,749** | **6,348** |

### Data Preprocessing

The following preprocessing operations were performed:

1. Loaded images and identity annotations.
2. Organized images using identity labels.
3. Created separate training, validation, and testing splits.
4. Ensured identity-disjoint splits.
5. Resized images to 112 × 112 pixels.
6. Converted images into RGB format.
7. Converted images into PyTorch tensors.

**Important:** The training, validation, and testing identities were kept separate to evaluate the model on identities not encountered during training.

---

## 5. Face Recognition Evaluation

The trained MobileFaceNet model was evaluated using genuine and impostor face pairs.

- **Genuine pairs:** Two images belonging to the same identity.
- **Impostor pairs:** Two images belonging to different identities.

A total of 25,000 genuine pairs and 25,000 impostor pairs were sampled for each evaluation split.

### Test Results

| Metric | Test Result |
|---|---:|
| ROC-AUC | 0.9747 |
| Equal Error Rate (EER) | 7.354% |
| EER Threshold | 0.9495422 |
| Accuracy | 92.646% |
| False Acceptance Rate (FAR) | 7.612% |
| False Rejection Rate (FRR) | 7.096% |
| True Acceptance Rate (TAR) | 92.904% |

### Validation Results

| Metric | Validation Result |
|---|---:|
| ROC-AUC | 0.9699 |
| Equal Error Rate (EER) | 8.114% |
| EER Threshold | 0.9491418 |

The validation-derived threshold was also evaluated on the test set to examine verification performance using a fixed threshold.

**Note:** These results represent sampled-pair evaluation on CelebA. They should not be interpreted as guaranteed real-world webcam authentication performance.

---

## 6. Anti-Spoofing Model

### MiniFASNetV2

MiniFASNetV2 is integrated into the authentication pipeline to identify potential presentation attacks.

The anti-spoofing model checks whether a detected face appears to be a real face or a spoofed presentation.

The processing pipeline includes:

1. Detecting the face using RetinaFace.
2. Extracting the face region.
3. Applying the required crop scaling.
4. Resizing the cropped face to 80 × 80 pixels.
5. Passing the processed image to MiniFASNetV2.
6. Evaluating the model prediction before face recognition.

### Anti-Spoofing Evaluation

The model was evaluated using a NUAA dataset copy.

| Metric | Result |
|---|---:|
| Total Images | 12,614 |
| Accuracy | 95.6477% |
| Precision | 98.4887% |
| FAR | 0.9455% |
| FRR | 9.3634% |
| True Positive Rate | 90.6366% |
| True Negative Rate | 99.0545% |
| Balanced Accuracy | 94.8455% |

These results are specific to the evaluated dataset and do not guarantee equivalent performance under live webcam conditions.

---

## 7. Application Features

The project includes a desktop application with webcam-based registration and authentication.

### Face Registration

- Capture facial images through a webcam.
- Generate facial embeddings using MobileFaceNet.
- Store multiple embeddings for each registered identity.
- Save identity information in individual JSON files.

### Face Authentication

- Capture a face using the webcam.
- Detect the face using RetinaFace.
- Perform anti-spoofing verification.
- Generate a 128-dimensional facial embedding.
- Compare it with registered identity embeddings.
- Authenticate or reject the identity based on the similarity score.

### Identity Management

- Register new identities.
- View registered identities.
- Delete registered identities.
- Maintain individual JSON files for each identity.

### Registration Data Format

Each registered identity is stored in a separate JSON file.

Example:

```json
{
    "identity": "Yuvanesh",
    "embedding_size": 128,
    "num_embeddings": 6,
    "embeddings": [
        [0.12, -0.08, 0.21],
        [0.15, -0.04, 0.19]
    ]
}
```

*The embedding arrays are abbreviated for illustration. Actual files contain 128 values per embedding.*

---

## 8. Project Structure

```text
FACE-RECOGNITION-PROJECT/
│
├── app.py
│
├── checkpoints/
│   └── epoch_checkpoint.pth
│
├── data/
│   ├── celeba/
│   │   └── processed/
│   │
│   ├── antispoofing/
│   │   └── raw/
│   │
│   └── registered_faces/
│       ├── identity_1.json
│       └── identity_2.json
│
├── models/
│   └── anti_spoofing/
│       └── MiniFASNetV2.pth
│
├── notebooks/
│
├── results/
│
├── src/
│   ├── arcface.py
│   ├── config.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── face_recognition_model.py
│   ├── face_engine.py
│   ├── face_database.py
│   ├── mobilefacenet.py
│   ├── train.py
│   └── transforms.py
│
└── third_party/
    └── face-anti-spoofing/
```

---

## 9. Installation and Execution

### Prerequisites

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Pillow
- Tkinter
- RetinaFace-compatible UniFace implementation
- Required model checkpoints and dataset files

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

### Install Dependencies

Install the required Python packages according to your project environment.

```bash
pip install torch torchvision
pip install numpy pandas opencv-python pillow tqdm
```

Install any additional dependencies required by the RetinaFace and anti-spoofing implementations.

### Run the Application

```bash
python app.py
```

Make sure the required model weights and configuration paths are available before launching the application.

---

## 10. Hardware Used

The primary development and training environment included:

| Component | Specification |
|---|---|
| GPU | NVIDIA GTX 1650 |
| GPU Memory | 4 GB |
| RAM | 8 GB |
| Development Language | Python |
| Deep Learning Framework | PyTorch |

---

## 11. Limitations and Future Improvements

Although the project demonstrates a complete facial authentication pipeline, several limitations remain.

### Current Limitations

- Training and benchmark evaluation were performed using CelebA, which does not establish performance across Indian demographic groups.
- Real-world webcam performance can differ from dataset-based evaluation.
- Similarity thresholds require calibration for the actual registration and authentication pipeline.
- Anti-spoofing performance may vary under different lighting conditions, camera qualities, and presentation attacks.
- The system has not been established as production-grade security software.

### Potential Improvements

- Evaluate using more demographically diverse face datasets.
- Perform real-world genuine and impostor testing with consent.
- Improve face alignment and preprocessing consistency.
- Evaluate performance across different cameras and lighting conditions.
- Investigate additional presentation-attack scenarios.
- Optimize inference speed and memory consumption.

---

## 12. Key Learnings

Through this project, the following concepts and practical skills were explored:

- Facial feature extraction using CNN architectures.
- MobileFaceNet architecture and lightweight deep learning.
- ArcFace loss and angular-margin-based classification.
- Identity-disjoint dataset splitting.
- Genuine and impostor pair generation.
- Cosine similarity-based face verification.
- ROC curves, AUC, FAR, FRR, and EER.
- Face detection using RetinaFace.
- Face anti-spoofing using MiniFASNetV2.
- PyTorch model training and inference.
- Webcam integration using OpenCV.
- JSON-based identity embedding storage.
- Building a desktop authentication application using Tkinter.

---

## 13. Disclaimer

This project was developed for educational and experimental purposes.

The reported dataset evaluation metrics should not be treated as proof of production-level security or demographic fairness. Real-world deployment requires independent evaluation, appropriate threshold calibration, presentation-attack testing, and privacy safeguards.

Facial images and biometric embeddings should only be collected and processed with appropriate consent and protection.

---

## 14. Acknowledgements

This project builds upon research and implementations related to:

- MobileFaceNet – Lightweight face recognition.
- ArcFace – Additive Angular Margin Loss.
- RetinaFace – Face detection.
- MiniFASNetV2 – Face anti-spoofing.
- CelebA – Large-scale face attributes dataset.
- NUAA – Face anti-spoofing dataset.

