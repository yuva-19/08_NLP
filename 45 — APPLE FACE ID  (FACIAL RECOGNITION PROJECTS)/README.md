# Face Recognition & Anti-Spoofing Authentication System

A facial authentication system inspired by Apple Face ID, built using **MobileFaceNet, ArcFace, RetinaFace, and DeepFace Anti-Spoofing**.

The system combines facial detection, presentation attack detection, deep facial feature extraction, and cosine similarity-based verification to authenticate registered users.

---

## 📌 Project Overview

Traditional face recognition systems identify individuals by comparing facial features. However, a recognition system alone may be vulnerable to presentation attacks, such as using printed photographs or images displayed on mobile screens.

This project implements a multi-stage facial authentication pipeline that incorporates anti-spoofing checks before identity verification.

The primary objectives are:

- Detect and align faces from camera input.
- Detect potential face spoofing attempts.
- Extract discriminative facial embeddings.
- Verify identity using cosine similarity.
- Register and manage authorized users.
- Store facial embeddings instead of raw registration photographs.

## 🏗️ System Architecture

```text
                 CAMERA INPUT
                      |
                      ▼
                RETINAFACE
             Face Detection
             Facial Alignment
                      |
                      ▼
           DEEPFACE ANTI-SPOOFING
             Real / Spoof Check
                      |
               Real Presentation
                      |
                      ▼
               MOBILEFACENET
             Feature Extraction
                      |
                      ▼
              128-D EMBEDDING
                      |
                      ▼
                L2 NORMALIZATION
                      |
                      ▼
              COSINE SIMILARITY
                      |
                      ▼
             REGISTERED EMBEDDINGS
                      |
                      ▼
             AUTHENTICATION DECISION
                /           \
          Match Found     No Match
              |               |
              ▼               ▼
         ACCESS GRANTED   ACCESS DENIED
```

## 🧠 Technologies & Models

### 1. RetinaFace — Face Detection

RetinaFace is used to detect faces and facial landmarks in camera images.

Responsibilities:
- Detect facial regions.
- Generate face bounding boxes.
- Identify facial landmarks.
- Prepare the face region for subsequent processing.

### 2. DeepFace — Anti-Spoofing

DeepFace's anti-spoofing functionality is used to identify potential presentation attacks.

It checks whether the detected face appears to be a real facial presentation or a suspected spoof attempt.

Examples of potential spoofing attempts:
- Printed photographs.
- Images displayed on mobile screens.
- Replayed facial images.

### 3. MobileFaceNet — Feature Extraction

MobileFaceNet is a lightweight CNN architecture designed specifically for efficient face verification.

Key architectural concepts:
- Depthwise separable convolution.
- Inverted residual bottleneck blocks.
- Channel expansion and linear projection.
- PReLU activation.
- Global depthwise convolution.

The network generates a **128-dimensional facial embedding** representing the extracted facial features.

### 4. ArcFace — Discriminative Feature Learning

ArcFace (Additive Angular Margin Loss) is used during model training to improve facial identity discrimination.

It introduces an angular margin between identity classes, encouraging:
- Compact embeddings for the same identity.
- Better separation between different identities.
- Discriminative facial representations.

The training classification head is not required during registered-user authentication. The MobileFaceNet backbone is used to extract embeddings.

### 5. Cosine Similarity — Identity Verification

Cosine similarity measures the similarity between two facial embeddings.

\[
\text{Cosine Similarity}(A,B)
=
\frac{A \cdot B}{\|A\|_2\|B\|_2}
\]

Both embeddings are L2-normalized before comparison.

A configurable similarity threshold determines whether the captured face matches a registered identity.

---

## ⚙️ Authentication Workflow

1. The user captures a facial image through the Streamlit camera interface.
2. RetinaFace detects the face and prepares the facial region.
3. DeepFace anti-spoofing checks for potential presentation attacks.
4. If the presentation passes the anti-spoofing check, MobileFaceNet extracts a 128-dimensional embedding.
5. L2 normalization scales the embedding to unit length.
6. The embedding is compared against registered user embeddings using cosine similarity.
7. The system grants or denies access based on the similarity threshold.

## ✨ Key Features

- **Face Authentication:** Verify registered users through facial embeddings.
- **User Registration:** Register new users and generate facial embedding records.
- **Anti-Spoofing:** Include a presentation attack detection stage.
- **Face Detection:** Automatically detect and process facial regions.
- **User Management:** View, search, and delete registered users.
- **Embedding Storage:** Store facial embedding records in `.npz` files rather than retaining raw registration photographs.
- **Interactive Interface:** Streamlit-based interface for registration and authentication.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Face Detection | RetinaFace |
| Anti-Spoofing | DeepFace |
| Face Recognition Backbone | MobileFaceNet |
| Training Objective | ArcFace |
| Similarity Metric | Cosine Similarity |
| Interface | Streamlit |
| Numerical Processing | NumPy |
| Hardware Acceleration | CUDA |

---

## 📊 Model Configuration

| Parameter | Configuration |
|---|---|
| Backbone | MobileFaceNet |
| Training Head | ArcFace |
| Embedding Dimension | 128 |
| Training Identity Classes | 5,103 |
| Embedding Normalization | L2 Normalization |
| Verification Metric | Cosine Similarity |
| Face Detection | RetinaFace |
| Anti-Spoofing | DeepFace |

## 🔐 Data Storage & Privacy

The registration process stores facial embeddings in `.npz` files instead of retaining raw registration photographs.

These embeddings are biometric data and should still be treated as sensitive information.

- Keep registered embedding files private.
- Do not commit personal facial embeddings to public repositories.
- Exclude private checkpoints and user data from Git unless there is a deliberate reason to share them.
- Protect access to the registration and user-management functions.

## ⚠️ Limitations

This project is an experimental facial authentication prototype and is not a certified replacement for Apple Face ID or a production-grade biometric security system.

Important considerations:

- Anti-spoofing predictions are not guaranteed to detect every presentation attack.
- Authentication performance depends on lighting, camera quality, face alignment, and image conditions.
- Similarity threshold selection requires proper validation using independent genuine and impostor samples.
- Performance across different demographics must be evaluated.
- A successful comparison against a registered embedding does not independently establish real-world security.

## 🔮 Future Improvements

- Improve anti-spoofing robustness against different presentation attacks.
- Calibrate the authentication threshold using an independent validation dataset.
- Evaluate False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER), and ROC-AUC.
- Perform demographic robustness testing.
- Optimize inference for mobile and edge devices.
- Explore stronger challenge-based liveness detection.
- Implement secure biometric data encryption and access controls.

---

## 🎯 Project Objective

The ultimate goal of this project is to develop a lightweight, efficient, and more secure facial authentication pipeline by integrating face detection, anti-spoofing, and deep-learning-based facial recognition.

**Built with Python, PyTorch, and Deep Learning.**