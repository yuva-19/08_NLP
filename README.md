# 🚀 NLP & Computer Vision — Learning Repository

A structured, hands-on learning repository covering **Natural Language Processing (NLP), Computer Vision (CV), and practical AI/ML engineering**.

This repository documents the progression from classical text processing and numerical representations to Transformer-based NLP, modern Computer Vision, model development, deployment, and production-oriented ML workflows.

The curriculum is organized into **60 modules**, following a progressive learning approach:

**Understand → Implement → Experiment → Evaluate → Build → Deploy → Present**

---

## 🎯 Learning Objectives

The primary objective of this repository is to develop practical knowledge of NLP and Computer Vision while building the ability to implement, evaluate, explain, and deploy machine learning systems.

By completing the curriculum, the intended outcomes are:

- Understand classical and modern NLP techniques.
- Build complete text-processing and text-classification pipelines.
- Understand word embeddings and contextual representations.
- Implement Seq2Seq, attention, and Transformer architectures.
- Work with pretrained Transformer models using Hugging Face.
- Fine-tune models for practical NLP tasks.
- Develop text summarization and multilingual translation systems.
- Understand image processing and enhancement using OpenCV.
- Build image-classification systems using deep learning.
- Understand and implement object detection and segmentation.
- Develop facial recognition and landmark recognition systems.
- Build practical AI applications using APIs and interactive interfaces.
- Containerize and deploy trained models.
- Apply experiment tracking, model versioning, and basic monitoring.
- Prepare technical documentation and portfolio-ready projects.

---

# 📚 Curriculum Overview

The repository follows a structured 60-module curriculum divided into major learning areas.

## Phase 1 — Natural Language Processing

<details>
<summary><strong>Modules 1–8: NLP Foundations & Classical Text Representation</strong></summary>

- Introduction to NLP
- NLP Pipelines & Text Data
- Text Normalization & Preprocessing
- Text Processing Implementation
- From Text to Numbers
- Bag of Words Implementation
- TF-IDF
- TF-IDF Implementation & Classical Baselines

**Core concepts:**
- NLP pipelines
- Regular expressions
- Unicode and text normalization
- Tokenization
- Stopword removal
- Stemming and lemmatization
- N-grams
- One-hot encoding
- Bag of Words
- Term Frequency and Inverse Document Frequency
- Logistic Regression, Naive Bayes, and Linear SVM

**Libraries:** Python, NLTK, spaCy, Scikit-learn

</details>

<details>
<summary><strong>Modules 9–15: Word Embeddings & Text Classification</strong></summary>

- Word Embeddings
- Word2Vec
- GloVe & FastText
- Pretrained Embeddings
- Text Classification
- Sentiment Analysis Implementation
- Document Classification

**Core concepts:**
- Dense vector representations
- Distributional hypothesis
- CBOW and Skip-Gram
- Negative sampling
- Cosine similarity
- Static vs contextual embeddings
- Binary, multiclass, and multilabel classification
- Sentiment analysis
- Classification evaluation
- Error analysis

**Libraries:** NumPy, PyTorch, Gensim, Scikit-learn

</details>

<details>
<summary><strong>Modules 16–22: Sequence NLP, Seq2Seq & Attention</strong></summary>

- NLP-Specific Sequence Modeling
- Sequence Labeling
- Sequence-to-Sequence for NLP
- Seq2Seq Implementation
- Why Attention?
- Attention Mechanism
- Attention Implementation

**Core concepts:**
- Language as sequential data
- Padding and attention masks
- POS tagging
- Named Entity Recognition
- BIO tagging
- Encoder-decoder architecture
- Teacher forcing
- Context-vector bottleneck
- Query, Key, and Value
- Scaled dot-product attention
- Self-attention and cross-attention
- Multi-head attention

**Implementation:** PyTorch-based attention and sequence-modeling implementations.

</details>

<details>
<summary><strong>Modules 23–28: Transformers & Modern Transformer NLP</strong></summary>

- Transformer Architecture
- Transformer Implementation & Generation
- Subword Tokenization
- BERT & Pretrained Encoder Models
- Hugging Face Transformers & Fine-Tuning
- Transformer NLP Tasks: NER, POS & QA

**Core concepts:**
- Transformer encoder and decoder
- Positional encoding
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Padding and causal masks
- Autoregressive generation
- Greedy decoding and beam search
- BPE, WordPiece, and SentencePiece
- BERT architecture and pretraining
- Pretrained model families
- Token classification
- Extractive question answering
- Fine-tuning workflows

**Libraries:** PyTorch, Hugging Face Transformers, Hugging Face Tokenizers, Datasets

</details>

<details>
<summary><strong>Modules 29–33: NLP Applications & GUVI Projects</strong></summary>

- Airbnb Customer Review Summarization — Part 1
- Airbnb Customer Review Summarization — Part 2
- Machine Translation & Multilingual NLP
- Microsoft Translator: Multilingual Chatbot
- Semantic NLP & Sentence Embeddings

**Core concepts:**
- Extractive and abstractive summarization
- Encoder-decoder Transformers
- Fine-tuning for summarization
- Text generation
- ROUGE evaluation
- Translation architectures
- Multilingual tokenization
- BLEU evaluation
- Language identification
- Sentence embeddings
- Semantic similarity
- Document similarity

**Major applications:**
- Airbnb Customer Review Summarization
- Microsoft Translator — Multilingual Chatbot
- Sentence similarity and semantic search exercises

</details>

---

## Phase 2 — Computer Vision

<details>
<summary><strong>Modules 34–39: Computer Vision Fundamentals & Image Classification</strong></summary>

- Computer Vision Fundamentals
- Image Operations
- Image Enhancement
- Image Processing
- Image Classification & CV Data Pipelines
- Transfer Learning for Computer Vision

**Core concepts:**
- Image representation
- Pixels, channels, and color spaces
- Image tensors
- Image loading and saving
- Resizing, cropping, rotation, and transformations
- Image augmentation
- Histograms and contrast enhancement
- Noise reduction and filtering
- Edge detection
- Thresholding
- Morphological operations
- Contour detection
- Image classification
- Dataset preparation
- Class imbalance
- Transfer learning
- Feature extraction and fine-tuning

**Libraries:** OpenCV, Pillow, NumPy, PyTorch, TorchVision

**Models covered:** ResNet, MobileNet, EfficientNet, ConvNeXt

</details>

<details>
<summary><strong>Modules 40–42: Object Detection</strong></summary>

- Object Detection Fundamentals
- Object Detection Architectures
- Object Detection Implementation

**Core concepts:**
- Image classification vs object detection
- Bounding boxes
- Object localization
- Confidence scores
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS)
- Precision and Recall
- Average Precision (AP)
- Mean Average Precision (mAP)
- Two-stage and single-stage detectors
- Region proposals
- Feature pyramids
- Anchor-based and anchor-free concepts
- Detection training and evaluation

**Architectures covered:**
- R-CNN
- Fast R-CNN
- Faster R-CNN
- SSD
- RetinaNet
- YOLO
- DETR

**Practical outcome:** Build and evaluate a custom object detection system.

</details>

<details>
<summary><strong>Modules 43–44: Image Segmentation</strong></summary>

- Object Segmentation
- Segmentation Implementation

**Core concepts:**
- Semantic segmentation
- Instance segmentation
- Pixel-level classification
- Binary and multiclass masks
- Mask preprocessing
- Encoder-decoder architecture
- Skip connections
- Upsampling
- Segmentation loss functions
- IoU
- Dice score
- Mean IoU
- Pixel accuracy

**Architectures covered:**
- FCN
- U-Net
- DeepLab
- Mask R-CNN

**Practical outcome:** Build an image segmentation pipeline and visualize predicted masks against ground-truth masks.

</details>

<details>
<summary><strong>Modules 45–47: Face Recognition & Facial Security</strong></summary>

- Face Detection, Landmarks & Alignment
- Facial Recognition
- Apple Face ID / Facial Recognition Projects

**Core concepts:**
- Face detection
- Facial landmarks
- Face alignment
- Face embeddings
- Cosine similarity
- Euclidean distance
- Face verification
- Face identification
- Similarity thresholds
- Genuine and impostor pairs
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- ROC-style evaluation
- Presentation attacks and spoofing
- Recognition security and privacy

**Project applications:**
- Face detection and landmark visualization
- Facial recognition system
- Apple Face ID-inspired authentication system

</details>

<details>
<summary><strong>Modules 48–51: Landmark Recognition & Modern Computer Vision</strong></summary>

- Landmark Recognition
- Modern Computer Vision: Vision Transformers
- Computer Vision Model Building, Evaluation & Error Analysis
- Tesla Autopilot Clone — Object Detection

**Core concepts:**
- Landmark classification
- Transfer learning for landmark recognition
- CNN vs Vision Transformer
- Image patches and patch embeddings
- Positional embeddings
- Vision Transformer architecture
- Modern vision model awareness
- End-to-end CV workflows
- Classification, detection, and segmentation metrics
- Model comparison
- Dataset bias and distribution shift
- Failure-case analysis
- Real-time perception constraints

**Major project applications:**
- Google Street View — Landmark Recognition
- Tesla Autopilot Clone — Object Detection

The Tesla project focuses on a simplified autonomous-driving perception system, particularly object detection, rather than autonomous-driving planning and control.

</details>

---

## Phase 3 — AI Application Engineering & Deployment

<details>
<summary><strong>Modules 52–56: Application Development, APIs, Docker & MLOps</strong></summary>

- AI Application Engineering
- Model Serving with FastAPI & Streamlit
- Docker & Deployment
- Practical MLOps
- Production Inference, Monitoring & Model Quality

**Core concepts:**
- Project organization
- Python environments
- Dependency management
- Git and GitHub
- Configuration management
- Testing
- REST APIs
- FastAPI
- Streamlit interfaces
- Docker images and containers
- Containerized inference
- Basic cloud deployment concepts
- Experiment tracking
- MLflow
- Model versioning
- Model registry
- Inference latency
- Throughput and memory usage
- Production monitoring
- Data drift awareness
- Error handling

**Tools:** Git, GitHub, FastAPI, Streamlit, Docker, MLflow

</details>

<details>
<summary><strong>Modules 57–60: End-to-End Engineering & Career Readiness</strong></summary>

- End-to-End AI Project Engineering
- AI/ML Interview & Portfolio Readiness
- Final AI/ML Portfolio Capstone

**Core concepts:**
- Complete ML project lifecycle
- Data collection and validation
- Baseline development
- Model training and evaluation
- Experiment tracking
- Model serialization
- API development
- Docker deployment
- Monitoring
- Technical documentation
- Python and SQL revision
- ML and DL interview preparation
- NLP and CV interview preparation
- Project explanation and defense
- GitHub portfolio organization

**Final objective:** Develop projects that are understandable, reproducible, documented, and demonstrable without requiring someone to inspect the original notebooks.

</details>

---

# 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Classical NLP | NLTK, spaCy |
| Classical ML | Scikit-learn |
| Deep Learning | PyTorch |
| NLP Models | Hugging Face Transformers |
| Tokenization | Hugging Face Tokenizers |
| Word Embeddings | Word2Vec, GloVe, FastText |
| Computer Vision | OpenCV, Pillow |
| Vision Models | TorchVision |
| APIs | FastAPI |
| User Interfaces | Streamlit |
| Containers | Docker |
| Experiment Tracking | MLflow |
| Version Control | Git, GitHub |
| Development | Jupyter Notebook, VS Code |

---

# 📁 Repository Organization

The repository is organized into numbered learning directories, keeping theoretical lessons, practical implementations, and project work grouped by subject.

The main learning sections include:

```text
week 8-NLP/
│
├── 01 — INTRODUCTION TO NLP/
├── 02 — NLP PIPELINES & TEXT DATA/
├── 03 — TEXT NORMALIZATION & PREPROCESSING/
├── 04 — TEXT PROCESSING IMPLEMENTATION/
├── 05 — FROM TEXT TO NUMBERS/
├── 06 — BAG OF WORDS IMPLEMENTATION/
├── 07 — TF-IDF/
├── 08 — TF-IDF IMPLEMENTATION/
├── 09 — WORD EMBEDDINGS/
├── 10 — WORD2VEC/
├── 11 — GLOVE & FASTTEXT/
├── 12 — PRETRAINED EMBEDDINGS/
├── 13 — TEXT CLASSIFICATION/
├── 14 — SENTIMENT ANALYSIS IMPLEMENTATION/
├── 15 — DOCUMENT CLASSIFICATION/
├── 16 — NLP-SPECIFIC SEQUENCE MODELING/
├── 17 — SEQUENCE LABELING/
├── 18 — SEQUENCE-TO-SEQUENCE FOR NLP/
├── 19 — SEQ2SEQ IMPLEMENTATION/
├── 20 — WHY ATTENTION/
├── 21 — ATTENTION MECHANISM/
├── 22 — ATTENTION IMPLEMENTATION/
├── 23 — TRANSFORMER ARCHITECTURE/
├── 24 — TRANSFORMER IMPLEMENTATION/
├── 25 — SUBWORD TOKENIZATION/
├── 26 — BERT & PRETRAINED ENCODER MODELS/
├── 27 — HUGGING FACE & FINE-TUNING/
├── 28 — TRANSFORMER NLP TASKS/
├── 29 — AIRBNB SUMMARIZATION/
├── 30 — MACHINE TRANSLATION/
│
├── 31 — COMPUTER VISION FUNDAMENTALS/
├── 32 — IMAGE OPERATIONS/
├── 33 — IMAGE ENHANCEMENT/
├── 34 — IMAGE PROCESSING/
├── 35 — IMAGE CLASSIFICATION/
├── 36 — TRANSFER LEARNING/
├── 37 — OBJECT DETECTION FUNDAMENTALS/
├── 38 — OBJECT DETECTION ARCHITECTURES/
├── 39 — OBJECT DETECTION IMPLEMENTATION/
├── 40 — OBJECT SEGMENTATION/
├── 41 — SEGMENTATION ARCHITECTURES/
├── 42 — SEGMENTATION IMPLEMENTATION/
├── 43 — FACE DETECTION & LANDMARKS/
├── 44 — FACIAL RECOGNITION/
├── 47 — APPLE FACE ID/
│
├── requirements.txt
├── syllabus.md
└── README.md
```

*The tree above highlights the numbered learning folders visible in the repository. It is an overview, not a claim that every syllabus module has a separate directory with an identical name.*

---

# 🚀 Major Portfolio Projects

The curriculum includes practical projects at different stages of learning.

| Project | Main Learning Area |
|---|---|
| Cardeko Car Price Prediction | Machine Learning |
| Airbnb Customer Review Summarization | Transformer NLP |
| Microsoft Translator — Multilingual Chatbot | Machine Translation |
| Apple Face ID / Facial Recognition Security | Computer Vision |
| Facial Recognition System | Face Recognition |
| Google Street View — Landmark Recognition | Image Classification |
| Tesla Autopilot Clone — Object Detection | Object Detection |

Additional implementation exercises include:

- Text preprocessing toolkit
- Bag of Words and TF-IDF implementations
- Word2Vec implementation
- Sentiment analysis
- Document classification
- Seq2Seq implementation
- Attention implementation
- Mini Transformer sequence-to-sequence system
- Transformer NER and QA
- Image preprocessing toolkit
- Image enhancement pipeline
- Image segmentation system
- Custom image classification system
- Custom object detection system

---

# ⚙️ Environment Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd week-8-NLP
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

If the repository's `requirements.txt` contains the dependencies for your environment:

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Alternatively, open the repository using Visual Studio Code and run the notebooks using its Jupyter extension.

**Note:** Some modules may require additional datasets, pretrained checkpoints, or model-specific dependencies. Follow the instructions provided in the relevant module or project directory.

---

# 📊 Learning & Implementation Methodology

Each topic follows a practical learning cycle:

1. **Conceptual Understanding** — Understand the purpose and working principles.
2. **Mathematical Understanding** — Study the important equations and underlying operations where applicable.
3. **Implementation** — Implement the concepts using Python and relevant libraries.
4. **Experimentation** — Change parameters, compare approaches, and observe behaviour.
5. **Evaluation** — Use appropriate metrics to assess performance.
6. **Error Analysis** — Inspect incorrect predictions and identify limitations.
7. **Documentation** — Record the implementation, observations, and results.

For computationally intensive operations, progress indicators such as `tqdm` are used where appropriate to make execution progress visible.

---

# 📈 Evaluation Metrics Covered

### NLP

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Perplexity
- BLEU
- ROUGE
- Cosine Similarity
- Retrieval and Ranking Metrics
- Entity-level Precision, Recall, and F1

### Computer Vision

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Intersection over Union (IoU)
- Average Precision (AP)
- Mean Average Precision (mAP)
- Dice Score
- Mean IoU
- Pixel Accuracy

### Model Engineering

- Training and validation loss
- Inference latency
- Throughput
- Memory usage
- Model size
- Parameter count
- Failure rates

---

# 📖 Documentation & Learning Resources

The curriculum is supported by practical documentation and established learning resources:

- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/usage)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PyTorch Documentation](https://docs.pytorch.org/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TorchVision Documentation](https://docs.pytorch.org/vision/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)

---

# 🎯 Final Learning Outcome

The intended progression of this repository is:

```text
                 NLP FOUNDATIONS
                        ↓
                CLASSICAL NLP
                        ↓
                WORD EMBEDDINGS
                        ↓
                 SEQUENCE NLP
                        ↓
                    SEQ2SEQ
                        ↓
                   ATTENTION
                        ↓
                  TRANSFORMERS
                        ↓
                 TRANSFORMER NLP
                        ↓
                HUGGING FACE
                        ↓
                  FINE-TUNING
                        ↓
               NLP APPLICATIONS
                        ↓
              COMPUTER VISION
                        ↓
              IMAGE PROCESSING
                        ↓
              IMAGE CLASSIFICATION
                        ↓
                TRANSFER LEARNING
                        ↓
                OBJECT DETECTION
                        ↓
                 SEGMENTATION
                        ↓
               FACIAL RECOGNITION
                        ↓
              MODERN VISION MODELS
                        ↓
             AI APPLICATION ENGINEERING
                        ↓
                MODEL DEPLOYMENT
                        ↓
                     MLOPS
                        ↓
              END-TO-END PROJECTS
                        ↓
                PORTFOLIO READINESS
```

## 🏆 Repository Goal

Build a strong practical foundation in **Natural Language Processing, Computer Vision, and AI/ML Engineering** through structured learning, implementation, experimentation, evaluation, and real-world project development.

The focus is not simply on learning model names or running pretrained models. It is on understanding how systems work, implementing important components, evaluating their behaviour, identifying their limitations, and developing complete applications.

---

**Learning through implementation. Building through experimentation. Growing through engineering.** 🚀