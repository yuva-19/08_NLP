# MODULE 1 — INTRODUCTION TO NLP

## 1.1 What Is NLP?

- Natural Language Processing
- NLP vs ML vs DL
- Human language vs machine representation
- Structured vs unstructured text
- NLP pipeline
- NLP applications

### Applications

- Sentiment analysis
- Text classification
- Search
- Recommendation
- Machine translation
- Summarization
- Question answering
- Chatbots
- Information extraction
- Named Entity Recognition
- Semantic search

### 1.2 Evolution of NLP

```text
Rule-Based NLP
      ↓
Statistical NLP
      ↓
Classical Machine Learning
      ↓
Word Embeddings
      ↓
RNN / LSTM
      ↓
Seq2Seq + Attention
      ↓
Transformers
      ↓
Pretrained Transformers
      ↓
LLMs
```

### 1.3 NLP Challenges

- Ambiguity
- Polysemy
- Synonymy
- Context
- Sarcasm
- Negation
- Spelling variation
- Noisy text
- Long-range dependencies
- Multilinguality
- Domain-specific vocabulary

### 🎓 GUVI THEORY

**Introduction to NLP**

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 2 — NLP PIPELINES & TEXT DATA

## 2.1 NLP Pipeline

```text
Raw Text
   ↓
Cleaning
   ↓
Normalization
   ↓
Segmentation
   ↓
Tokenization
   ↓
Representation
   ↓
Model
   ↓
Prediction
   ↓
Evaluation
```

## 2.2 Text Data Types

- Documents
- Sentences
- Words
- Characters
- Tokens
- Subwords

## 2.3 Text Quality Problems

- HTML
- URLs
- Emojis
- Punctuation
- Special characters
- Repeated spaces
- Spelling variations
- Numbers
- Mixed casing
- Contractions

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 3 — TEXT NORMALIZATION & PREPROCESSING

## 3.1 Normalization

- Lowercasing
- Whitespace normalization
- Punctuation handling
- Special-character handling
- Number handling
- URL handling
- HTML removal
- Emoji handling

## 3.2 Unicode

- Unicode basics
- Unicode normalization
- UTF-8
- Why Unicode matters in multilingual NLP

## 3.3 Regular Expressions

Learn practical regex for:

- URLs
- Emails
- Numbers
- Special characters
- HTML tags
- Repeated spaces
- Custom patterns

## 3.4 Sentence Segmentation

- Sentence boundaries
- Abbreviations
- Sentence splitting

## 3.5 Word Tokenization

- Word-level tokenization
- Character tokenization
- Token boundaries
- Contractions

## 3.6 Stop Words

- What are stopwords?
- Why remove them?
- When NOT to remove them
- Effect on sentiment
- Effect on search

## 3.7 Stemming

- Porter Stemmer
- Snowball Stemmer
- Advantages
- Limitations

## 3.8 Lemmatization

- Root/base form
- Dictionary-based normalization
- POS-aware lemmatization
- Stemming vs lemmatization

### 🎓 GUVI THEORY

**Text Processing**

GUVI's current text-processing material also emphasizes cleaning unwanted elements such as punctuation, stopwords, emojis and HTML, along with stemming, lemmatization, n-grams and regex. syllabus(20261020-190706)

---

# MODULE 4 — TEXT PROCESSING IMPLEMENTATION

## 4.1 Build a Complete Preprocessor

Implement:

```text
Raw Text
   ↓
Unicode normalization
   ↓
HTML removal
   ↓
URL removal
   ↓
Whitespace normalization
   ↓
Case normalization
   ↓
Sentence tokenization
   ↓
Word tokenization
   ↓
Stopword handling
   ↓
Stemming / Lemmatization
   ↓
Clean text
```

## 4.2 Compare Processing Strategies

Experiment with:

- No preprocessing
- Minimal preprocessing
- Aggressive preprocessing

Observe:

- Vocabulary size
- Number of tokens
- Information loss
- Classification performance

## 4.3 Libraries

Learn practical usage of:

- NLTK
- spaCy
- Python `re`

### 💻 GUVI IMPLEMENTATION

**Text Processing Implementation — Part 1**

**Text Processing Implementation — Part 2**

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 5 — FROM TEXT TO NUMBERS

## 5.1 Why Text Representation?

Machine-learning models cannot directly consume raw text.

Therefore:

```text
Text
 ↓
Numerical Representation
 ↓
ML Model
```

## 5.2 One-Hot Encoding

- Vocabulary
- One-hot vector
- Sparse representation
- Dimensionality problem
- Lack of semantic relationships

One-hot vectors are sparse and do not inherently encode meaningful similarity between words; embeddings solve much of this representation problem.

## 5.3 N-Grams

- Unigram
- Bigram
- Trigram
- Higher-order n-grams
- Word n-grams
- Character n-grams

## 5.4 Bag of Words

Understand:

- Vocabulary
- Document-term matrix
- Word frequency
- Sparse vectors
- Limitations

### 🎓 GUVI THEORY

**Text Representation — BoW** syllabus(20261020-190706)

---

# MODULE 6 — BAG OF WORDS IMPLEMENTATION

## 6.1 Implement BoW Manually

Given:

```text
"I love NLP"
"I love Python"
```

Create:

```text
Vocabulary
 ↓
Token IDs
 ↓
Document vectors
```

## 6.2 Scikit-Learn

Learn:

- `CountVectorizer`
- vocabulary
- sparse matrix
- `fit`
- `transform`
- `fit_transform`

## 6.3 Experiment

Compare:

- Unigram
- Bigram
- Unigram + Bigram

### 💻 GUVI IMPLEMENTATION

**Text Representation Implementation — Part 1** syllabus(20261020-190706)

---

# MODULE 7 — TF-IDF

## 7.1 Term Frequency

Understand:

```text
TF
```

## 7.2 Inverse Document Frequency

Understand:

```text
IDF
```

## 7.3 TF-IDF

```text
TF-IDF = TF × IDF
```

Understand:

- Why frequent words receive lower importance
- Why rare informative words receive higher importance
- Document length effects
- Sparse representation

Scikit-learn's text workflow similarly builds count representations and transforms them into TF-IDF before classification.

## 7.4 TF-IDF Limitations

- No deep semantic understanding
- No word order beyond n-grams
- Sparse vectors
- Same word generally receives the same representation
- Weak contextual understanding

### 🎓 GUVI THEORY

**Text Representation — TF-IDF** syllabus(20261020-190706)

---

# MODULE 8 — TF-IDF IMPLEMENTATION + CLASSICAL BASELINE

## 8.1 Implement

- TF manually
- IDF manually
- TF-IDF manually

Then use:

- `TfidfVectorizer`

## 8.2 Classical NLP Pipeline

```text
Text
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
Classifier
 ↓
Prediction
```

## 8.3 Baseline Models

Learn enough to use:

- Logistic Regression
- Naive Bayes
- Linear SVM

Scikit-learn's standard text workflow combines vectorization, TF-IDF and a classifier in a pipeline. [Scikit-learn](https://scikit-learn.org/0.18/tutorial/text_analytics/working_with_text_data.html?utm_source=chatgpt.com)

### 💻 GUVI IMPLEMENTATION

**Text Representation Implementation — Part 2**

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 9 — WORD EMBEDDINGS

## 9.1 Why Embeddings?

Understand the transition:

```text
One-Hot
   ↓
BoW
   ↓
TF-IDF
   ↓
Dense Word Embeddings
```

## 9.2 Embedding Space

- Dense vectors
- Dimensionality
- Semantic similarity
- Distance
- Cosine similarity

## 9.3 Distributional Hypothesis

> Words appearing in similar contexts tend to have related meanings.

## 9.4 Static Embeddings

- One vector per word
- Context-independent representation

## 9.5 Important Limitation

Example:

```text
bank
```

can mean:

- financial institution
- river bank

A static embedding struggles to represent both meanings contextually.

Modern contextual embeddings address this by producing representations dependent on surrounding text. syllabus(20261020-190706)

---

# MODULE 10 — WORD2VEC

## 10.1 Word2Vec

Understand:

- CBOW
- Skip-gram
- Context window
- Positive examples
- Negative examples
- Embedding matrix

## 10.2 CBOW

```text
Context
 ↓
Target word
```

## 10.3 Skip-Gram

```text
Target word
 ↓
Context words
```

## 10.4 Negative Sampling

- Why it exists
- Positive pairs
- Negative pairs

## 10.5 Embedding Similarity

Experiment with:

- cosine similarity
- nearest neighbours
- analogy-style relationships

### 💻 IMPLEMENTATION

Implement a simplified Word2Vec workflow.

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 11 — GLOVE & FASTTEXT

## 11.1 GloVe

Understand conceptually:

- Global co-occurrence statistics
- Word-context matrix
- Embedding learning

## 11.2 Word2Vec vs GloVe

Compare:

- Training objective
- Local vs global information
- Practical behaviour

## 11.3 FastText

Understand:

- Character n-grams
- Subword information
- Rare words
- Unknown words
- Morphologically rich languages

### 🔥 Priority

Word2Vec: **🔥🔥🔥**

GloVe: **🔥🔥**

FastText: **🔥🔥**

Do not spend weeks implementing every algorithm from scratch. syllabus(20261020-190706)

---

# MODULE 12 — PRETRAINED EMBEDDINGS

## 12.1 Pretrained Embeddings

Learn:

- Loading pretrained vectors
- Vocabulary matching
- Unknown words
- Embedding matrix construction

## 12.2 Embedding Layer

Connect:

```text
Token ID
 ↓
Embedding Lookup
 ↓
Dense Vector
```

## 12.3 Static vs Contextual Embeddings

```text
Static
Word → One vector

Contextual
Word + Context → Context-dependent vector
```

### 💻 GUVI IMPLEMENTATION

**Word Embedding Implementations**

**Pretrained Embedding Implementations**

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 13 — TEXT CLASSIFICATION

## 13.1 Classification Problems

- Binary classification
- Multiclass classification
- Multilabel classification

## 13.2 Sentiment Analysis

- Positive
- Negative
- Neutral

## 13.3 Complete Pipeline

```text
Dataset
 ↓
Cleaning
 ↓
Tokenization
 ↓
Representation
 ↓
Model
 ↓
Training
 ↓
Evaluation
```

## 13.4 Classical Baselines

Compare:

- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- TF-IDF + SVM

### 🎓 GUVI THEORY

**Text Classification — Sentiment Analysis** syllabus(20261020-190706)

---

# MODULE 14 — SENTIMENT ANALYSIS IMPLEMENTATION

## 14.1 Dataset

Learn:

- Dataset inspection
- Labels
- Class distribution
- Train/validation/test split

## 14.2 Preprocessing

Build the complete preprocessing pipeline.

## 14.3 Baseline

Implement:

```text
TF-IDF
 +
Logistic Regression
```

## 14.4 Evaluation

Reuse:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix

Your DL syllabus already covers these evaluation concepts.

## 14.5 Error Analysis

Inspect:

- False positives
- False negatives
- Negation
- Sarcasm
- Ambiguous reviews

### 💻 GUVI IMPLEMENTATION

**Sentiment Analysis Implementation — Part 1**

**Part 2**

**Part 3** syllabus(20261020-190706)

---

# MODULE 15 — DOCUMENT CLASSIFICATION

## 15.1 Document Classification

Examples:

- News category
- Spam detection
- Topic classification
- Review classification

## 15.2 Classical Pipeline

```text
Document
 ↓
TF-IDF
 ↓
Classifier
 ↓
Class
```

## 15.3 Model Comparison

Compare:

- Naive Bayes
- Logistic Regression
- SVM

## 15.4 Error Analysis

Identify:

- Short documents
- Rare vocabulary
- Ambiguous categories
- Domain-specific terminology

### 💻 GUVI IMPLEMENTATION

**Document Classification Implementation — Part 1**

### 🔥 Priority

**🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 16 — NLP-SPECIFIC SEQUENCE MODELING

⚠️ **Do NOT relearn RNN/LSTM/GRU mathematics.**

The Deep Learning syllabus already contains these architectures and their implementation.

Instead learn:

## 16.1 Language as Sequential Data

- Token sequences
- Variable-length text
- Sequence padding
- Attention masks
- Vocabulary
- Embedding lookup

## 16.2 NLP Sequence Tasks

```text
Many-to-One
→ Sentiment classification

Many-to-Many
→ POS tagging
→ NER

Many-to-Many with different lengths
→ Translation
→ Summarization
```

## 16.3 RNN/LSTM/GRU for NLP

Understand applications rather than architecture mathematics:

- Sentiment analysis
- Language modeling
- Sequence classification
- Sequence labeling
- Seq2Seq

### 🔄 REVISION

Only revise the specific DL material required for the above applications. syllabus(20261020-190706)

---

# MODULE 17 — SEQUENCE LABELING

This is an important NLP concept.

## 17.1 POS Tagging

- Noun
- Verb
- Adjective
- Adverb
- Pronoun
- etc.

## 17.2 Named Entity Recognition

Identify:

- Person
- Organization
- Location
- Date
- Money
- Product

## 17.3 BIO Tagging

```text
B-PER
I-PER
B-ORG
I-ORG
O
```

## 17.4 Token Classification

Understand how modern Transformer models perform token-level classification. syllabus(20261020-190706)

---

# MODULE 18 — SEQUENCE-TO-SEQUENCE FOR NLP

The DL syllabus already teaches the basic encoder-decoder architecture, so here the focus shifts to **NLP applications and limitations**.

## 18.1 Seq2Seq Review

- Encoder
- Decoder
- Context vector
- Input sequence
- Output sequence

## 18.2 NLP Applications

- Translation
- Summarization
- Chatbots
- Text generation

## 18.3 Teacher Forcing

Already known from DL.

Understand its NLP role.

## 18.4 Context Vector Bottleneck

Why does compressing a long source sequence into one vector become problematic?

## 18.5 Exposure Bias

Understand conceptually.

### 🎓 GUVI THEORY

**Advanced NLP Techniques — Seq2Seq**

**Seq2Seq Part 2**

**Seq2Seq Part 3** syllabus(20261020-190706)

---

# MODULE 19 — SEQ2SEQ IMPLEMENTATION

## 19.1 Implement

Build a basic:

```text
Encoder
 ↓
Context
 ↓
Decoder
```

## 19.2 NLP Dataset

Use a small language-pair or text transformation dataset.

## 19.3 Training

Understand:

- Input sequence
- Target sequence
- `<SOS>`
- `<EOS>`
- Teacher forcing
- Padding
- Masking

## 19.4 Inference

Understand:

```text
Encoder
 ↓
Decoder
 ↓
Token
 ↓
Next token
 ↓
...
 ↓
<EOS>
```

### 💻 GUVI IMPLEMENTATION

**seq2seq Implementation** syllabus(20261020-190706)

---

# MODULE 20 — WHY ATTENTION?

## 20.1 Problem with Vanilla Seq2Seq

- Long sequences
- Information bottleneck
- Fixed context vector
- Difficult long-range dependencies

## 20.2 Attention Intuition

Instead of:

```text
Entire sentence
      ↓
One context vector
```

use:

```text
Decoder
   ↓
Look at relevant encoder states
   ↓
Construct context
```

## 20.3 Context Vector

Understand how the context vector changes at every decoding step. syllabus(20261020-190706)

---

# MODULE 21 — ATTENTION MECHANISM

## 21.1 Query

## 21.2 Key

## 21.3 Value

Understand:

```text
Q
K
V
```

## 21.4 Attention Scores

Conceptually:

```text
Query × Keys
      ↓
Scores
      ↓
Softmax
      ↓
Weights
      ↓
Weighted Values
```

## 21.5 Scaled Dot-Product Attention

Learn the mathematics:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

## 21.6 Self-Attention

Understand:

- Query
- Key
- Value
- Same sequence produces Q/K/V
- Token-to-token relationships

## 21.7 Cross-Attention

Understand:

```text
Decoder queries
      ↓
Encoder keys/values
```

## 21.8 Multi-Head Attention

- Multiple attention heads
- Different relationship patterns
- Concatenation
- Output projection

The DL syllabus already introduced self-attention and multi-head attention through ViT, but explicitly deferred deep Transformer mathematics to NLP.

### 🎓 GUVI THEORY

**Advanced NLP Techniques — Attention** syllabus(20261020-190706)

---

# MODULE 22 — ATTENTION IMPLEMENTATION

## 22.1 Implement Scaled Dot-Product Attention

From scratch with PyTorch tensors.

## 22.2 Implement Self-Attention

Given:

```text
X
 ↓
Q K V
 ↓
Attention
 ↓
Output
```

## 22.3 Implement Multi-Head Attention

Understand:

```text
Input
 ↓
Q K V
 ↓
Multiple heads
 ↓
Concat
 ↓
Linear projection
```

## 22.4 Visualize Attention

Inspect:

- Attention weights
- Important tokens
- Different heads

### 💻 GUVI IMPLEMENTATION

**Attention Implementation — Part 1**

### 🔥 Priority

**🔥🔥🔥 Must Know** syllabus(20261020-190706)

---

# MODULE 23 — TRANSFORMER ARCHITECTURE

## 24.1 Why Transformers?

Compare:

```text
RNN
 ↓
Sequential computation
```

with:

```text
Transformer
 ↓
Attention-based parallel processing
```

## 24.2 Encoder-Decoder Transformer

```text
Source
 ↓
Encoder
 ↓
Cross-Attention
 ↓
Decoder
 ↓
Target
```

## 24.3 Transformer Encoder

- Multi-head self-attention
- Add & Norm
- Feed-forward network
- Residual connection
- LayerNorm

## 24.4 Transformer Decoder

- Masked self-attention
- Cross-attention
- Feed-forward network
- Residual connections
- Layer normalization

## 24.5 Positional Information

Understand why position information is required.

The Transformer architecture and its progression from recurrent computation to attention-based processing are also central to Stanford's current NLP curriculum. [Stanford University](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 24 — TRANSFORMER IMPLEMENTATION & GENERATION

## 25.1 GUVI Transformer Implementation

Build a compact Transformer containing:

- Token embeddings
- Positional encoding
- Multi-head attention
- Feed-forward layers
- Residual connections
- Layer normalization
- Encoder
- Decoder

## 25.2 Masking

Implement and visualize:

- Padding masks
- Causal masks
- Encoder-decoder attention masks

## 25.3 Training

Build a small sequence-to-sequence task and monitor:

- Training loss
- Validation loss
- Token accuracy
- Sequence-level quality

## 25.4 Inference

Implement:

- Autoregressive decoding
- Greedy decoding
- Beam search concept
- `<SOS>` / `<EOS>`
- Maximum sequence length
- Repetition handling

## 25.5 Framework Comparison

Compare the manual implementation with PyTorch's Transformer components.

### 💻 GUVI IMPLEMENTATION

**Transformer Implementation — Part 1**

### Project

**Mini Transformer Sequence-to-Sequence System**

---

# MODULE 25 — SUBWORD TOKENIZATION

## 26.1 Why Subwords?

Problems with word-level tokenization:

- Unknown words
- Rare words
- Misspellings
- Morphology
- Huge vocabularies

## 26.2 BPE

- Byte Pair Encoding
- Merge operations
- Vocabulary creation
- Subword units

## 26.3 WordPiece

- BERT association
- Subword vocabulary
- Greedy matching
- `##` continuation tokens

## 26.4 SentencePiece / Unigram

- Raw-text tokenization
- Language-independent processing
- BPE
- Unigram model

## 26.5 Practical Tokenizer Analysis

For the same corpus compare:

- Vocabulary size
- Token count
- Unknown-token behaviour
- Sequence length
- Rare-word handling

Modern tokenizer documentation explicitly distinguishes BPE, WordPiece and Unigram approaches and their different construction strategies. [Hugging Face](https://huggingface.co/docs/tokenizers/components?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 26 — BERT & PRETRAINED ENCODER MODELS

## 27.1 BERT Architecture

- Encoder-only Transformer
- Bidirectional contextual representations
- WordPiece tokenization
- `[CLS]`
- `[SEP]`
- Attention masks

## 27.2 BERT Pretraining

- Masked Language Modeling
- Original next-sentence objective — conceptual awareness
- Contextual representations

## 27.3 BERT Heads

Understand:

- Sequence classification
- Token classification
- Question answering

## 27.4 BERT Family

Compare conceptually:

- BERT
- RoBERTa
- DistilBERT

Focus on:

> Architecture → pretraining objective → downstream use

Hugging Face's current task documentation continues to use BERT-family encoder models for sequence and token classification, with WordPiece and special tokens central to the workflow. [Hugging Face](https://huggingface.co/docs/transformers/main/tasks_explained?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 27 — HUGGING FACE TRANSFORMERS & FINE-TUNING

## 28.1 Core APIs

Learn practical use of:

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `AutoModelForTokenClassification`
- `AutoModelForQuestionAnswering`
- `AutoModelForSeq2SeqLM`

## 28.2 Dataset Pipeline

- Dataset loading
- Splitting
- `map`
- Batched tokenization
- Dynamic padding
- Data collators

## 28.3 Training

- `TrainingArguments`
- `Trainer`
- Checkpoints
- Evaluation
- Best-model loading
- Inference

## 28.4 Fine-Tuning Workflow

```text
Pretrained Model
 ↓
Task Dataset
 ↓
Tokenizer
 ↓
Data Collator
 ↓
Fine-Tuning
 ↓
Validation
 ↓
Checkpoint
 ↓
Inference
```

## 28.5 Project

Fine-tune a pretrained encoder for sentiment or topic classification.

Hugging Face's current documentation demonstrates this exact workflow using tokenization, dynamic padding, `Trainer`, evaluation and checkpointing. [Hugging Face](https://huggingface.co/docs/transformers/main/tasks/sequence_classification?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 28 — TRANSFORMER NLP TASKS: NER, POS & QA

## 29.1 Transformer NER

- Token classification
- BIO labels
- Subword-to-label alignment
- Ignoring special tokens
- Entity reconstruction

Hugging Face's token-classification workflow explicitly treats NER as token classification and demonstrates fine-tuning a Transformer for the task. [Hugging Face](https://huggingface.co/docs/transformers/v4.40.1/tasks/token_classification?utm_source=chatgpt.com)

## 29.2 Transformer POS Tagging

- Token-level labels
- Dataset preparation
- Alignment
- Evaluation

## 29.3 Extractive Question Answering

Understand:

```text
Question + Context
        ↓
Transformer
        ↓
Start position
+
End position
        ↓
Answer span
```

## 29.4 QA Project

Fine-tune a compact pretrained encoder on an extractive QA dataset.

Hugging Face's current QA workflow uses a pretrained Transformer, question/context tokenization, span prediction and task-specific evaluation. [Hugging Face](https://huggingface.co/docs/transformers/main/en/tasks/question_answering?utm_source=chatgpt.com)

### Project

**Transformer NER + QA Mini Portfolio**

---

# MODULE 29 — AIRBNB CUSTOMER REVIEW SUMMARIZATION — PART 1

### 🚀 GUVI PROJECT

## 30.1 Problem

Build a system that summarizes Airbnb customer reviews for decision support.

## 30.2 Dataset Analysis

Study:

- Review text
- Reference summaries
- Train/validation/test split
- Text lengths
- Noise
- Duplicates
- Outliers

## 30.3 Summarization Types

### Extractive

Select important existing sentences.

### Abstractive

Generate a new summary.

## 30.4 Preprocessing

```text
Review
 ↓
Cleaning
 ↓
Tokenizer
 ↓
Truncation
 ↓
Input IDs
 ↓
Attention Mask
```

## 30.5 Model

Use an encoder-decoder pretrained Transformer appropriate for summarization.

Understand:

- Encoder
- Decoder
- Cross-attention
- Generation
- Decoder labels

Modern Hugging Face summarization workflows use encoder-decoder models such as T5 and tokenize both documents and target summaries separately. [Hugging Face](https://huggingface.co/docs/transformers/v4.26.0/en/tasks/summarization?utm_source=chatgpt.com)

---

# MODULE 30 — AIRBNB CUSTOMER REVIEW SUMMARIZATION — PART 2

## 31.1 Fine-Tuning

Configure:

- Learning rate
- Batch size
- Epochs
- Gradient accumulation where necessary
- Checkpointing
- Validation

## 31.2 Generation

Implement and compare:

- Greedy decoding
- Beam search
- Length limits
- Minimum length
- Repetition controls

## 31.3 Evaluation

Use:

- ROUGE
- Relevance
- Fluency
- Factuality
- Human inspection

## 31.4 Error Analysis

Inspect:

- Hallucinated details
- Missing important information
- Repetition
- Very short outputs
- Very long outputs
- Poor handling of noisy reviews

## 31.5 Project Submission

Deliver:

- Dataset report
- Preprocessing pipeline
- Model architecture
- Training configuration
- Evaluation results
- Example predictions
- Error analysis
- Limitations
- README
- Demo

### 🚀 GUVI PROJECT CHECKPOINT

**Airbnb Customer Review Summarization — completed portfolio project**

---

# MODULE 31 — MACHINE TRANSLATION & MULTILINGUAL NLP

## 32.1 Translation as Seq2Seq

```text
Source Language
 ↓
Tokenizer
 ↓
Encoder
 ↓
Cross-Attention
 ↓
Decoder
 ↓
Target Language
```

## 32.2 Multilingual Concepts

- Language IDs
- Shared/subword vocabularies
- Cross-language representations
- Code-switching
- Morphological differences
- Low-resource languages

## 32.3 Generation

- Greedy decoding
- Beam search
- Length control
- Repetition
- Unknown/rare-word behaviour

## 32.4 Evaluation

- BLEU
- Human evaluation
- Adequacy
- Fluency
- Translation error categories

## 32.5 Practical Exercise

Build a small two-language translation system using a pretrained encoder-decoder Transformer.

Translation is treated as a sequence-to-sequence task in current Hugging Face documentation, with encoder-decoder models such as T5/BART supporting the workflow. [Hugging Face](https://huggingface.co/docs/transformers/main/tasks_explained?utm_source=chatgpt.com)

---

# MODULE 32 — MICROSOFT TRANSLATOR: MULTILINGUAL CHATBOT

### 🚀 GUVI PROJECT

## 33.1 Project Architecture

```text
User Input
 ↓
Language Detection
 ↓
Translation / NLP Processing
 ↓
Response Generation
 ↓
Target-Language Response
```

## 33.2 Model Layer

Understand:

- Encoder
- Decoder
- Cross-attention
- Generation

## 33.3 Multilingual Layer

Implement:

- Language identification
- Language-specific preprocessing
- Translation routing
- Target-language output

## 33.4 Evaluation

Measure:

- BLEU
- Fluency
- Adequacy
- Translation errors
- Latency

## 33.5 Chatbot Layer

Add:

- Input validation
- Conversation state
- Error handling
- Unsupported-language handling
- User-facing response formatting

## 33.6 Presentation

Explain:

1. Problem
2. Architecture
3. Languages
4. Dataset/API
5. Tokenization
6. Model
7. Translation flow
8. Evaluation
9. Limitations
10. Real-world applications

### 🚀 PROJECT CHECKPOINT

**Microsoft Translator — Multilingual Chatbot**

GUVI currently lists the Microsoft Translator multilingual chatbot among its portfolio projects. [Guvi](https://www.guvi.in/mlp/ai-ml-student-pro?utm_source=chatgpt.com)

---

# MODULE 33 — SEMANTIC NLP & SENTENCE EMBEDDINGS

## 34.1 Word vs Sentence Representations

```text
Word2Vec
 ↓
Word representation

Sentence Transformer
 ↓
Sentence representation
```

## 34.2 Semantic Similarity

```text
Sentence A
Sentence B
      ↓
Embeddings
      ↓
Cosine Similarity
```

## 34.3 Applications

- Semantic similarity
- Duplicate detection
- Clustering
- Recommendation
- Semantic search

## 34.4 Practical Exercise

Use a pretrained sentence-embedding model to:

- Encode documents
- Calculate similarity
- Find nearest sentences
- Cluster related text
- Detect duplicates

Sentence Transformers documentation supports pretrained sentence embeddings and semantic textual similarity workflows. [sbert.net](https://www.sbert.net/examples/sentence_transformer/training/sts/?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥 Should Know**

---

# MODULE 34 — COMPUTER VISION FUNDAMENTALS

## 35.1 Introduction to Computer Vision

- What is Computer Vision?
- Image understanding
- Image classification
- Object detection
- Segmentation
- Recognition
- Localization

## 35.2 Image Representation

- Pixel
- Height
- Width
- Channels
- Grayscale
- RGB
- RGBA
- Color spaces
- Image tensors
- Batch dimension
- Data types
- Normalization

## 35.3 Image Tensor Layouts

Understand:

```text
H × W × C
```

and:

```text
C × H × W
```

## 35.4 Vision Dataset Structure

- Images
- Labels
- Bounding boxes
- Masks
- Train/validation/test
- Class imbalance
- Annotation quality

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 35 — IMAGE OPERATIONS

## 36.1 OpenCV & Pillow

Learn practical image I/O:

- Load
- Save
- Display
- Convert
- Inspect metadata

## 36.2 Geometric Operations

- Resize
- Crop
- Rotate
- Flip
- Translate
- Affine transformation
- Perspective transformation

## 36.3 Tensor Operations

- Image → tensor
- Tensor → image
- Normalization
- Batch creation

## 36.4 Data Augmentation

- Horizontal flip
- Rotation
- Random crop
- Resize/crop
- Color transformations

## 36.5 Mini Project

Build an **Image Preprocessing Toolkit** that accepts an image folder and produces standardized, augmented datasets.

OpenCV's official image-processing material includes smoothing, morphology, transformations, thresholding, contours and related operations. [OpenCV Documentation](https://docs.opencv.org/4.10.0/d7/da8/tutorial_table_of_content_imgproc.html?utm_source=chatgpt.com)

---

# MODULE 36 — IMAGE ENHANCEMENT

## 37.1 Intensity Operations

- Brightness
- Contrast
- Dynamic range
- Clipping

## 37.2 Histograms

- Histogram
- Histogram interpretation
- Histogram equalization
- Contrast enhancement
- Histogram comparison

## 37.3 Noise

- Gaussian noise
- Salt-and-pepper noise
- Sensor noise
- Noise reduction

## 37.4 Filtering

- Gaussian filtering
- Median filtering
- Mean filtering
- Bilateral filtering
- Sharpening

## 37.5 Practical Lab

Build an image-enhancement pipeline and compare:

```text
Original
 ↓
Noise
 ↓
Denoising
 ↓
Contrast Enhancement
 ↓
Sharpening
```

### Project

**Document/Image Enhancement Tool**

---

# MODULE 37 — IMAGE PROCESSING

## 38.1 Edge Detection

- Image gradients
- Sobel
- Laplacian
- Canny

## 38.2 Thresholding

- Binary thresholding
- Inverse thresholding
- Adaptive thresholding
- Otsu concept

OpenCV describes thresholding as a basic segmentation technique for separating foreground regions from background according to intensity. [OpenCV Documentation](https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html?utm_source=chatgpt.com)

## 38.3 Morphology

- Erosion
- Dilation
- Opening
- Closing
- Morphological gradients

## 38.4 Contours

- Contour detection
- Bounding boxes
- Convex hull
- Image moments
- Shape measurements

## 38.5 Classical Segmentation Awareness

- Watershed
- Distance transform
- Connected components

## 38.6 OCR Awareness

Understand:

- Text regions
- Preprocessing
- Thresholding
- Character recognition pipeline

### Project

**Automatic Document/Image Processing Pipeline**

---

# MODULE 38 — IMAGE CLASSIFICATION & CV DATA PIPELINES

## 39.1 Image Classification

- Classes
- Labels
- Image tensors
- Classification heads
- Training/inference pipeline

## 39.2 Dataset Engineering

- Train/validation/test
- Stratification
- Class imbalance
- Duplicate detection
- Leakage prevention
- Annotation quality

## 39.3 Augmentation Strategy

Compare:

- No augmentation
- Mild augmentation
- Aggressive augmentation

## 39.4 Evaluation

Reuse existing DL knowledge:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix

Add CV-specific analysis:

- Per-class performance
- Misclassified images
- Dataset bias
- Failure cases

## 39.5 Project

Build a **Custom Image Classification System** using a CNN already learned in the Deep Learning syllabus.

PyTorch's official vision material similarly uses datasets, transforms, normalization, augmentation and pretrained networks as the practical image-training workflow. [PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial?utm_source=chatgpt.com)

---

# MODULE 39 — TRANSFER LEARNING FOR COMPUTER VISION

## 40.1 Why Transfer Learning?

- Limited datasets
- Expensive training
- Pretrained representations
- Feature extraction
- Fine-tuning

## 40.2 Feature Extraction

```text
Pretrained Backbone
 ↓
Freeze Backbone
 ↓
New Classification Head
 ↓
Train
```

## 40.3 Fine-Tuning

```text
Pretrained Backbone
 ↓
Unfreeze selected layers
 ↓
Small Learning Rate
 ↓
Task-Specific Training
```

## 40.4 Models

Practical use of:

- ResNet
- MobileNet
- EfficientNet
- ConvNeXt

## 40.5 Model Comparison

Measure:

- Accuracy
- Parameters
- Model size
- Inference time
- Memory
- Dataset requirements

PyTorch's official transfer-learning tutorial explicitly presents both fine-tuning a pretrained ConvNet and using a pretrained network as a fixed feature extractor. [PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 40 — OBJECT DETECTION FUNDAMENTALS

## 41.1 Detection vs Classification

Classification:

```text
Image → Class
```

Detection:

```text
Image → Class + Bounding Box
```

## 41.2 Bounding Boxes

- `(x, y, width, height)`
- `(x1, y1, x2, y2)`
- Box coordinates
- Image normalization

## 41.3 Localization

- Object center
- Width/height
- Class prediction
- Confidence score

## 41.4 Intersection over Union

Understand:

\[
IoU = \frac{Area\ of\ Intersection}{Area\ of\ Union}
\]

## 41.5 Non-Maximum Suppression

Understand:

```text
Candidate Boxes
 ↓
Confidence Ranking
 ↓
IoU Comparison
 ↓
Suppress Duplicates
 ↓
Final Detections
```

## 41.6 Detection Metrics

- Precision
- Recall
- AP
- mAP
- IoU thresholds

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 41 — OBJECT DETECTION ARCHITECTURES

## 42.1 Two-Stage Detectors

Understand:

- R-CNN family
- Fast R-CNN
- Faster R-CNN
- Region proposals
- Detection heads

## 42.2 Single-Stage Detectors

Understand conceptually:

- SSD
- RetinaNet
- YOLO family

## 42.3 Modern Detector Awareness

- Feature pyramids
- Anchor-based vs anchor-free concepts
- Transformer-based detection
- DETR

Stanford CS231n's current CV curriculum covers single-stage and two-stage detectors and includes YOLO, Faster R-CNN and DETR in its detection material. [CS231n](https://cs231n.stanford.edu/2025/schedule.html?utm_source=chatgpt.com)

## 42.4 Practical Model Selection

Choose based on:

- Accuracy
- Latency
- Model size
- Hardware
- Dataset size
- Real-time requirement

### Priority

**🔥🔥 Must Know conceptually**

---

# MODULE 42 — OBJECT DETECTION IMPLEMENTATION

## 43.1 Dataset Preparation

Create:

- Images
- Class labels
- Bounding-box annotations
- Train/validation split

## 43.2 Training Pipeline

```text
Dataset
 ↓
Transforms
 ↓
Backbone
 ↓
Detection Head
 ↓
Loss
 ↓
Training
 ↓
Validation
```

## 43.3 Pretrained Detection Models

Use practical pretrained models available through the PyTorch vision ecosystem.

TorchVision currently provides pretrained object-detection models including Faster R-CNN, FCOS, RetinaNet, SSD and SSDlite. [PyTorch Documentation](https://docs.pytorch.org/vision/master/models.html?utm_source=chatgpt.com)

## 43.4 Evaluation

Calculate:

- IoU
- Precision
- Recall
- AP
- mAP

## 43.5 Project

**Custom Object Detection System**

Deliver:

- Dataset
- Annotation format
- Training code
- Evaluation
- Visualized predictions
- Failure analysis
- Saved model

---

# MODULE 43 — OBJECT SEGMENTATION

## 44.1 Why Segmentation?

Detection:

```text
Bounding Box
```

Segmentation:

```text
Pixel-Level Mask
```

## 44.2 Semantic Segmentation

Every pixel receives a class.

## 44.3 Instance Segmentation

Every object instance receives its own mask.

## 44.4 Mask Representation

- Binary masks
- Multi-class masks
- Instance IDs

## 44.5 Metrics

- IoU
- Dice score
- Pixel accuracy
- Mean IoU

## 44.6 Architectures

Understand:

- FCN
- U-Net
- DeepLab
- Mask R-CNN

TorchVision currently provides pretrained semantic-segmentation models such as FCN and DeepLabV3 and an instance-segmentation implementation based on Mask R-CNN. [PyTorch Documentation](https://docs.pytorch.org/vision/master/models.html?utm_source=chatgpt.com)

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 44 — SEGMENTATION IMPLEMENTATION

## 45.1 U-Net Implementation

Build:

```text
Encoder
 ↓
Bottleneck
 ↓
Decoder
 ↓
Segmentation Mask
```

Understand:

- Encoder features
- Skip connections
- Upsampling
- Pixel-level predictions

## 45.2 Dataset

- Image-mask pairs
- Mask preprocessing
- Class encoding
- Augmentation consistency

## 45.3 Training

- Segmentation loss
- Validation
- Checkpointing
- Visualization

## 45.4 Evaluation

Compare:

- IoU
- Dice
- Pixel accuracy
- Per-class performance

## 45.5 Project

**Road/Building/Object Segmentation**

Produce:

- Input image
- Ground-truth mask
- Predicted mask
- Error visualization

---

# MODULE 45 — FACE DETECTION, LANDMARKS & ALIGNMENT

## 46.1 Face Detection

Understand:

- Face bounding boxes
- Detection confidence
- Multiple faces
- False positives

## 46.2 Facial Landmarks

- Eyes
- Nose
- Mouth
- Face geometry

## 46.3 Face Alignment

Understand why alignment improves recognition:

```text
Detected Face
 ↓
Landmarks
 ↓
Geometric Alignment
 ↓
Normalized Face
 ↓
Recognition
```

## 46.4 Preprocessing

- Cropping
- Resizing
- Normalization
- Illumination handling

## 46.5 Practical Lab

Build a webcam/image-based face detection and landmark visualization system.

### Priority

**🔥🔥🔥 Must Know**

---

# MODULE 46 — FACIAL RECOGNITION

## 47.1 Recognition Pipeline

```text
Image
 ↓
Face Detection
 ↓
Face Alignment
 ↓
Face Representation
 ↓
Embedding
 ↓
Similarity
 ↓
Decision
```

## 47.2 Face Embeddings

Understand:

- Embedding vectors
- Distance
- Cosine similarity
- Euclidean distance

## 47.3 Verification

Question:

> "Are these two faces the same person?"

## 47.4 Identification

Question:

> "Which enrolled identity does this face correspond to?"

## 47.5 Thresholds

Understand:

- Genuine pairs
- Impostor pairs
- Similarity threshold
- False acceptance
- False rejection
- ROC-style analysis

## 47.6 Security Awareness

Understand:

- Spoofing
- Presentation attacks
- Lighting variation
- Pose variation
- Demographic/data bias
- Privacy considerations

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 47 — APPLE FACE ID / FACIAL RECOGNITION PROJECTS

## 48.1 GUVI Project — Apple Face ID: Facial Recognition Security

Build a simplified face-authentication pipeline.

## 48.2 Project Components

```text
Camera/Image
 ↓
Face Detection
 ↓
Alignment
 ↓
Embedding
 ↓
Enrollment
 ↓
Similarity
 ↓
Threshold
 ↓
Authentication
```

## 48.3 Security Evaluation

Test:

- Same-person images
- Different-person images
- Different lighting
- Different poses
- Different image quality

## 48.4 Facial Recognition System — Part 2

Extend the project with:

- Multiple enrolled identities
- Verification
- Identification
- Threshold calibration
- Error analysis

## 48.5 Project Submission

Include:

- Architecture
- Dataset
- Enrollment procedure
- Recognition method
- Threshold selection
- FAR/FRR-style analysis
- Security limitations
- Privacy considerations

GUVI currently lists **Apple Face ID: Facial Recognition Security** as a portfolio project. [Guvi](https://www.guvi.in/mlp/ai-ml-student-pro?utm_source=chatgpt.com)

---

# MODULE 48 — LANDMARK RECOGNITION

## 49.1 Landmark Recognition

Treat landmark recognition as an image-classification problem:

```text
Image
 ↓
Preprocessing
 ↓
Feature Extractor
 ↓
Classifier
 ↓
Landmark Class
```

## 49.2 Dataset

Study:

- Landmark categories
- Class imbalance
- Similar-looking landmarks
- Geographic variation
- Image quality

## 49.3 Transfer Learning

Use a pretrained vision backbone.

Compare:

- ResNet
- EfficientNet
- MobileNet
- ConvNeXt

## 49.4 Error Analysis

Inspect:

- Similar landmarks
- Occlusion
- Weather
- Night images
- Unusual viewpoints

## 49.5 Project

**Google Street View — Landmark Recognition**

Build the classification system and prepare the project submission.

GUVI identifies Google Street View landmark recognition among the Computer Vision project progression requested for this curriculum. Pasted text(20261020-190629)

---

# MODULE 49 — MODERN COMPUTER VISION: VISION TRANSFORMERS

## 50.1 CNN → Transformer Transition

Understand:

```text
CNN
 ↓
Local receptive fields
 ↓
Hierarchical visual features
```

versus:

```text
Vision Transformer
 ↓
Image patches
 ↓
Patch embeddings
 ↓
Self-attention
 ↓
Global relationships
```

## 50.2 Vision Transformer

- Image patches
- Patch embeddings
- Positional embeddings
- Self-attention
- Multi-head attention
- Transformer encoder
- Classification head

## 50.3 CNN vs ViT

Compare:

- Inductive bias
- Local vs global processing
- Data requirements
- Compute
- Transfer learning
- Inference

## 50.4 Modern Vision Awareness

Briefly understand:

- Swin Transformer
- Hierarchical vision Transformers
- CNN/Transformer hybrids
- Pretrained vision models

Do not turn this into a second Transformer mathematics course.

Stanford CS231n's current curriculum explicitly connects CNNs, object detection, segmentation and Transformers/ViT as part of modern vision progression. [CS231n](https://cs231n.stanford.edu/2025/schedule.html?utm_source=chatgpt.com)

---

# MODULE 50 — COMPUTER VISION MODEL BUILDING, EVALUATION & ERROR ANALYSIS

## 51.1 Complete CV Workflow

```text
Problem
 ↓
Dataset
 ↓
Annotation
 ↓
Preprocessing
 ↓
Augmentation
 ↓
Baseline
 ↓
Model
 ↓
Training
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Improvement
```

## 51.2 Classification Evaluation

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- Per-class metrics

## 51.3 Detection Evaluation

- IoU
- Precision/Recall
- AP
- mAP
- Confidence thresholds
- NMS behaviour

## 51.4 Segmentation Evaluation

- IoU
- Dice
- Mean IoU
- Pixel accuracy

## 51.5 Model Comparison

Compare:

| Dimension | Model A | Model B |
|---|---|---|
| Accuracy | | |
| Parameters | | |
| Model size | | |
| Inference time | | |
| Memory | | |
| Failure cases | | |

## 51.6 Error Analysis

Investigate:

- Bad labels
- Distribution shift
- Class imbalance
- Occlusion
- Lighting
- Resolution
- Background bias
- False positives
- False negatives

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 51 — TESLA AUTOPILOT CLONE — OBJECT DETECTION

### 🚗 GUVI CAPSTONE PROJECT

## 52.1 Perception Problem

Build a simplified autonomous-driving **perception** system focused on object detection.

This is **not** an autonomous-driving planning/control course.

## 52.2 Input

- Road images
- Dashcam-style frames
- Vehicle/pedestrian/object annotations

## 52.3 Detection Pipeline

```text
Road Image
 ↓
Preprocessing
 ↓
Object Detector
 ↓
Bounding Boxes
 ↓
Confidence
 ↓
NMS
 ↓
Detected Road Objects
```

## 52.4 Objects

Depending on dataset:

- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles
- Pedestrians
- Traffic-related objects

## 52.5 Evaluation

Measure:

- IoU
- Precision
- Recall
- AP
- mAP
- Per-class performance
- False detections

## 52.6 Perception Constraints

Study conceptually:

- Real-time inference
- Camera perspective
- Small objects
- Occlusion
- Motion blur
- Lighting
- Weather
- Domain shift

## 52.7 Project Submission

Deliver:

- Dataset description
- Annotation format
- Model architecture
- Training pipeline
- Detection examples
- Metrics
- Failure analysis
- Inference speed
- Limitations
- Demo

### 🚀 GUVI PROJECT

**Tesla Autopilot Clone — Object Detection**

**Tesla Autopilot Clone — Project Submission**

The project is explicitly included in the requested GUVI project set and is consistent with GUVI's broader emphasis on practical Computer Vision projects. Pasted text(20261020-190629)

---

# MODULE 52 — AI APPLICATION ENGINEERING

## 53.1 Project Structure

Learn to organize:

```text
project/
├── src/
├── data/
├── models/
├── notebooks/
├── tests/
├── configs/
├── scripts/
├── requirements.txt
└── README.md
```

## 53.2 Python Environment Management

- Virtual environments
- Dependencies
- `requirements.txt`
- Reproducible installs
- Environment variables

## 53.3 Git & GitHub

- Repository creation
- Branches
- Commits
- `.gitignore`
- README
- Tags/releases
- Portfolio organization

## 53.4 Configuration

Separate:

- Hyperparameters
- Paths
- Model settings
- Environment configuration

## 53.5 Testing

Basic:

- Unit tests
- Data validation tests
- Preprocessing tests
- Inference tests

### 🔥 Priority

**🔥🔥🔥 Must Know**

---

# MODULE 53 — MODEL SERVING WITH FASTAPI & STREAMLIT

## 54.1 Inference Pipeline

```text
User Input
 ↓
Validation
 ↓
Preprocessing
 ↓
Model
 ↓
Postprocessing
 ↓
Response
```

## 54.2 FastAPI

Learn:

- REST endpoints
- Request schemas
- Response schemas
- Input validation
- Error handling
- Model loading at startup

## 54.3 NLP API

Example endpoints:

```text
POST /predict
POST /classify
POST /summarize
POST /translate
```

## 54.4 CV API

Example:

```text
POST /detect
POST /classify
POST /segment
```

## 54.5 Streamlit

Build a user-facing demo for:

- NLP classification
- Summarization
- Image classification
- Object detection

### Project

**Deploy One NLP + One CV Model Behind APIs**

---

# MODULE 54 — DOCKER & DEPLOYMENT

## 55.1 Why Containers?

Understand:

- Reproducibility
- Dependencies
- Runtime isolation
- Portable deployment

## 55.2 Docker Fundamentals

Learn:

- Dockerfile
- Image
- Container
- Ports
- Environment variables
- Volumes
- `.dockerignore`

Docker's current Python guide emphasizes packaging an application together with its dependencies, configuration and runtime into a portable container image. [Docker Documentation](https://docs.docker.com/guides/python/?utm_source=chatgpt.com)

## 55.3 Containerize AI Models

Build:

```text
Model
 +
Preprocessing
 +
API
 +
Dependencies
        ↓
Docker Image
        ↓
Container
        ↓
REST API
```

## 55.4 Basic Cloud Deployment

Learn the concepts necessary to deploy a containerized inference application to a managed cloud/container service.

Do **not** turn this into a cloud-architecture specialization.

## 55.5 Deployment Project

Deploy one of:

- Sentiment classifier
- Summarization model
- Image classifier
- Object detector

---

# MODULE 55 — PRACTICAL MLOPS

## 56.1 Experiment Tracking

Track:

- Parameters
- Metrics
- Dataset version
- Code version
- Model artifacts
- Training duration

## 56.2 Reproducibility

Use:

- Random seeds
- Configuration files
- Dependency locking
- Versioned datasets
- Versioned models

## 56.3 MLflow

Learn practical:

- Experiments
- Runs
- Metrics
- Parameters
- Artifacts
- Model logging

MLflow Tracking currently records parameters, code versions, metrics and output artifacts for individual runs. [MLflow AI Platform](https://mlflow.org/docs/latest/tracking?utm_source=chatgpt.com)

## 56.4 Model Registry

Understand:

- Model versions
- Tags
- Aliases
- Lineage
- Promotion workflow

MLflow's Model Registry provides versioning, lineage, aliases and metadata for managing models through their lifecycle. [MLflow AI Platform](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/?utm_source=chatgpt.com)

## 56.5 Deployment Awareness

Connect:

```text
Experiment
 ↓
Best Model
 ↓
Registry
 ↓
Validation
 ↓
Deployment
 ↓
Monitoring
```

### 🔥 Priority

**🔥🔥 Should Know**

---

# MODULE 56 — PRODUCTION INFERENCE, MONITORING & MODEL QUALITY

## 57.1 Inference Performance

Measure:

- Latency
- Throughput
- Batch size
- Memory usage
- Model loading time

## 57.2 Model Monitoring

Track:

- Prediction distributions
- Input distributions
- Error rates
- Latency
- Failure rates

## 57.3 Data Drift Awareness

Understand:

- Training distribution
- Production distribution
- Feature drift
- Image distribution changes
- Text distribution changes

## 57.4 Model Quality Monitoring

For NLP:

- Classification quality
- Translation quality
- Summarization quality

For CV:

- Detection quality
- Classification quality
- Segmentation quality

## 57.5 Failure Handling

Implement:

- Input validation
- Graceful errors
- Logging
- Timeouts
- Invalid-image handling
- Empty-text handling
- Model-loading failures

### Goal

Know enough to answer:

> "What happens after the model goes into production?"

---

# MODULE 57 — END-TO-END AI PROJECT ENGINEERING

## 58.1 Complete Lifecycle

Build the full pipeline:

```text
Problem Definition
 ↓
Data Collection
 ↓
Data Validation
 ↓
Preprocessing
 ↓
Baseline
 ↓
Model
 ↓
Training
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Experiment Tracking
 ↓
Model Versioning
 ↓
API
 ↓
Docker
 ↓
Deployment
 ↓
Monitoring
```

## 58.2 Choose One Major Project

Choose one:

### NLP

**Production Sentiment / Document Classification System**

or

### NLP

**Production Summarization System**

or

### CV

**Production Object Detection System**

## 58.3 Required Deliverables

- Problem statement
- Dataset documentation
- Data pipeline
- Baseline
- Model
- Experiments
- Evaluation
- Error analysis
- Model artifact
- API
- Docker image
- Deployment
- README
- Architecture diagram
- Demo

## 58.4 Portfolio Standard

The project must be understandable to someone who has **never seen the notebook**.

---

# MODULE 58 — AI/ML INTERVIEW + PORTFOLIO READINESS

## 59.1 Python

Revise:

- Data structures
- Functions
- OOP basics
- NumPy
- Pandas
- File handling
- Exceptions
- Iteration/vectorization

## 59.2 SQL

Practical fundamentals:

- SELECT
- WHERE
- GROUP BY
- HAVING
- JOINs
- Subqueries
- CTEs
- Window functions
- Aggregations

## 59.3 ML Interview Topics

Be able to explain:

- Bias vs variance
- Overfitting
- Regularization
- Cross-validation
- Data leakage
- Class imbalance
- Evaluation metrics
- Feature engineering

## 59.4 Deep Learning Interview Topics

Explain:

- Backpropagation
- Optimizers
- Learning rate
- CNNs
- Transfer learning
- RNN/LSTM/GRU
- Attention
- Transformers

## 59.5 NLP Interview Topics

Explain:

- BoW
- TF-IDF
- Word2Vec
- Static vs contextual embeddings
- Tokenization
- BPE
- WordPiece
- BERT
- NER
- Seq2Seq
- Attention
- Transformer
- Fine-tuning
- BLEU
- ROUGE
- Perplexity

## 59.6 Computer Vision Interview Topics

Explain:

- Image tensors
- Convolution
- Transfer learning
- IoU
- NMS
- AP/mAP
- Object detection
- Semantic segmentation
- Instance segmentation
- Face recognition
- Embeddings
- Vision Transformers

## 59.7 Project Defense

For every major project, answer:

1. What problem did you solve?
2. Why this dataset?
3. What preprocessing did you use?
4. Why this model?
5. What baseline did you compare against?
6. What metric did you use?
7. What were the biggest errors?
8. What did you change after error analysis?
9. How would you deploy it?
10. What are its limitations?

---

# MODULE 59 — FINAL AI/ML PORTFOLIO CAPSTONE

## 60.1 Portfolio Architecture

Your final portfolio should demonstrate **multiple areas**, not six versions of the same classifier.

### NLP

**Project 1 — Classical NLP**

TF-IDF + classification.

**Project 2 — Transformer NLP**

BERT-based classification/NER.

**Project 3 — Airbnb Customer Review Summarization**

Encoder-decoder Transformer.

**Project 4 — Microsoft Translator**

Multilingual translation/chatbot.

### Computer Vision

**Project 5 — Image Processing**

OpenCV enhancement/processing pipeline.

**Project 6 — Image Classification**

Transfer-learning classifier.

**Project 7 — Facial Recognition**

Face detection → alignment → embeddings → verification.

**Project 8 — Google Street View Landmark Recognition**

Transfer-learning landmark classifier.

**Project 9 — Tesla Autopilot Clone**

Object detection perception system.

## 60.2 Final Capstone Integration

Select **one** project and take it all the way:

```text
Dataset
 ↓
Data Preparation
 ↓
Baseline
 ↓
Model Development
 ↓
Experiment Tracking
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Model Selection
 ↓
Model Serialization
 ↓
FastAPI
 ↓
Streamlit
 ↓
Docker
 ↓
Deployment
 ↓
Monitoring
```

## 60.3 Final Technical Documentation

Produce:

- `README.md`
- Architecture diagram
- Dataset documentation
- Model card
- Installation instructions
- Training instructions
- Evaluation report
- Inference instructions
- API documentation
- Deployment instructions
- Known limitations
- Future improvements

## 60.4 Final GitHub Portfolio

Each major project should contain:

```text
project/
├── README.md
├── src/
├── data/
├── configs/
├── notebooks/
├── models/
├── tests/
├── requirements.txt
├── Dockerfile
└── demo/
```

## 60.5 Final Job-Readiness Check

You should be able to demonstrate:

### Machine Learning

- Classical ML
- Evaluation
- Feature engineering
- End-to-end projects

### Deep Learning

- PyTorch
- Training
- CNNs
- RNN/LSTM/GRU
- Transfer learning
- Experimentation

### NLP

- Text processing
- BoW
- TF-IDF
- Embeddings
- Sequence NLP
- Seq2Seq
- Attention
- Transformers
- BERT
- Subword tokenization
- Hugging Face
- Fine-tuning
- NER
- Summarization
- Translation
- NLP evaluation

### Computer Vision

- Image operations
- Image enhancement
- Image processing
- OpenCV
- Image classification
- Transfer learning
- Object detection
- IoU
- NMS
- mAP
- Segmentation
- Facial recognition
- Landmark recognition
- Modern vision models
- Perception systems

### Engineering

- Git/GitHub
- APIs
- FastAPI
- Streamlit
- Docker
- Deployment
- Experiment tracking
- Model versioning
- Basic monitoring
- Practical MLOps

### Portfolio

- Cardeko Car Price Prediction
- Airbnb Customer Review Summarization
- Microsoft Translator — Multilingual Chatbot
- Apple Face ID / Facial Recognition Security
- Facial Recognition System
- Google Street View — Landmark Recognition
- Tesla Autopilot Clone — Object Detection

The resulting progression deliberately keeps the existing NLP Modules 1–22 intact, then uses GUVI's project-oriented NLP/CV progression while filling the important gaps identified by the other syllabi and current university/framework curricula. GUVI's current AI/ML offering continues to emphasize NLP, Computer Vision and practical portfolio projects, while Stanford CS224N/CS231N and current PyTorch/Hugging Face documentation support the deeper Transformer, transfer-learning, detection, segmentation and evaluation progression used above. [Guvi](https://www.guvi.in/mlp/ai-ml-student-pro?utm_source=chatgpt.com)

**Hard boundary preserved:** this 60-module curriculum does **not** turn into an LLM/RAG specialization. There is no RAG, vector-database curriculum, LangChain, LlamaIndex, agentic AI, LoRA/QLoRA, RLHF, DPO, instruction-tuning specialization, or LLM application architecture.