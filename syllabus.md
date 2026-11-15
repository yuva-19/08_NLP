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


# **MODULE 34 — HUMAN SPEECH, AUDIO & DIGITAL SIGNAL PROCESSING**

Foundation — Must complete

**Objective:** Understand how humans produce speech, how sound travels, how microphones capture it, and how computers represent and process audio.

## **34.1 — Introduction to Audio & Speech Processing**

Understand:

- What is sound?
- What is audio?
- What is speech?
- Audio processing vs speech processing
- Digital audio processing
- Speech recognition
- Speech synthesis
- Speech-to-text
- Text-to-speech
- Audio classification
- Speech emotion recognition
- Speaker recognition
- Speaker verification
- Real-world applications of speech processing

## **34.2 — Fundamentals of Sound Waves**

Understand the physical characteristics of sound.

- Sound wave formation
- Mechanical waves
- Longitudinal waves
- Compression and rarefaction
- Amplitude
- Frequency
- Time period
- Wavelength
- Phase
- Speed of sound
- Relationship between frequency, wavelength and velocity
- Frequency and perceived pitch
- Amplitude and perceived loudness
- Decibels (dB)
- Sound pressure level (SPL)
- Human hearing frequency range
- Audible, infrasonic and ultrasonic frequencies

## **34.3 — Human Speech Production & Vocal Anatomy**

Important — Missing from original syllabus

Understand how the human body produces speech.

**A. Respiratory system**

- Lungs
- Diaphragm
- Airflow and air pressure
- Pulmonic egressive airflow
- Role of breathing in speech production

**B. Phonatory system**

- Larynx
- Vocal folds (vocal cords)
- Glottis
- Vocal fold vibration
- Voiced and unvoiced sounds
- Fundamental frequency (F0)
- Vocal fold tension
- Vocal fold vibration rate

**C. Articulatory system**

- Tongue
- Lips
- Teeth
- Hard palate
- Soft palate (velum)
- Alveolar ridge
- Nasal cavity
- Oral cavity
- Jaw movement

**D. Speech production mechanism**

- How airflow becomes sound
- How vocal folds generate voiced sounds
- How unvoiced sounds are produced
- Articulation
- Place of articulation
- Manner of articulation
- Vowels and consonants
- Nasal and oral sounds

**E. Source-filter theory**

- Glottal source
- Vocal tract filter
- Resonance
- Formants
- F1, F2 and F3
- Relationship between vocal tract shape and speech characteristics

## **34.4 — Introduction to Phonetics & Phonology**

Understand the linguistic foundations of speech.

- What is phonetics?
- What is phonology?
- Articulatory phonetics
- Acoustic phonetics
- Auditory phonetics
- Phoneme
- Allophone
- Phone
- Grapheme
- Syllables
- Consonants
- Vowels
- Diphthongs
- Monophthongs
- Voicing
- International Phonetic Alphabet (IPA)
- Phoneme vs grapheme
- Phoneme vs syllable
- Speech sounds vs written characters

**Practical understanding:**

- Identify vowels and consonants.
- Compare voiced and unvoiced sounds.
- Understand how different pronunciations produce different acoustic patterns.

## **34.5 — Analog & Digital Audio Signals**

Understand how physical sound becomes a digital signal.

- Analog signals
- Digital signals
- Continuous-time signals
- Discrete-time signals
- Continuous amplitude
- Discrete amplitude
- Analog-to-Digital Conversion (ADC)
- Digital-to-Analog Conversion (DAC)
- Sampling
- Quantization
- Quantization error
- Quantization noise
- Nyquist-Shannon sampling theorem
- Nyquist frequency
- Aliasing
- Anti-aliasing filters

## **34.6 — Digital Audio Parameters**

Understand the important properties of digital audio.

**A. Sampling rate**

- Sampling rate (Hz)
- Samples per second
- Sampling frequency
- Nyquist frequency
- 8 kHz
- 16 kHz
- 22.05 kHz
- 44.1 kHz
- 48 kHz
- Why speech applications commonly use 16 kHz
- Why changing a sampling-rate label is not actual resampling

**B. Bit depth**

- Quantization levels
- 8-bit audio
- 16-bit audio
- 24-bit audio
- Dynamic range
- Quantization noise

**C. Channels**

- Mono
- Stereo
- Multichannel audio
- Left and right channels
- Channel conversion

**D. Other parameters**

- Audio duration
- Bitrate
- File size
- Uncompressed vs compressed audio
- Lossless vs lossy compression

**E. Audio formats**

- WAV
- MP3
- FLAC
- AAC
- OGG
- PCM

## **34.7 — Time-Domain Signal Analysis**

Understand audio signals in their original time representation.

- What is the time domain?
- Amplitude vs time
- Audio samples
- Sample indexing
- Signal duration
- Peak amplitude
- Mean amplitude
- Absolute amplitude
- RMS (Root Mean Square)
- Signal energy
- Zero Crossing Rate (ZCR)
- Silence and non-silent regions
- Periodic and aperiodic signals
- Speech signal characteristics
- Clipping and distortion

**Visualizations:**

- Audio waveform
- Speech waveform
- Silence regions
- Clipped waveform
- Comparison of two audio waveforms

**Practical Lab 34A**

Using Python:

- Load an audio file.
- Extract audio samples.
- Play the audio.
- Plot the waveform.
- Identify silence.
- Calculate duration.
- Calculate peak amplitude.
- Calculate RMS energy.
- Calculate Zero Crossing Rate.
- Detect clipping.

## **34.8 — Frequency-Domain Signal Analysis**

Understand the frequency components present inside audio.

- Time domain vs frequency domain
- Frequency representation
- Periodic signals and frequency components
- Fourier series (conceptual)
- Fourier Transform (FT)
- Discrete Fourier Transform (DFT)
- Fast Fourier Transform (FFT)
- Frequency bins
- Frequency resolution
- Magnitude spectrum
- Phase spectrum
- Power spectrum
- Spectral energy
- Harmonics
- Fundamental frequency
- Frequency components of voiced speech

**Practical Lab 34B**

- Generate a sine wave.
- Generate signals with multiple frequencies.
- Apply FFT.
- Visualize frequency spectrum.
- Identify dominant frequencies.
- Compare different speech sounds.

## **34.9 — Short-Time Fourier Transform (STFT)**

Understand why ordinary FFT is insufficient for analysing speech that changes over time.

- Limitations of global FFT
- Non-stationary signals
- Short-time analysis
- Frame-based audio processing
- Windowing
- Window length
- Hop length
- Overlapping frames
- Hann window
- Hamming window
- STFT calculation
- Magnitude STFT
- Power STFT
- Time-frequency representation

**Important concepts:**

- Time-frequency resolution trade-off
- Frequency resolution vs time resolution
- Spectral leakage
- Windowing effects

## **34.10 — Spectrogram & Time-Frequency Visualization**

Understand how speech frequencies change over time.

- What is a spectrogram?
- STFT to spectrogram
- Magnitude spectrogram
- Power spectrogram
- Log spectrogram
- Decibel scaling
- Time axis
- Frequency axis
- Color intensity and spectral energy
- Reading spectrograms
- Identifying vowels
- Identifying consonants
- Voiced and unvoiced regions
- Formant structures
- Speech transitions

**Practical Lab 34C**

- Generate a spectrogram.
- Compare waveform and spectrogram.
- Visualize a vowel.
- Visualize a consonant.
- Identify voiced and unvoiced regions.

## **34.11 — Audio Noise & Signal Quality**

Understand the difference between useful speech information and unwanted sound.

**Types of noise**

- Background noise
- White noise
- Gaussian noise
- Pink noise (conceptual)
- Traffic noise
- Wind noise
- Microphone noise
- Electrical interference
- Reverberation
- Echo

**Signal quality**

- Signal-to-Noise Ratio (SNR)
- Signal power
- Noise power
- SNR in decibels
- Clean speech vs noisy speech
- Noise floor
- Clipping
- Distortion

**Noise reduction concepts**

- Noise suppression
- Filtering
- Spectral subtraction
- Wiener filtering (conceptual)
- Noise gates (conceptual)
- Adaptive noise cancellation (conceptual)

## **34.12 — Digital Audio Filtering**

Understand how filters modify frequency components.

- What is a digital filter?
- Filter frequency response
- Cutoff frequency
- Passband
- Stopband
- Transition band
- Filter order
- FIR filters
- IIR filters
- Low-pass filter
- High-pass filter
- Band-pass filter
- Band-stop filter
- Notch filter
- Butterworth filter
- Filtering speech signals

**Practical Lab 34D**

- Add white noise to speech.
- Design a low-pass filter.
- Design a high-pass filter.
- Apply a band-pass filter.
- Compare original and filtered recordings.
- Compare waveforms and spectrograms.
- Evaluate whether filtering improved speech intelligibility.

## **34.13 — Audio Enhancement & Restoration**

Understand how to improve poor-quality audio recordings.

- Audio normalization
- Peak normalization
- RMS normalization
- Loudness normalization (conceptual)
- Dynamic range
- Noise reduction
- Spectral subtraction
- Filtering-based enhancement
- Reverberation reduction (conceptual)
- Dereverberation (conceptual)
- Speech enhancement vs noise suppression
- Limitations of audio enhancement
- Artifacts introduced by denoising

**Practical Lab 34E**

Create a noisy speech recording and follow:

```
Original Speech
      ↓
Add Background Noise
      ↓
Listen to Noisy Speech
      ↓
Visualize Waveform
      ↓
Analyze SNR
      ↓
Apply Noise Reduction
      ↓
Listen to Enhanced Speech
      ↓
Compare Spectrograms
      ↓
Evaluate Improvement
```

## **34.14 — Python Audio Processing Environment**

Learn the essential libraries and their roles.

| Library     | Purpose                     |
| ----------- | --------------------------- |
| NumPy       | Numerical audio arrays      |
| SciPy       | Signal processing           |
| Librosa     | Audio analysis and features |
| SoundFile   | Reading and writing audio   |
| SoundDevice | Recording and playback      |
| Matplotlib  | Audio visualization         |
| IPython     | Notebook audio playback     |
| Torchaudio  | PyTorch audio processing    |

**Practical Lab 34F — Audio Analysis Notebook**

Build a notebook that accepts an audio file and displays:

- Audio playback
- File information
- Sampling rate
- Number of channels
- Duration
- Waveform
- Amplitude statistics
- RMS energy
- ZCR
- FFT frequency spectrum
- STFT
- Spectrogram
- Noise analysis
- Filtering
- Audio enhancement

Module 34 Completion Project

**Project: Digital Audio Analysis & Enhancement Toolkit**

You should be able to load an unfamiliar audio recording, inspect its properties, understand its signal, visualize its frequency content and perform basic enhancement.

---

# MODULE 35 — AUDIO REPRESENTATIONS & FEATURE ENGINEERING FOR MACHINE LEARNING

**Purpose:** Learn how to turn a speech recording into useful numerical representations for an emotion classifier.

## 35.1 — ML-Specific Audio Preparation

### Learn
- Consistent sample-rate handling and actual resampling
- Mono conversion and channel-selection considerations
- Amplitude normalization and its limitations
- Trimming and padding recordings
- Fixed-length versus variable-length inputs
- Audio segmentation and utterance boundaries
- Voice Activity Detection (VAD)
- Speech-region detection versus emotion-bearing pauses
- Handling recordings with different durations
- Reproducible preprocessing functions

### Practical Lab 35A — Preprocessing Pipeline

Create a function that:
1. Loads an audio recording.
2. Checks sample rate, channels, duration, and amplitude.
3. Converts it to the chosen channel format.
4. Resamples only when required.
5. Trims or pads it according to a chosen strategy.
6. Returns the processed waveform and metadata.

**Important:** Keep the original recording. Avoid aggressive silence removal or loudness normalization that could remove emotion-related information.

## 35.2 — Frame-Level and Utterance-Level Features

### Learn
- Frame-level features versus utterance-level features
- Short-time feature sequences
- Mean, standard deviation, minimum, maximum, and range
- Percentiles and statistical pooling
- Temporal variation across an utterance
- Feature aggregation into fixed-length vectors
- Why averaging every frame can discard emotional dynamics

### Retain and apply
- RMS energy statistics
- Short-time energy statistics
- Zero Crossing Rate (ZCR) statistics
- Duration and silence-related measurements

### Practical Lab 35B — Time-Domain Feature Vectors

For each recording:
1. Calculate frame-level RMS and ZCR.
2. Calculate utterance-level mean and standard deviation.
3. Visualize how the values change over time.
4. Store the measurements in one row per recording.

## 35.3 — Spectral Features for Emotion Recognition

### Learn
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral contrast
- Spectral flux
- Spectral flatness
- Frame-level extraction
- Utterance-level statistical aggregation
- What each feature captures and its limitations

### Priority
- **Core:** Spectral centroid, bandwidth, rolloff
- **Useful:** Spectral contrast and flux
- **Optional:** Spectral flatness

### Practical Lab 35C — Spectral Feature Extraction

Extract selected spectral features from several speech recordings. Compare their distributions across recordings and emotion labels.

## 35.4 — Mel Scale and Mel Filter Banks

### Learn
- Why human frequency perception is nonlinear
- Linear frequency scale versus Mel scale
- Hertz-to-Mel and Mel-to-Hertz conversion
- Mel filter banks
- Triangular Mel filters
- Applying Mel filters to a power spectrum
- Number of Mel bands
- What Mel scaling preserves and compresses
- Hertz spectrogram versus Mel spectrogram

**Keep the theory focused:** No need to repeat auditory anatomy in detail. Understand why Mel representations are commonly used in speech models.

## 35.5 — Mel Spectrogram and Log-Mel Spectrogram

### Learn
- Power spectrogram as input to Mel filtering
- Mel spectrogram generation
- Log-Mel transformation
- Power-to-decibel conversion
- Dynamic range compression
- Number of Mel bands
- FFT size, hop length, and window length
- How parameter choices change the representation
- Fixed-size model inputs
- Normalizing Mel features using training-set statistics

### Practical Lab 35D — Mel Feature Notebook

Generate and compare:
1. Waveform
2. STFT spectrogram
3. Mel spectrogram
4. Log-Mel spectrogram

Experiment with different Mel-band counts and frame settings.

## 35.6 — Mel-Frequency Cepstral Coefficients (MFCC)

**High priority — understand the complete extraction process.**

### Learn
- What MFCCs represent
- Cepstrum and cepstral representation
- MFCC versus Mel spectrogram
- Why MFCCs are useful as compact speech features
- Pre-emphasis (optional in modern pipelines)
- Framing and windowing (brief recap)
- Power spectrum
- Mel filter bank
- Logarithmic compression
- Discrete Cosine Transform (DCT)
- MFCC coefficient matrix
- Number of coefficients
- C0 and higher-order coefficients
- Delta MFCC
- Delta-delta MFCC
- Cepstral Mean Normalization (CMN)
- Cepstral Mean and Variance Normalization (CMVN)
- Sensitivity to noise and information loss

### MFCC Extraction Pipeline

Raw Audio
    ↓
Pre-emphasis (optional)
    ↓
Framing
    ↓
Windowing
    ↓
FFT
    ↓
Power Spectrum
    ↓
Mel Filter Banks
    ↓
Logarithm
    ↓
Discrete Cosine Transform
    ↓
MFCC Coefficients

### Practical Lab 35E — MFCC Extraction

1. Extract MFCCs from multiple speech recordings.
2. Visualize MFCC matrices.
3. Change the number of coefficients.
4. Extract delta and delta-delta coefficients.
5. Compare recordings with different emotion labels.
6. Compare frame-level MFCCs with utterance-level statistical summaries.

## 35.7 — Pitch, Prosody and Voice-Related Features

**Very important for SER.**

### Learn
- Fundamental frequency (F0) and perceived pitch
- Pitch tracking
- Voiced and unvoiced frames
- Pitch contour
- Mean, minimum, maximum, standard deviation, and range of F0
- Pitch variation over time
- RMS energy contour
- Speaking rate
- Pause duration and silence ratio
- Rhythm and timing
- Jitter and shimmer
- Harmonics-to-noise ratio (HNR)
- Formants and vocal-tract resonance

### Practical Lab 35F — Prosodic Feature Extraction

For each recording:
1. Estimate the F0 contour.
2. Identify voiced and unvoiced regions.
3. Calculate pitch statistics.
4. Calculate energy statistics.
5. Estimate pause and duration-related features.
6. Visualize pitch and energy contours together.
7. Save the features with the recording ID.

**Caution:** Pitch trackers can make octave errors and fail on unvoiced or noisy regions. Inspect estimates instead of treating every output as correct.

## 35.8 — Comparing Audio Representations

### Learn
- Waveform versus spectrogram
- Spectrogram versus Mel spectrogram
- Mel spectrogram versus Log-Mel
- Log-Mel versus MFCC
- Handcrafted features versus learned representations
- Frame-level versus utterance-level representations
- Fixed-length vectors versus variable-length sequences
- Feature concatenation
- Feature scaling and standardization
- Feature selection
- Principal Component Analysis (PCA)

### Practical Lab 35G — Reusable Feature Extractor

Build a reusable function that accepts an audio file and returns:
- MFCC statistics
- Delta and delta-delta MFCC statistics
- Pitch statistics
- RMS statistics
- Selected spectral statistics
- Duration and pause-related features
- Log-Mel representation
- Recording metadata

Save extracted features and labels in a structured file.

## 35.9 — Audio Augmentation for SER

### Learn
- Why augmentation is used
- Background-noise addition
- Small volume variations
- Time shifting
- Time stretching
- Pitch shifting
- Random cropping
- Augmentation strength and realism
- How augmentation can change perceived emotion
- Why augmentation must be applied only to training data

### Practical Lab 35H — Safe Augmentation

Create original and augmented versions of selected training recordings. Listen to them and compare pitch, duration, energy, and emotion-bearing characteristics.

### Module 35 Completion Deliverable

**Audio Feature Extraction Toolkit**

A notebook or Python module that:
- Accepts a speech recording
- Performs consistent preprocessing
- Extracts MFCC, pitch, energy, and selected spectral features
- Generates Log-Mel spectrograms
- Produces fixed-length feature vectors
- Saves features and metadata

---

# MODULE 36 — SPEECH EMOTION RECOGNITION & DATASET ENGINEERING

**Purpose:** Understand emotional vocal expression, choose a suitable dataset, and prepare a reliable training and evaluation dataset.

## 36.1 — Foundations of Speech Emotion Recognition

### Learn
- What is Speech Emotion Recognition (SER)?
- Audio-based versus text-based emotion recognition
- Speech emotion recognition versus sentiment analysis
- Speech emotion recognition versus speaker recognition
- Emotion classification and emotion intensity
- Categorical versus dimensional emotion representation
- Valence and arousal
- Emotion expression versus actual internal emotional state
- Real-world applications and limitations

**Important:** An SER model predicts patterns associated with dataset emotion labels. A voice recording alone cannot reliably reveal a person's true internal emotional state.

## 36.2 — How Emotions Affect Vocal Expression

### Learn
- How vocal-fold tension and vibration can change
- How F0 and pitch variation may change
- Changes in vocal intensity and energy
- Breathing patterns and breathiness
- Articulation and speaking rate
- Pauses, rhythm, and fluency
- Voice quality: breathiness, roughness, creakiness, and strain
- Intonation and emphasis
- Why emotional patterns vary between speakers and situations

### Practical Lab 36A — Same Sentence, Different Expressions

Record the same sentence using several intended emotional expressions.

Compare:
- Pitch contours
- RMS energy
- Speaking rate and pauses
- MFCC
- Log-Mel spectrogram
- Voice quality where measurable

## 36.3 — Emotion Categories and Acoustic Patterns

### Learn
- Neutral
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised
- Typical tendencies in pitch, energy, timing, and voice quality
- High-arousal versus low-arousal expressions
- Similarities between emotion classes
- Why neutral and subtle emotions can be difficult to classify
- Speaker, language, accent, acting style, and recording effects

**Important:** These are tendencies, not fixed rules. Do not assume that high pitch always means happiness.

## 36.4 — Emotion-Relevant Feature Groups

### A. Pitch and prosody
- Mean, minimum, maximum, and standard deviation of F0
- Pitch range and contour
- Pitch variability
- Speaking rate and pauses

### B. Energy
- RMS mean and standard deviation
- Energy range and contour
- Short-time energy statistics

### C. Spectral
- MFCC and delta MFCC
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral contrast
- Spectral flux

### D. Voice quality
- Jitter
- Shimmer
- HNR
- Formant measurements

Prioritize robust MFCC, pitch, and energy features before spending significant time on difficult voice-quality measurements.

## 36.5 — SER Dataset Selection

Study the following datasets and select one primary dataset before beginning model training.

### A. RAVDESS
- Dataset structure and file naming
- Emotion labels and intensity labels
- Actor identification
- Speech versus song recordings
- Speaker distribution
- Recording characteristics

### B. CREMA-D
- Dataset structure and file naming
- Actor identification
- Emotion labels and intensity labels
- Sentence variation
- Audio and video components
- Speaker distribution

### C. TESS
- Dataset structure
- Emotion labels
- Speaker information
- Recording characteristics
- Limitations of speaker and recording diversity

### D. SAVEE
- Dataset structure
- Speaker information
- Emotion labels
- Recording characteristics
- Dataset limitations

### Dataset Comparison

Compare:
- Number of recordings
- Number of speakers
- Emotion categories
- Language
- Recording conditions
- Class distribution
- Speaker diversity
- Dataset limitations and likely biases

**Selection guidance:** Start with one dataset and complete a reliable baseline. Add another dataset only when the first pipeline works. Do not merge datasets blindly.

## 36.6 — Dataset Preparation and Data Engineering

### Learn
- Downloading and extracting datasets
- Directory structure
- Reading metadata and file names
- Extracting emotion labels
- Speaker ID extraction
- Label mapping
- Audio validation
- Detecting corrupted and duplicate files
- Handling missing data
- Class distribution
- Label encoding
- Building a metadata DataFrame
- Keeping recording ID, speaker ID, emotion label, and file path together

### Practical Lab 36B — Dataset Loader

Build a dataset loader that:
1. Scans the dataset directory.
2. Reads each recording's metadata.
3. Extracts speaker and emotion labels.
4. Checks whether audio files can be loaded.
5. Detects missing or duplicate entries.
6. Creates a clean metadata table.
7. Reports recordings per emotion and speaker.

## 36.7 — Dataset Splitting and Leakage Prevention

**Critical topic — do not skip.**

### Learn
- Training, validation, and test sets
- Random splitting
- Stratified splitting
- Group-based splitting
- Speaker-independent splitting
- Cross-validation with speaker groups
- Data leakage
- Speaker leakage
- Duplicate leakage
- Train-test contamination
- Why high test accuracy can be misleading when the same speaker appears in training and testing

### Required Evaluation Principle

All recordings from a speaker must remain in the same partition for a speaker-independent evaluation.

### Practical Lab 36C — Reliable Data Split

Create training, validation, and test partitions using speaker IDs.

Verify that:
- No speaker occurs in more than one partition.
- Emotion classes are represented as well as the dataset permits.
- No duplicate recording crosses partitions.
- Validation and test recordings are not used for augmentation or model fitting.

## 36.8 — Class Imbalance and Training-Only Augmentation

### Learn
- Balanced versus imbalanced classes
- Majority and minority classes
- How imbalance affects predictions
- Class weights
- Weighted loss
- Random oversampling and undersampling
- Focal loss (conceptual)
- Audio augmentation strategies from Module 35
- Why validation and test data must remain untouched

### Practical Lab 36D — Class Distribution and Imbalance

Visualize class counts. Compare a baseline training setup with a class-weighted setup where appropriate.

Apply augmentation only after splitting, and only to training recordings.

## 36.9 — Exploratory Data Analysis for SER

### Learn
- Emotion-class distribution
- Speaker distribution
- Recording duration distribution
- Sampling-rate and channel checks
- Feature distributions by emotion
- MFCC distributions
- Pitch and energy distributions
- Correlation analysis
- Outlier detection
- PCA visualization
- t-SNE (conceptual; optional visualization)

### Practical Lab 36E — Dataset EDA Notebook

Build an EDA notebook that includes:
- Emotion and speaker counts
- Recording-duration plots
- Representative waveforms and Log-Mel spectrograms
- Pitch and energy comparisons
- MFCC feature distributions
- Feature correlations
- PCA visualization of extracted features

### Module 36 Completion Deliverable

**SER Dataset and Feature Analysis System**

Deliver:
- Dataset loader
- Clean metadata table
- Emotion and speaker distribution analysis
- Audio validation report
- Audio and feature visualizations
- Speaker-independent train/validation/test splits
- Extracted feature dataset
- Documented dataset limitations

---

# MODULE 37 — SPEECH EMOTION CLASSIFICATION: MACHINE LEARNING & DEEP LEARNING

**Purpose:** Train and compare emotion classifiers, evaluate their ability to generalize to unseen speakers, and build an inference application.

## 37.1 — Classification Concepts Required for SER

### Learn
- Multiclass classification
- Features, labels, and class indices
- Training, validation, and test data
- Decision boundaries
- Class probabilities
- Softmax and argmax
- Cross-entropy loss
- Overfitting and underfitting
- Generalization
- Bias-variance trade-off
- Confusion between similar emotion classes

Focus on applying these concepts to an imbalanced, speaker-dependent audio problem.

## 37.2 — Traditional Machine Learning Baselines

Start with handcrafted, fixed-length feature vectors from Module 35.

### A. Logistic Regression
- Multiclass classification
- Softmax probabilities
- Regularization
- Feature scaling

### B. Support Vector Machine (SVM)
- Hyperplane and maximum margin
- Support vectors
- Linear and RBF kernels
- C and gamma parameters
- Feature scaling

### C. Random Forest
- Decision trees
- Bootstrap sampling
- Random feature selection
- Ensemble voting
- Feature importance

### D. K-Nearest Neighbors (KNN)
- Distance-based classification
- Euclidean and Manhattan distance
- K parameter
- Importance of feature scaling

### E. Gradient Boosting
- Boosting concept
- Sequential weak learners
- XGBoost or LightGBM (optional practical extension)

### Practical Lab 37A — Traditional ML Baselines

Train and compare:
- Logistic Regression
- SVM
- Random Forest
- KNN

Test feature combinations:
- MFCC statistics
- MFCC + pitch
- MFCC + RMS energy
- MFCC + selected spectral features
- Combined acoustic features

Use the same speaker-independent data split for fair comparison. Fit scalers only on the training data.

## 37.3 — Deep Learning Foundations for Audio Classification

### Learn
- Neural-network layers, weights, and biases
- Forward propagation
- Backpropagation
- Activation functions
- Gradient descent and optimizers
- Learning rate
- Batch size and epochs
- Cross-entropy loss
- Dropout
- Batch normalization
- Weight initialization
- Regularization
- Early stopping
- Training and validation curves

## 37.4 — CNN for Log-Mel Spectrogram Classification

**Core deep-learning model.**

### Learn
- Why a spectrogram can be represented as a 2D input
- 2D convolution
- Filters and kernels
- Feature maps
- Stride and padding
- Receptive field
- ReLU
- Max pooling and average pooling
- Flattening versus global pooling
- Fully connected layers
- Softmax classification

### CNN Architecture

Log-Mel Spectrogram
        ↓
Convolution + ReLU
        ↓
Pooling
        ↓
Convolution + ReLU
        ↓
Pooling
        ↓
Global Pooling / Flatten
        ↓
Fully Connected Layer
        ↓
Softmax
        ↓
Emotion Prediction

### Practical Lab 37B — CNN Emotion Classifier

1. Prepare fixed-shape Log-Mel inputs.
2. Build a CNN classifier.
3. Train using the training partition.
4. Monitor validation loss and metrics.
5. Save the best checkpoint.
6. Evaluate on the held-out test partition.
7. Inspect the confusion matrix and misclassified examples.

## 37.5 — Sequence Modelling: RNN, LSTM and GRU

### Learn
- Sequential data and temporal dependencies
- Recurrent Neural Networks (RNN)
- Hidden states
- Unrolled RNN
- Vanishing and exploding gradients
- Backpropagation Through Time
- Long Short-Term Memory (LSTM)
- Cell state and hidden state
- Forget, input, and output gates
- Gated Recurrent Unit (GRU)
- Update and reset gates
- LSTM versus GRU

### Practical Understanding
- CNN: learns local time-frequency patterns.
- LSTM/GRU: models temporal dependencies.
- CNN-LSTM: combines local spectral feature learning and temporal modelling.

## 37.6 — CNN-LSTM for SER

### Learn
- CNN feature extraction from spectrograms
- Converting feature maps into a sequence
- Passing sequences to an LSTM
- Temporal aggregation
- Classification layer
- Advantages and limitations of CNN-LSTM

### Architecture

Log-Mel Spectrogram
        ↓
CNN Layers
        ↓
Feature Maps
        ↓
Sequence Representation
        ↓
LSTM
        ↓
Fully Connected Layer
        ↓
Emotion Prediction

### Practical Lab 37C — CNN-LSTM Classifier

Build and train a CNN-LSTM model. Compare it with the CNN using the same data partitions and evaluation metrics.

Keep this model only if it adds meaningful value.

## 37.7 — Transformer-Based and Pretrained Speech Representations

**Important modern approach.** Focus on understanding and using pretrained representations, rather than implementing a large Transformer from scratch.

### Learn
- Why attention helps sequence modelling
- Self-attention and multi-head attention
- Positional information
- Transformer encoder
- Contextual representations
- Self-supervised speech learning
- Pretraining and fine-tuning
- Transfer learning
- Frozen encoder versus trainable encoder
- Classification head

### Model Families

- **Wav2Vec 2.0:** Learns speech representations from raw audio through self-supervised pretraining.
- **HuBERT:** Learns contextual speech representations using masked prediction.
- **WavLM:** Pretrained speech representations designed for a range of speech tasks.
- **Audio Spectrogram Transformer (AST):** Applies Transformer-based learning to spectrogram inputs.

### Practical Priorities

1. Use one pretrained speech encoder to extract representations.
2. Train a simple emotion-classification head.
3. Compare a frozen encoder with partial or full fine-tuning only if resources permit.
4. Compare results with handcrafted-feature and CNN baselines.

### Practical Lab 37D — Pretrained Audio Representation Classifier

Use one suitable pretrained model. Extract embeddings or fine-tune it for the emotion labels.

Record:
- Model name
- Preprocessing requirements
- Sample rate
- Input format
- Training configuration

**Compute note:** Start with a frozen encoder or feature extraction approach if GPU memory or training time is limited.

## 37.8 — Training and Experiment Management

### Learn
- Training and validation loops
- Batching and data loaders
- Optimizers and learning-rate scheduling
- Training versus validation loss
- Training versus validation metrics
- Early stopping
- Checkpoint saving
- Best-model selection
- Random seeds
- Reproducibility
- Hyperparameter tuning
- Experiment tracking
- Consistent preprocessing between training and inference

### Practical Lab 37E — Experiment Log

For every experiment, record:
- Model architecture
- Input representation
- Dataset and split
- Feature configuration
- Augmentation settings
- Hyperparameters
- Training and validation results
- Test results
- Training time and model size

## 37.9 — Evaluation and Error Analysis

**Do not judge the project using accuracy alone.**

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Per-class precision, recall, and F1-score

### Additional Evaluation
- One-vs-rest ROC-AUC (optional)
- Precision-recall curves (optional)
- Prediction confidence
- Calibration (conceptual)
- False positives and false negatives
- Class-wise error analysis
- Misclassified audio inspection

### SER-Specific Evaluation
- Speaker-independent performance
- Speaker variation
- Dataset bias
- Class imbalance
- Accent and language variation
- Recording-condition robustness
- Cross-dataset generalization (optional extension)

### Practical Lab 37F — Model Evaluation Report

For each candidate model:
1. Calculate accuracy and macro F1-score.
2. Display the confusion matrix.
3. Report per-class precision and recall.
4. Inspect representative errors.
5. Compare results on the same held-out speakers.
6. Explain limitations and likely sources of error.

## 37.10 — Model Comparison and Ablation Studies

### Compare Models

| Model | Input |
|---|---|
| Logistic Regression | Handcrafted feature statistics |
| SVM | Handcrafted feature statistics |
| Random Forest | Combined acoustic features |
| CNN | Log-Mel spectrogram |
| CNN-LSTM | Log-Mel sequence |
| Pretrained speech encoder | Learned speech representation |

### Ablation Experiments
- MFCC alone
- Pitch alone
- Energy alone
- MFCC + pitch
- MFCC + energy
- MFCC + pitch + energy
- Log-Mel CNN
- Pretrained representations

### Learn
- Which feature groups contribute useful information
- Whether added model complexity improves generalization
- Whether a model performs differently across speakers
- Why comparisons must use the same data split
- Why small performance differences may not be statistically meaningful

## 37.11 — Deployment and Inference

### Learn
- Saving trained models and preprocessing parameters
- Loading checkpoints
- Inference pipeline
- Input validation
- Consistent sample rate and channel handling
- Feature extraction consistency
- Class-index-to-emotion mapping
- Prediction probabilities
- Single-recording inference
- Batch inference
- Inference latency
- Offline versus real-time classification
- Basic API or application integration

### Practical Lab 37G — Inference Pipeline

Upload / Record Audio
        ↓
Validate Audio
        ↓
Preprocess Audio
        ↓
Generate Model Input
        ↓
Trained SER Model
        ↓
Prediction Probabilities
        ↓
Predicted Dataset Emotion Label
        ↓
Display Results

Build a simple interface that accepts an audio file and displays:
- Audio playback
- Predicted emotion label
- Class probabilities
- Optional waveform or Log-Mel visualization

Make clear that the output is a prediction of dataset vocal-emotion categories, not a definitive reading of a person's internal feelings.

## 37.12 — Final Portfolio Project: Audio-Based Speech Emotion Recognition System

### Stage 1 — Dataset Preparation
- Select one primary dataset (RAVDESS or CREMA-D are possible starting points).
- Download and inspect recordings.
- Extract emotion and speaker labels.
- Check corrupted and duplicate files.
- Analyse class distribution.
- Create speaker-independent train, validation, and test partitions.

### Stage 2 — Exploratory Audio Analysis
- Listen to representative recordings.
- Visualize waveforms and Log-Mel spectrograms.
- Analyse pitch and energy.
- Compare acoustic patterns across emotion labels.
- Document dataset limitations.

### Stage 3 — Feature Engineering
- Extract MFCC and delta MFCC.
- Extract pitch and energy statistics.
- Extract selected spectral features.
- Build fixed-length feature vectors.
- Generate Log-Mel inputs.
- Save features and metadata.

### Stage 4 — Traditional ML Baselines

Train and compare:
- Logistic Regression
- SVM
- Random Forest
- KNN

### Stage 5 — Deep-Learning Models

Build and compare:
- CNN using Log-Mel spectrograms
- CNN-LSTM (if useful)
- One pretrained speech representation classifier

### Stage 6 — Evaluation
- Accuracy
- Macro F1-score
- Per-class precision and recall
- Confusion matrix
- Speaker-independent test performance
- Error analysis
- Model comparison and ablation study

### Stage 7 — Deployment
- Save the best model and preprocessing configuration.
- Build an audio-upload interface.
- Validate new recordings.
- Run preprocessing and inference.
- Display predicted label and class probabilities.
- Test with recordings not used during training.

### Final Project Architecture

                  AUDIO INPUT
                      |
                      v
              AUDIO VALIDATION
                      |
                      v
              AUDIO PREPROCESSING
                      |
                      v
                REPRESENTATION
                 /          \
                v            v
        HANDCRAFTED        LOG-MEL /
          FEATURES        PRETRAINED
                |         REPRESENTATION
                v            |
         TRADITIONAL ML       v
            MODELS       DEEP LEARNING
                |            |
                v            v
           ML RESULTS    DL RESULTS
                 \          /
                  v        v
                MODEL COMPARISON
                       |
                       v
                BEST MODEL SELECTION
                       |
                       v
                EMOTION PREDICTION
                       |
                       v
                 RESULTS INTERFACE

### Final Deliverables
- Clean dataset metadata
- Speaker-independent data splits
- Reusable preprocessing and feature-extraction code
- EDA notebook
- Traditional ML baseline models
- Deep-learning model(s)
- Pretrained-model experiment
- Evaluation and comparison report
- Saved best model
- Inference application
- README documenting setup, dataset, methodology, results, limitations, and how to run the project

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