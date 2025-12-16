# 🏥 Medical Auto-Triage System

## Project Overview

The **Medical Auto-Triage System** is a hybrid machine learning application designed to automatically classify free-text patient symptom descriptions into the most appropriate medical specialty (e.g., Cardiology, Surgery, Neurology). The system combines **traditional NLP techniques (TF-IDF)** with **deep contextual embeddings (BERT)** to improve triage accuracy, reduce patient routing time, and support healthcare staff with data-driven decision-making.

This project serves as both a **research prototype** and a **proof-of-concept triage assistant**, demonstrating how modern NLP methods can assist in early-stage clinical workflows.

---

## Key Objectives

- Reduce manual triage time for patient intake
- Improve accuracy of initial specialty routing
- Handle free-form, unstructured symptom descriptions
- Provide confidence-aware recommendations to guide decision-making

---

## Dataset Description

### Primary Dataset

- **Medical Transcriptions Dataset (Kaggle)**  
  https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions  
- ~5,000 original samples  
- After cleaning and preprocessing: ~3,500–4,000 usable samples  
- Each entry contains:
  - Medical transcription text
  - Associated medical specialty
  - Additional metadata fields

### Why This Dataset?

- Contains **realistic clinical language** and symptom descriptions  
- Direct mapping between **free-text input** and **specialty labels**  
- Closely matches the intended triage use case  

### Data Limitations

- Medical datasets are difficult to obtain due to **HIPAA and privacy restrictions**
- Limited sample size compared to large-scale clinical systems
- Some class imbalance across specialties

Despite these constraints, the dataset is well-suited for demonstrating a **hybrid NLP triage pipeline**.

---

## Development Approach

We followed an **iterative model development strategy**, progressively increasing model complexity and representational power:

### 1. Baseline: TF-IDF + Logistic Regression
- Fast to train and evaluate
- Highly interpretable
- Strong baseline for text classification
- Captures explicit medical terminology and keywords

### 2. Contextual Model: BERT
- Uses transformer-based contextual embeddings
- Captures semantic meaning beyond exact word matches
- Understands clinical synonyms  
  *(e.g., “heart attack” vs “myocardial infarction”)*

### 3. Hybrid Model: TF-IDF + BERT
- Combines:
  - **TF-IDF** → lexical and n-gram precision
  - **BERT embeddings** → semantic understanding
- Concatenated feature representation
- Final classifier trained on the combined feature space

This hybrid approach leverages the strengths of both traditional and deep NLP models.

---

## Model Architecture

### Overview

- **Architecture Type:** Hybrid NLP model  
- **Embedding Model:** BERT (bert-base-uncased, frozen)  
- **Feature Engineering:** TF-IDF (unigrams, bigrams, trigrams)  
- **Classifier:** Logistic Regression  
- **Input:** Free-text symptom descriptions (English)  
- **Output:** Medical specialty classification with confidence scores  
- **Number of Classes:** 10 medical departments  

### Why Hybrid?

- TF-IDF captures **exact clinical terms**
- BERT captures **context and semantics**
- Combined representation improves robustness on small datasets

---

## Features

- **Real-Time Inference**  
  Instant department recommendations for new symptom input

- **Confidence-Aware Output**  
  Provides probability scores to guide routing decisions

- **Symptom Understanding**  
  Handles unstructured, free-text clinical descriptions

- **Interactive Web Interface**  
  Built with **Gradio** for easy deployment and sharing

- **Modular Design**  
  Easily extendable to additional datasets or languages

---

## Understanding the Output

### Department Recommendation

The system returns:
- **Predicted medical specialty**
- **Confidence score (%)**

#### Confidence Interpretation

- **High Confidence (>70%)**  
  Strong recommendation for this department

- **Moderate Confidence (40–70%)**  
  Likely department, consider secondary options

- **Low Confidence (<40%)**  
  Uncertain prediction — route to general medicine or manual triage

---

## Use Cases

- **Hospital Triage Systems**  
  Assist intake nurses with early department routing

- **Telemedicine Platforms**  
  Provide initial specialty recommendations for virtual consultations

- **Clinical Research Prototypes**  
  Baseline system for evaluating automated triage approaches

- **Educational Tool**  
  Demonstrates applied NLP in healthcare contexts

---

## Technical Stack & Requirements

### Core Dependencies

- **Python 3.8+**
- **PyTorch 2.0+**
- **Hugging Face Transformers**
- **Scikit-learn**
- **Pandas & NumPy**
- **Gradio**
- **Pickle** (model serialization)

---

## Future Improvements

- Expand training data using multiple medical datasets
- Mobile and web application deployment
- Voice-to-text symptom input
- Multilingual symptom support
- Integration with hospital directories and nearest urgent care lookup
- Incorporate patient history and demographic features
- Fine-tune BERT on domain-specific medical text

---

## Disclaimer

This project is a **research and educational prototype** and is **not intended for clinical use**. Predictions should not replace professional medical judgment.
