# app.py
# Run with:
#   pip install gradio scikit-learn
#   python app.py

import gradio as gr
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# -------------------------
# Model Loading (Optional)
# -------------------------

# MODEL_AVAILABLE = os.path.exists("specialty_classifier.pkl")
# BERT_AVAILABLE = os.path.exists("./bert_model")

# if MODEL_AVAILABLE:
#     with open("specialty_classifier.pkl", "rb") as f:
#         data = pickle.load(f)
#         model = data["model"]
#         vectorizer = data["vectorizer"]

# Load Logistic Regression side

MODEL_AVAILABLE = True

# Load Logistic Regression + TF-IDF
with open("logreg.pkl", "rb") as f:
    logreg_bundle = pickle.load(f)

logreg_model = logreg_bundle["model"]
vectorizer = logreg_bundle["vectorizer"]
label_encoder = logreg_bundle["label_encoder"]

# Load BERT
tokenizer = AutoTokenizer.from_pretrained("./bert_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_model")
bert_model.eval()

def predict_specialty(text):
    """
    Returns:
    {
        "specialty": str,
        "confidence": float (0–1)
    }
    """

    if not text or len(text.strip()) < 10:
        return {"specialty": "Unknown", "confidence": 0.0}

    # ---------- Logistic Regression ----------
    X_tfidf = vectorizer.transform([text])
    logreg_probs = logreg_model.predict_proba(X_tfidf)

    # ---------- BERT ----------
    encodings = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = bert_model(**encodings)
        bert_probs = softmax(outputs.logits.numpy(), axis=1)

    # ---------- Ensemble ----------
    ensemble_probs = (bert_probs + logreg_probs) / 2
    idx = ensemble_probs.argmax(axis=1)[0]

    return {
        "specialty": label_encoder.inverse_transform([idx])[0],
        "confidence": float(ensemble_probs[0, idx])
    }



# -------------------------
# Triage Function
# -------------------------

def triage(
    name,
    age,
    temperature,
    pain_level,
    history,
    surgeries,
    medications,
    symptoms,
    level1,
    level2
):
    # -------------------------
    # Basic Validation
    # -------------------------
    if not symptoms or len(symptoms.strip()) < 10:
        return "⚠️ Please provide a more detailed description of symptoms."

    # -------------------------
    # Priority Screening Logic
    # -------------------------
    priority_message = ""

    level1_flag = level1 and len(level1) > 0
    level2_flag = level2 and len(level2) > 0
    high_fever_flag = temperature is not None and temperature >= 104
    severe_pain_flag = pain_level is not None and pain_level >= 9

    if level1_flag or high_fever_flag or severe_pain_flag:
        priority_message = (
            "⚠️ **Priority Notice**\n\n"
            "Based on your responses, some symptoms may require "
            "**immediate medical attention**. Please consider seeking care promptly.\n\n"
        )
    elif level2_flag:
        priority_message = (
            "⚠️ **Priority Notice**\n\n"
            "Your responses suggest symptoms that may benefit from "
            "**prompt medical evaluation**.\n\n"
        )

    if not MODEL_AVAILABLE:
        priority_message += "_Note: System is currently running in demo mode._\n\n"

    # -------------------------
    # Combine Inputs for ML
    # -------------------------
    combined_text = f"""
    Patient Name: {name}
    Age: {age}
    Temperature: {temperature}
    Pain Level: {pain_level}
    Medical History: {', '.join(history) if history else 'None'}
    Past Surgeries: {', '.join(surgeries) if surgeries else 'None'}
    Current Medications: {medications}
    Symptoms: {symptoms}
    """

    # -------------------------
    # ML Prediction
    # -------------------------
    result = predict_specialty(combined_text)
    specialty = result["specialty"]
    confidence = result["confidence"]

    # -------------------------
    # Final Output
    # -------------------------
    return (
        priority_message +
        f"🏥 **Recommended Department:** {specialty}\n\n"
        f"📊 **Confidence Score:** {confidence*100:.1f}%\n\n"
        "---\n"
        "**Intake Summary (for clinical review):**\n"
        f"{combined_text}"
    )


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(title="Patient Intake & Auto-Triage") as demo:
    gr.Markdown("## 🏥 Patient Intake & Auto-Triage System")
    gr.Markdown(
        "⚠️ *This tool is for educational and demonstration purposes only. "
        "It does not provide medical advice or diagnosis.*"
    )

    # Patient Info
    with gr.Row():
        name = gr.Textbox(label="Patient Name")
        age = gr.Number(label="Age", value=30)

    with gr.Row():
        temperature = gr.Number(label="Temperature (°F)", value=98.6)
        pain_level = gr.Slider(0, 10, value=5, label="Pain Level (0–10)")

    # Safety Screening
    gr.Markdown("### 🩺 Quick Safety Screening")

    level1 = gr.CheckboxGroup(
        [
            "Severe difficulty breathing",
            "Uncontrolled bleeding",
            "Loss of consciousness",
            "Major trauma or injury"
        ],
        label="Urgent Symptoms (Select any that apply)"
    )

    level2 = gr.CheckboxGroup(
        [
            "Chest pain or pressure",
            "Sudden weakness or numbness",
            "Confusion or trouble speaking",
            "Severe or worsening pain"
        ],
        label="Concerning Symptoms (Select any that apply)"
    )

    # Medical History
    history = gr.CheckboxGroup(
        [
            "Diabetes",
            "Hypertension",
            "Asthma",
            "Heart Disease",
            "None"
        ],
        label="Personal Medical History"
    )

    surgeries = gr.CheckboxGroup(
        [
            "Appendectomy",
            "Heart Surgery",
            "Joint Replacement",
            "None"
        ],
        label="Past Surgeries / Procedures"
    )

    medications = gr.Textbox(
        label="Current Medications",
        placeholder="e.g., Metformin, Ibuprofen"
    )

    symptoms = gr.Textbox(
        label="Describe Current Symptoms",
        lines=4,
        placeholder="Describe what the patient is currently experiencing..."
    )

    submit = gr.Button("Run Triage")
    output = gr.Markdown()

    submit.click(
        triage,
        inputs=[
            name,
            age,
            temperature,
            pain_level,
            history,
            surgeries,
            medications,
            symptoms,
            level1,
            level2
        ],
        outputs=output
    )

demo.launch(share=True)
