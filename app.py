# before you run cod emake sure you: pip install gradio && python app.py

import gradio as gr
import pickle
import random
import os

MODEL_AVAILABLE = os.path.exists("specialty_classifier.pkl")

if MODEL_AVAILABLE:
    with open("specialty_classifier.pkl", "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        vectorizer = data["vectorizer"]

def predict(symptoms):
    if len(symptoms) < 10:
        return "Please enter more details about your symptoms."

    if not MODEL_AVAILABLE:
        specialty, confidence = random.choice([
            ("General Medicine", 0.72),
            ("Neurology", 0.64),
            ("Orthopedics", 0.58),
            ("Cardiology", 0.81),
        ])
    else:
        text_tfidf = vectorizer.transform([symptoms])
        probs = model.predict_proba(text_tfidf)[0]
        idx = probs.argmax()
        specialty = model.classes_[idx]
        confidence = probs[idx]

    return (
        f"🏥 Recommended Department: **{specialty}**\n\n"
        f"📊 Confidence: {confidence*100:.1f}%"
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Describe your symptoms..."),
    outputs=gr.Markdown(),
    title="🏥 Medical Department Recommender",
    examples=[
        ["I have chest pain and difficulty breathing"],
        ["My back hurts and I have pain in my knee"],
        ["I've been feeling very anxious and can't sleep"],
    ]
)

demo.launch()