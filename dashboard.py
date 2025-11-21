import numpy as np

# Patch for deprecated numpy unicode_ attribute
if not hasattr(np, "unicode_"):
    np.unicode_ = str

import streamlit as st
import keras
import pickle
from keras_preprocessing.sequence import pad_sequences


# ====== Load Model and Tokenizer ======
@st.cache_resource
def load_model():
    return keras.models.load_model("model/bilstm_marathi.keras")


@st.cache_resource
def load_tokenizer():
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


model = load_model()
tokenizer = load_tokenizer()

# ====== Reverse Mapping ======
reverse_labels = {0: "Negative (-1)", 1: "Neutral (0)", 2: "Positive (1)"}

colors = {"Negative (-1)": "red", "Neutral (0)": "orange", "Positive (1)": "green"}


# ====== Prediction Function ======
def predict_sentiment(text: str):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])

    # If model finds no valid tokens, return Neutral with low confidence
    if len(seq[0]) == 0:
        return "Neutral (0)", 0.0, [0.0, 1.0, 0.0]

    # Pad sequence with explicit dtype to avoid numpy unicode error
    seq_padded = pad_sequences(
        seq,
        maxlen=125,
        padding="post",
        truncating="post",
        dtype="int32",  # <--- IMPORTANT FIX
    )

    pred = model.predict(seq_padded)
    label = np.argmax(pred)
    confidence = np.max(pred)

    return reverse_labels[label], confidence, pred[0]


# ====== UI ======
st.set_page_config(
    page_title="Marathi Sentiment Analysis", page_icon="📊", layout="centered"
)

st.title("📍 Marathi Sentiment Analysis")
st.write("**BiLSTM based model for real-time sentiment classification**")

user_input = st.text_area(
    "✏️ Enter Marathi text:", placeholder="e.g. मला हा चित्रपट खूप आवडला!"
)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        label, confidence, preds = predict_sentiment(user_input)

        st.markdown(f"### Result: **{label}**")
        st.progress(float(confidence))

        st.subheader("📌 Confidence Scores")
        st.write(f"Negative (-1): {preds[0]:.3f}")
        st.write(f"Neutral (0): {preds[1]:.3f}")
        st.write(f"Positive (1): {preds[2]:.3f}")

        st.balloons()

st.markdown("---")
st.caption("🚀 Developed using BiLSTM, TensorFlow & Streamlit")
