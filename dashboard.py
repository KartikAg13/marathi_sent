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
reverse_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
label_emojis = {0: "😔", 1: "😐", 2: "😊"}
label_colors = {
    0: "#ef4444",  # Red
    1: "#f59e0b",  # Amber
    2: "#10b981",  # Green
}


# ====== Prediction Function ======
def predict_sentiment(text: str):
    seq = tokenizer.texts_to_sequences([text])
    if len(seq[0]) == 0:
        return "Neutral", 0.0, [0.0, 1.0, 0.0]

    seq_padded = pad_sequences(
        seq,
        maxlen=125,
        padding="post",
        truncating="post",
        dtype="int32",
    )

    pred = model.predict(seq_padded, verbose=0)
    label = np.argmax(pred)
    confidence = np.max(pred)

    return reverse_labels[label], confidence, pred[0]


# ====== Custom CSS Styling ======
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 2.5rem 1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .sentiment-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .sentiment-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .sentiment-header p {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.03em;
    }
    
    [data-testid="stTextArea"] textarea {
        border-radius: 12px !important;
        border: 1.5px solid #334155 !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
        padding: 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    [data-testid="stTextArea"] textarea:focus {
        border-color: #f97316 !important;
        background: #0f172a !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1) !important;
    }
    
    [data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(249, 115, 22, 0.3);
    }
    
    [data-testid="stButton"] button:active {
        transform: translateY(0);
    }
    
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .sentiment-result {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .sentiment-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        animation: bounce 0.6s ease-out;
    }
    
    @keyframes bounce {
        0% { transform: translateY(-20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .sentiment-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
        letter-spacing: -0.01em;
    }
    
    .confidence-bar {
        background: #0f172a;
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        animation: fillBar 0.8s ease-out;
    }
    
    @keyframes fillBar {
        0% { width: 0; }
    }
    
    .confidence-text {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .score-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .score-item {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .score-item:hover {
        border-color: #475569;
        transform: translateY(-2px);
    }
    
    .score-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .score-value {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 300;
        letter-spacing: 0.02em;
        margin-top: 2rem;
    }
</style>
"""

st.set_page_config(
    page_title="Marathi Sentiment Analysis",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(custom_css, unsafe_allow_html=True)

# ====== Header ======
col_center = st.columns([1])[0]
with col_center:
    st.markdown(
        '<div class="sentiment-header"><h1>मराठी भावनिकता</h1><p>SENTIMENT ANALYSIS · BILSTM MODEL</p></div>',
        unsafe_allow_html=True,
    )

# ====== Input Section ======
user_input = st.text_area(
    "✏️ Enter Marathi text:",
    placeholder="उदाहरण: मला हा चित्रपट खूप आवडला!",
    height=120,
    label_visibility="visible",
)

analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ====== Results Section ======
if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some Marathi text to analyze!")
    else:
        label, confidence, preds = predict_sentiment(user_input)
        label_idx = list(reverse_labels.values()).index(label)
        color = label_colors[label_idx]
        emoji = label_emojis[label_idx]

        # Result card
        st.markdown(
            f"""
        <div class="result-card">
            <div class="sentiment-result">
                <div class="sentiment-emoji">{emoji}</div>
                <div class="sentiment-label" style="color: {color};">{label}</div>
                <div class="confidence-text">Confidence: {confidence * 100:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence * 100}%; background: {color};"></div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.balloons()

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">BiLSTM + TensorFlow + Streamlit · Marathi NLP</div>',
    unsafe_allow_html=True,
)
