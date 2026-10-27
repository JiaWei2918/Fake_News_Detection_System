import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ---------- NLTK setup ----------
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
download_nltk()

# ---------- Load tokenizer ----------
@st.cache_resource
def load_tokenizer():
    with open('models/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# ---------- Load Saved Model ----------
@st.cache_resource
def load_model():
    try:
        model = tf.saved_model.load('models/bilstm_savedmodel')
        # Try common signature names
        if 'serve' in model.signatures:
            return model.signatures['serve']
        elif 'serving_default' in model.signatures:
            return model.signatures['serving_default']
        else:
            # Fallback: take the first available signature
            return next(iter(model.signatures.values()))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---------- Preprocessing ----------
stop_words = set(stopwords.words('english'))
custom_stopwords = {'u', 'us', 's', 'amp', 'im', 'ive', 'dont', 'didnt', 'wasnt'}
stop_words = stop_words.union(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ---------- Prediction ----------
def predict(text, tokenizer, infer, max_len=500):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    input_tensor = tf.constant(padded, dtype=tf.float32)
    outputs = infer(input_tensor)
    # Output is a dict; take the first tensor (the probability of fake)
    prob_fake = float(list(outputs.values())[0].numpy()[0][0])
    prob_real = 1 - prob_fake
    return prob_real, prob_fake

# ---------- LIME explainer (cached) ----------
@st.cache_resource
def get_lime_explainer():
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=['Real', 'Fake'])

# ---------- Custom CSS ----------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    /* Result boxes */
    .result-box {
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.6rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .fake {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        color: #b71c1c;
    }
    .real {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        color: #1b5e20;
    }
    /* Confidence progress bar */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    /* LIME word styling */
    .word-positive {
        color: #2e7d32;
        font-weight: bold;
    }
    .word-negative {
        color: #d32f2f;
        font-weight: bold;
    }
    /* Sidebar text */
    .sidebar-text {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📊 About the Model")
    st.write("**Model:** BiLSTM (Bidirectional LSTM)")
    st.write("**Dataset:** Combined (ISOT + LIAR + Kaggle)")
    st.write("**Test Accuracy:** 90.03%")
    st.write("**Explainability:** LIME (Local Interpretable Model-agnostic Explanations)")
    st.write("---")
    st.write("**Instructions:**")
    st.write("1. Enter a news text (English)")
    st.write("2. Click 'Check News'")
    st.write("3. See the prediction and why the model decided that way")
    st.write("---")
    st.write("🔗 Built for INT4203E Assignment")
    st.caption("© 2026 - Educational Purpose Only")

# ---------- Main Content ----------
st.markdown('<div class="main-title">📰 AI-Based Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a news headline or article to check if it’s Real or Fake</div>', unsafe_allow_html=True)

# Note about language
st.info("ℹ️ This model is trained on **English** news articles only. Results may vary for other languages.")

# Load model and tokenizer once
try:
    tokenizer = load_tokenizer()
    infer = load_model()
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    st.stop()
st.success("✅ Model ready!")

# Example dropdown
example = st.selectbox(
    "Quick test with an example:",
    ["Custom input", "Fake: COVID-19 microchip", "Real: Fed rate hike"]
)
user_input = ""
if example == "Fake: COVID-19 microchip":
    user_input = "COVID-19 vaccine contains microchips to track people, whistleblower reveals."
elif example == "Real: Fed rate hike":
    user_input = "The Federal Reserve raised interest rates by 0.25% today, citing strong economic growth."
else:
    user_input = ""

# Text input area
input_text = st.text_area("Paste news text here:", value=user_input, height=200)

if st.button("Check News", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            prob_real, prob_fake = predict(input_text, tokenizer, infer, max_len=500)

        st.subheader("Prediction Result")
        if prob_fake > 0.5:
            st.markdown(f"""
                <div class="result-box fake">
                    🚨 FAKE NEWS<br>
                    Confidence: {prob_fake:.2%}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box real">
                    ✅ REAL NEWS<br>
                    Confidence: {prob_real:.2%}
                </div>
            """, unsafe_allow_html=True)

        st.write("---")
        st.write("**Confidence scores:**")
        col1, col2 = st.columns(2)
        col1.metric("Real", f"{prob_real:.2%}")
        col2.metric("Fake", f"{prob_fake:.2%}")
        st.progress(prob_fake, text=f"Fake confidence: {prob_fake:.1%}")

        # LIME explanation
        with st.expander("🔍 Why did the model decide this?"):
            st.markdown("**Explanation generation may take 60-90 seconds.**")
            with st.spinner("Generating explanation..."):
                explainer = get_lime_explainer()

                def lime_predict(texts):
                    # Returns probability matrix [[prob_real, prob_fake], ...]
                    probs = []
                    batch_size = 32
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        
                        for t in batch:
                            r, f = predict(t, tokenizer, infer, max_len=500)
                            probs.append([r, f])
                    return np.array(probs)

                exp = explainer.explain_instance(
                    input_text,
                    lime_predict,
                    num_features=10,
                    top_labels=1,
                    num_samples=500
                )

                pred_label = 0 if prob_real > prob_fake else 1
                top_words = exp.as_list(label=pred_label)

                st.markdown(f"**Top words influencing the prediction (class: {exp.class_names[pred_label]}):**")

                for word, weight in top_words:
                    cls = "word-positive" if weight > 0 else "word-negative"
                    st.markdown(f'<span class="{cls}">{word}: {weight:.3f}</span><br>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Model: BiLSTM | Trained on combined fake news dataset | For educational purposes only.")