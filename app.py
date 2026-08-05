import streamlit as st
import joblib
import numpy as np
from utils import clean_text  # Uses the exact preprocessing from notebook 02

# Page setup
st.set_page_config(
    page_title="Language Detector | Saravanan",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom UI design
st.markdown("""
<style>
    /* Main banner styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .author-tag {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffdd59;
        margin-top: -0.5rem;
    }
    
    /* Result card styling */
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #2a5298;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('language_model.pkl')
    return vectorizer, model

vectorizer, model = load_artifacts()

# Language code mapping dictionary
LABEL_MAP = {
    0: 'Arabic', 1: 'Danish', 2: 'Dutch', 3: 'English', 
    4: 'French', 5: 'German', 6: 'Greek', 7: 'Hindi', 8: 'Italian',
    9: 'Kannada', 10: 'Malayalam', 11: 'Portuguese', 12: 'Russian',
    13: 'Spanish', 14: 'Swedish', 15: 'Tamil', 16: 'Turkish'
}

# Sidebar - Project Information
with st.sidebar:
    st.title("🌐 Language Detector")
    st.markdown("**Created by:** SARAVANAVEL R")
    st.divider()
    
    st.markdown("### 📊 App Overview")
    st.info("Please input at least a sentence, since the model is trained with a small dataset hehe!!")
    
    st.divider()
    st.markdown("### 💡 Supported Languages")
    st.caption(", ".join(LABEL_MAP.values()))

# Main Banner Header
st.markdown("""
<div class="main-header">
    <h1>🌐 Language Detection App</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">my first app!!!</p>
    <p class="author-tag">by SARAVANAN ✌️🤓</p>
</div>
""", unsafe_allow_html=True)

# Layout: Split into Input Column (Left) and Results Column (Right)
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("📝 Input Section")
    st.write("Enter text in any supported language to identify it instantly.")
    st.caption("💡 *Tip: Input at least a full sentence for best accuracy since the dataset is small!*")
    
    user_input = st.text_area(
        label="Input Text Area",
        placeholder="Type or paste your text here...",
        height=180,
        label_visibility="collapsed"
    )
    
    detect_btn = st.button("🚀 Detect Language", type="primary", use_container_width=True)

with right_col:
    st.subheader("🎯 Results & Predictions")
    
    if detect_btn:
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # Preprocess and predict
            cleaned_text = clean_text(user_input)
            features = vectorizer.transform([cleaned_text])
            
            pred_code = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100
            
            language_name = LABEL_MAP.get(pred_code, f"Code {pred_code}")
            
            # Display Main Result Card
            st.markdown(f"""
            <div class="result-card">
                <small style="color: #6c757d; font-weight: bold;">PREDICTED LANGUAGE</small>
                <h2 style="color: #1e3c72; margin: 0;">{language_name} (Code: {pred_code})</h2>
                <br>
                <span style="background-color: #e3f2fd; color: #0d47a1; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: bold;">
                    Confidence: {confidence:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Fun success callout message
            st.success("yay😍!!! now you know the language of the text you've entered!!")
            
            # Top 3 Prediction breakdown
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            st.write("##### 📈 Probabilities Breakdown")
            for idx in top_3_indices:
                lang = LABEL_MAP.get(idx, f"Code {idx}")
                prob = probabilities[idx] * 100
                st.write(f"**{lang}**: {prob:.1f}%")
                st.progress(int(prob))
    else:
        st.info("👈 Enter text on the left and click **Detect Language** to view results.")