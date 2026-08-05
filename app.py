import streamlit as st
import joblib
import numpy as np
import io
from deep_translator import GoogleTranslator
from gtts import gTTS
from utils import clean_text  # Exact preprocessing from notebook 02

# Page setup
st.set_page_config(
    page_title="Language Detector | Saravanan",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
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

# Mapping dictionary: Name & ISO 639-1 Code
LANG_INFO = {
    0: {'name': 'Arabic', 'code': 'ar'},
    1: {'name': 'Danish', 'code': 'da'},
    2: {'name': 'Dutch', 'code': 'nl'},
    3: {'name': 'English', 'code': 'en'},
    4: {'name': 'French', 'code': 'fr'},
    5: {'name': 'German', 'code': 'de'},
    6: {'name': 'Greek', 'code': 'el'},
    7: {'name': 'Hindi', 'code': 'hi'},
    8: {'name': 'Italian', 'code': 'it'},
    9: {'name': 'Kannada', 'code': 'kn'},
    10: {'name': 'Malayalam', 'code': 'ml'},
    11: {'name': 'Portuguese', 'code': 'pt'},
    12: {'name': 'Russian', 'code': 'ru'},
    13: {'name': 'Spanish', 'code': 'es'},
    14: {'name': 'Swedish', 'code': 'sv'},
    15: {'name': 'Tamil', 'code': 'ta'},
    16: {'name': 'Turkish', 'code': 'tr'}
}

# Sidebar Info
with st.sidebar:
    st.title("🌐 Language Detector")
    st.markdown("**Created by:** SARAVANAVEL R")
    st.divider()
    
    st.markdown("### 📊 this app Features")
    st.markdown("- 🔍 **NLP Detection**\n- 🔊 **Audio Pronunciation**\n- 🔄 **AI Translation**")
    st.info("Input at least a sentence for the best accuracy!")
    
    st.divider()
    st.write("note:")
    st.markdown("### 💡 Supported Languages")
    st.caption(", ".join([info['name'] for info in LANG_INFO.values()]))
    st.write("more languages coming soon !!!")

# Main Banner Header
st.markdown("""
<div class="main-header">
    <h1>🌐 Language Detection & AI Translator</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">my first web-application guys!! input any text to detect its language, pronunciation and it's meaning</p>
    <p class="author-tag">- by SARO🤓</p>
</div>
""", unsafe_allow_html=True)

# Layout
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("📝 Input Section")
    st.write("Enter text in any supported language to detect, translate, and listen.")
    
    user_input = st.text_area(
        label="Input Text Area",
        placeholder="Type or paste your text here...",
        height=180,
        label_visibility="collapsed"
    )
    
    # Target translation choice
    target_lang = st.selectbox(
        "Translate text into:",
        options=["English", "Tamil", "Hindi"],
        index=0
    )
    
    target_code_map = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
    
    detect_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

with right_col:
    st.subheader("🎯 Results & Insights")
    
    if detect_btn:
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # 1. Preprocess & Predict
            cleaned_text = clean_text(user_input)
            features = vectorizer.transform([cleaned_text])
            
            pred_code = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100
            
            lang_data = LANG_INFO.get(pred_code, {'name': f'Code {pred_code}', 'code': 'en'})
            language_name = lang_data['name']
            iso_code = lang_data['code']
            
            # Display Result Card
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
            
            # 2. Pronunciation Audio Guide (gTTS)
            st.markdown("##### 🔊 Listen to Pronunciation")
            try:
                tts = gTTS(text=user_input, lang=iso_code)
                sound_file = io.BytesIO()
                tts.write_to_fp(sound_file)
                st.audio(sound_file, format='audio/mp3')
            except Exception as e:
                st.caption("⚠️ Audio preview not available for this input format.")
            
            st.divider()
            
            # 3. AI Translation (deep-translator)
            st.markdown(f"##### 🔄 Translation ({target_lang})")
            try:
                translated_text = GoogleTranslator(
                    source='auto', 
                    target=target_code_map[target_lang]
                ).translate(user_input)
                
                st.success(translated_text)
            except Exception as e:
                st.error("Could not complete translation right now.")
            
            st.success("yay😍!!! now you know the language, it's pronunciation, and it's meaning!!")
            
            # 4. Top 3 Probabilities
            with st.expander("📊 View Probability Breakdown"):
                top_3_indices = np.argsort(probabilities)[::-1][:3]
                for idx in top_3_indices:
                    l_name = LANG_INFO.get(idx, {}).get('name', f"Code {idx}")
                    prob = probabilities[idx] * 100
                    st.write(f"**{l_name}**: {prob:.1f}%")
                    st.progress(int(prob))
    else:
        st.info("👈 Enter text on the left and click **Analyze Text** to view detection, audio, and translation.")
