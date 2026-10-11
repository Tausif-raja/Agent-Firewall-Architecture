import os
import streamlit as st
from prompt_injection.detect_prompt_injection import PromptInjectionDetector, USE_GEMINI

# ---------------------------
# Init Detector (Gemini key optional)
# ---------------------------
gemini_key = os.environ.get("GOOGLE_API_KEY")
detector = PromptInjectionDetector(use_gemini=USE_GEMINI and (gemini_key is not None),
                                   gemini_api_key=gemini_key)

# --------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🛡️ Prompt Injection Detector", page_icon="🤖", layout="wide")

st.title("🛡️ Prompt Injection Detector")
st.write("यह tool आपके prompt को **Direct / Indirect / Semantic** checks से scan करता है।")

# Text Input
user_input = st.text_area("🔹 Prompt ", height=150)

# External docs (indirect injection test के लिए)
st.subheader("📄 External Documents (optional)")
external_texts = st.text_area("External docs (newline separated):",
                              "This is a normal article about security best practices.")
external_docs = [d.strip() for d in external_texts.split("\n") if d.strip()]

# Button
if st.button("🚀 Detect Injection"):
    if user_input.strip() == "":
        st.warning("⚠️ कृपया कोई prompt डालें।")
    else:
        result = detector.detect(user_input, external_texts=external_docs)

        # Final verdict
        if result["final"] == "Injection":
            st.error(f"🚨 {result['final']}")
        else:
            st.success(f"✅ {result['final']}")

        # Reasons
        st.markdown("### 📝 Detection Details")
        for r in result["reasons"]:
            st.write(f"- {r}")

        # Expand raw results
        with st.expander("🔍 Raw Detection JSON"):
            st.json(result)