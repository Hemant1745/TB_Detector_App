import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
import gdown
from fpdf import FPDF
import re

st.set_page_config(page_title="🫁 TB Detection AI", layout="wide")

# -------------------------------
# 🎨 Styling
# -------------------------------
st.markdown("""
<style>
h1, h2, h3 {text-align: center;}
.result-card {
    background-color: rgba(0,0,0,0.05);
    padding: 1.5em; border-radius: 15px;
    text-align: center; margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 Load Model
# -------------------------------
@st.cache_resource
def load_tb_model():
    model_path = "model/my_model.keras"
    if not os.path.exists(model_path):
        st.info("📥 Downloading AI model...")
        os.makedirs("model", exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1C8fXLwrfV_ZV4xjYwJgdejLzBp7-rboi"
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

# -------------------------------
# 🧾 PDF Report Generator
# -------------------------------
def generate_pdf(filename, tb_prob, normal_prob, result_text, image_path):
    pdf = FPDF()
    pdf.add_page()

    # Add Unicode font
    font_path = "fonts/DejaVuSans.ttf"
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 14)

    pdf.cell(200, 10, txt="Tuberculosis Detection Report", ln=True, align="C")
    pdf.ln(10)

    if os.path.exists(image_path):
        pdf.image(image_path, x=60, y=25, w=90)
        pdf.ln(85)

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(200, 10, txt=f"File: {filename}", ln=True)
    pdf.cell(200, 10, txt=f"TB Probability: {tb_prob*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Normal Probability: {normal_prob*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Diagnosis: {result_text}", ln=True)
    pdf.cell(200, 10, txt=f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    report_name = "tb_report.pdf"
    pdf.output(report_name)
    return report_name


# -------------------------------
# 🚀 Main Dashboard
# -------------------------------
st.title("🫁 Tuberculosis Detection AI")

model = load_tb_model()
st.success("✅ AI Model Loaded")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        img_processed = img.convert("RGB").resize((224, 224))
        img_array = np.array(img_processed) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            pred = model.predict(img_array)[0][0]

        tb_prob = float(pred)
        normal_prob = 1 - tb_prob

        if tb_prob > 0.5:
            msg = "⚠️ Tuberculosis Detected"
        elif 0.3 <= tb_prob <= 0.5:
            msg = "⚠️ Possible Tuberculosis"
        else:
            msg = "✅ Normal Chest X-ray"

        st.markdown(f"""
        <div class='result-card'>
            <h2>{msg}</h2>
            <p>TB Probability: {tb_prob*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(tb_prob)
        st.text(f"Normal Probability: {normal_prob:.4f}")

        # PDF generation
        temp_image_path = "temp_image.png"
        img.save(temp_image_path)
        report_file = generate_pdf(uploaded_file.name, tb_prob, normal_prob, msg, temp_image_path)
        os.remove(temp_image_path)

        with open(report_file, "rb") as f:
            st.download_button("⬇️ Download Report", f, file_name=report_file, mime="application/pdf")

        st.markdown(f"🕓 Analysis Time: {datetime.now().strftime('%H:%M:%S')}")
