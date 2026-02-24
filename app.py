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
from fpdf import FPDF
from datetime import datetime
import os

def generate_pdf(filename, tb_prob, normal_prob, result_text, image_path):


    pdf = FPDF()
    pdf.add_page()

    # -------------------------------
    # Load Unicode Font
    # -------------------------------
    font_path = "fonts/DejaVuSans.ttf"
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 16)

    # -------------------------------
    # Header
    # -------------------------------
    pdf.cell(0, 10, "AI-Assisted Tuberculosis Screening Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Image File: {filename}", ln=True)
    pdf.ln(5)

    # -------------------------------
    # Insert X-ray Image
    # -------------------------------
    if os.path.exists(image_path):
        pdf.image(image_path, x=40, w=130)
        pdf.ln(85)

    # -------------------------------
    # Probability Summary
    # -------------------------------
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 8, "1. AI Probability Assessment", ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 7,
        f"Tuberculosis Probability: {tb_prob*100:.2f}%\n"
        f"Normal Probability: {normal_prob*100:.2f}%"
    )
    pdf.ln(3)

    # -------------------------------
    # Structured Interpretation
    # -------------------------------
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 8, "2. Structured Clinical Interpretation", ln=True)
    pdf.set_font("DejaVu", "", 11)

    if tb_prob >= 0.70:
        interpretation = (
            "The AI model identifies radiographic features highly suggestive of "
            "pulmonary tuberculosis. Immediate clinical correlation, sputum testing, "
            "and physician evaluation are strongly recommended."
        )
        confidence_level = "High Confidence Detection"
    elif 0.40 <= tb_prob < 0.70:
        interpretation = (
            "The AI model identifies potential abnormalities consistent with "
            "early or mild pulmonary tuberculosis. Further diagnostic evaluation "
            "is advised to confirm findings."
        )
        confidence_level = "Moderate Confidence Detection"
    else:
        interpretation = (
            "The AI model does not detect radiographic patterns consistent with "
            "active pulmonary tuberculosis. Clinical correlation is still advised "
            "if symptoms persist."
        )
        confidence_level = "Low Probability of TB"

    pdf.multi_cell(0, 7, interpretation)
    pdf.ln(3)

    # -------------------------------
    # Model Confidence Explanation
    # -------------------------------
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 8, "3. AI Confidence Explanation", ln=True)
    pdf.set_font("DejaVu", "", 11)

    confidence_text = (
        f"The confidence score is derived from the model's sigmoid output layer. "
        f"A value of {tb_prob:.4f} represents the predicted likelihood of tuberculosis "
        f"based on learned radiographic feature patterns. "
        f"This score does not represent a confirmed diagnosis."
    )

    pdf.multi_cell(0, 7, confidence_text)
    pdf.ln(3)

    # -------------------------------
    # Technical Model Information
    # -------------------------------
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 8, "4. Technical Model Information", ln=True)
    pdf.set_font("DejaVu", "", 11)

    pdf.multi_cell(0, 7,
        "Model Type: Convolutional Neural Network (CNN)\n"
        "Input Resolution: 224 x 224 RGB\n"
        "Inference Method: Single Image Forward Pass\n"
        "Output: Binary Classification (TB vs Normal)"
    )
    pdf.ln(3)

    # -------------------------------
    # Medical Disclaimer
    # -------------------------------
    pdf.set_font("DejaVu", "", 13)
    pdf.cell(0, 8, "5. Medical Disclaimer", ln=True)
    pdf.set_font("DejaVu", "", 10)

    disclaimer = (
        "This report is generated by an Artificial Intelligence system for "
        "screening support purposes only. It is not a substitute for professional "
        "medical diagnosis, radiological assessment, or clinical judgment. "
        "Final interpretation must be performed by a licensed medical practitioner. "
        "The developers of this system assume no liability for medical decisions "
        "made based on this report."
    )

    pdf.multi_cell(0, 6, disclaimer)

    # -------------------------------
    # Footer
    # -------------------------------
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 9)
    pdf.cell(0, 8, "AI TB Detection System | Research Use Only", align="C")

    report_name = "AI_TB_Screening_Report.pdf"
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
            msg = "Positive for Tuberculosis"
        elif 0.3 <= tb_prob <= 0.5:
            msg = "⚠️ Possible Tuberculosis"
        else:
            msg = "✅ No Radiographic Evidence of Tuberculosis"

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
