import tensorflow as tf
import cv2
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

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
import uuid
import qrcode

def generate_pdf(
    filename,
    tb_prob,
    normal_prob,
    image_path,
    patient_name,
    patient_age,
    patient_gender,
    patient_id,
    referring_physician
):


    # -----------------------------
    # Generate Report ID
    # -----------------------------
    report_id = f"TB-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    pdf = FPDF()
    pdf.add_page()

    # -----------------------------
    # Load Unicode Font
    # -----------------------------
    font_path = "fonts/DejaVuSans.ttf"
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # -----------------------------
    # Hospital Logo (optional)
    # -----------------------------
    # Hospital Logo
    if os.path.exists("assets/logo.jpg"):
        pdf.image("assets/logo.jpg", x=10, y=8, w=30)



    # -----------------------------
    # Header
    # -----------------------------
    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "AI-Assisted Tuberculosis Screening Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 8, f"Report ID: {report_id}", ln=True)
    pdf.cell(0, 8, f"Generated: {timestamp}", ln=True)
    pdf.ln(3)

    # -----------------------------
    # Patient Information
    # -----------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("DejaVu", "", 10)

    pdf.multi_cell(0, 6,
        f"Name: {patient_name or 'N/A'}\n"
        f"Patient ID: {patient_id or 'N/A'}\n"
        f"Age: {patient_age}\n"
        f"Gender: {patient_gender}\n"
        f"Referring Physician: {referring_physician or 'N/A'}"
    )
    pdf.ln(3)

    # -----------------------------
    # Insert X-ray Image
    # -----------------------------
    if os.path.exists(image_path):
        pdf.image(image_path, x=40, w=130)
        pdf.ln(85)

    # -----------------------------
    # AI Probability Assessment
    # -----------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "AI Probability Assessment", ln=True)
    pdf.set_font("DejaVu", "", 10)

    pdf.multi_cell(0, 6,
        f"Tuberculosis Probability: {tb_prob*100:.2f}%\n"
        f"Normal Probability: {normal_prob*100:.2f}%"
    )
    pdf.ln(3)

    # -----------------------------
    # Clinical Interpretation
    # -----------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "Clinical Interpretation", ln=True)
    pdf.set_font("DejaVu", "", 10)

    if tb_prob >= 0.70:
        interpretation = (
            "Findings highly suggestive of active pulmonary tuberculosis. "
            "Immediate clinical evaluation and laboratory confirmation recommended."
        )
    elif 0.40 <= tb_prob < 0.70:
        interpretation = (
            "Radiographic features possibly consistent with tuberculosis. "
            "Further diagnostic testing is advised."
        )
    else:
        interpretation = (
            "No radiographic evidence strongly indicative of active pulmonary tuberculosis."
        )

    pdf.multi_cell(0, 6, interpretation)
    pdf.ln(3)

    # -----------------------------
    # AI Confidence Explanation
    # -----------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "AI Confidence Explanation", ln=True)
    pdf.set_font("DejaVu", "", 10)

    pdf.multi_cell(0, 6,
        f"The confidence score ({tb_prob:.4f}) represents the predicted "
        f"likelihood of tuberculosis based on the AI model's learned "
        f"radiographic feature patterns. This does not constitute a confirmed diagnosis."
    )
    pdf.ln(3)

    # -----------------------------
    # QR Code
    # -----------------------------
    qr_data = (
        f"Report ID: {report_id}\n"
        f"Patient: {patient_name}\n"
        f"TB Probability: {tb_prob:.4f}\n"
        f"Generated: {timestamp}"
    )

    qr = qrcode.make(qr_data)
    qr_path = "temp_qr.png"
    qr.save(qr_path)

    pdf.image(qr_path, x=160, y=20, w=35)
    os.remove(qr_path)

    # -----------------------------
    # Digital Signature Block
    # -----------------------------
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "Authorized Digital Signature", ln=True)
    pdf.set_font("DejaVu", "", 10)

    pdf.multi_cell(0, 6,
        "AI Radiology System\n"
        "Digitally Generated Report\n"
        f"Signature ID: {report_id}"
    )
    pdf.ln(3)

    # -----------------------------
    # Medical Disclaimer
    # -----------------------------
    pdf.set_font("DejaVu", "", 9)
    pdf.multi_cell(0, 5,
        "This AI-generated report is intended for screening support purposes only "
        "and does not replace professional medical diagnosis. Clinical decisions "
        "must be made by a licensed medical practitioner."
    )

    report_name = f"{report_id}_TB_Report.pdf"
    pdf.output(report_name)

    return report_name

# -------------------------------
# 🚀 Main Dashboard
# -------------------------------
st.title("🫁 Tuberculosis Detection AI")

model = load_tb_model()
st.success("✅ AI Model Loaded")

# -------------------------
# Patient Information Form
# -------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    patient_id = st.text_input("Patient ID")
    referring_physician = st.text_input("Referring Physician")

st.divider()

# -------------------------
# File Upload
# -------------------------
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


    superimposed_img = cv2.addWeighted(
        np.array(img), 0.6, heatmap, 0.4, 0
    )

    st.subheader("📍 TB Localization (AI Highlight)")
    st.image(superimposed_img, use_container_width=True)
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

        report_file = generate_pdf(
            uploaded_file.name,
            tb_prob,
            normal_prob,
            temp_image_path,
            patient_name or "N/A",
            patient_age,
            patient_gender,
            patient_id or "N/A",
            referring_physician or "N/A"
        )

        os.remove(temp_image_path)

        with open(report_file, "rb") as f:
            st.download_button(
                "⬇️ Download Report",
                f,
                file_name=report_file,
                mime="application/pdf"
            )

        st.markdown(
            f"Analysis Time: {datetime.now().strftime('%H:%M:%S')}"
        )
