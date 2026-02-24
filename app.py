import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
import gdown
from fpdf import FPDF
import uuid
import qrcode

st.set_page_config(page_title="🫁 TB Detection AI", layout="wide")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
h1, h2, h3 {text-align: center;}
.result-card {
    background-color: rgba(0,0,0,0.05);
    padding: 1.5em;
    border-radius: 15px;
    text-align: center;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
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
# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    # Ensure gradient tracking
    img_tensor = tf.convert_to_tensor(img_array)
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.outputs[0],
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=True)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # 🔴 If grads is None, stop safely
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, conv_outputs),
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# -------------------------------
# PDF Generator
# -------------------------------
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

    report_id = f"TB-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    pdf = FPDF()
    pdf.add_page()

    font_path = "fonts/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", size=12)

    if os.path.exists("assets/logo.jpg"):
        pdf.image("assets/logo.jpg", x=10, y=8, w=30)

    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "AI-Assisted Tuberculosis Screening Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 8, f"Report ID: {report_id}", ln=True)
    pdf.cell(0, 8, f"Generated: {timestamp}", ln=True)
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)

    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 6,
        f"Name: {patient_name}\n"
        f"Patient ID: {patient_id}\n"
        f"Age: {patient_age}\n"
        f"Gender: {patient_gender}\n"
        f"Referring Physician: {referring_physician}"
    )
    pdf.ln(5)

    if os.path.exists(image_path):
        pdf.image(image_path, x=40, w=130)
        pdf.ln(85)

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "AI Probability Assessment", ln=True)

    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 6,
        f"Tuberculosis Probability: {tb_prob*100:.2f}%\n"
        f"Normal Probability: {normal_prob*100:.2f}%"
    )
    pdf.ln(5)

    if tb_prob >= 0.7:
        interpretation = "Findings highly suggestive of active pulmonary tuberculosis."
    elif 0.4 <= tb_prob < 0.7:
        interpretation = "Radiographic features possibly consistent with tuberculosis."
    else:
        interpretation = "No strong radiographic evidence of tuberculosis."

    pdf.multi_cell(0, 6, interpretation)
    pdf.ln(5)

    qr_data = f"Report ID: {report_id}\nTB Probability: {tb_prob:.4f}"
    qr = qrcode.make(qr_data)
    qr_path = "temp_qr.png"
    qr.save(qr_path)
    pdf.image(qr_path, x=160, y=20, w=35)
    os.remove(qr_path)

    pdf.set_font("DejaVu", "", 9)
    pdf.multi_cell(0, 5,
        "This AI-generated report is for screening purposes only "
        "and does not replace professional medical diagnosis."
    )

    report_name = f"{report_id}_TB_Report.pdf"
    pdf.output(report_name)

    return report_name

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🫁 Tuberculosis Detection AI")

model = load_tb_model()
# Force model graph build (Keras 3 fix)
dummy_input = np.zeros((1, 224, 224, 3))
_ = model(dummy_input)
st.success("✅ AI Model Loaded")


# Patient Form
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", 0, 120)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    patient_id = st.text_input("Patient ID")
    referring_physician = st.text_input("Referring Physician")

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["jpg", "jpeg", "png"],
    key="tb_upload"
)

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
            pred = model(img_array, training=False)
        tb_prob = float(pred.numpy()[0][0])
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

        # Grad-CAM
        if tb_prob > 0.5:
            heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2")

if heatmap is not None:
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        np.array(img),
        0.6,
        heatmap,
        0.4,
        0
    )

    st.subheader("📍 AI Highlighted TB Region")
    st.image(superimposed_img, use_container_width=True)

            

        # Generate PDF
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
