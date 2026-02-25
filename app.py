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
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

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
    heatmap_path,
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

    # Header
    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "AI-Assisted Tuberculosis Screening Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 8, f"Report ID: {report_id}", ln=True)
    pdf.cell(0, 8, f"Generated: {timestamp}", ln=True)
    pdf.ln(5)

    # Patient Info
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

    # ORIGINAL X-RAY
    if image_path and os.path.exists(image_path):
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 8, "Original Chest X-ray", ln=True)
        pdf.ln(3)
        pdf.image(image_path, x=30, w=150)
        pdf.ln(80)

    # HEATMAP SECTION
    if heatmap_path and os.path.exists(heatmap_path):
        pdf.add_page()  # NEW PAGE FOR HEATMAP
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, "AI Localization (Grad-CAM Heatmap)", ln=True)
        pdf.ln(5)
        pdf.image(heatmap_path, x=30, w=150)
        pdf.ln(80)

    # Probability
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "AI Probability Assessment", ln=True)
    pdf.set_font("DejaVu", "", 10)

    pdf.multi_cell(0, 6,
        f"Tuberculosis Probability: {tb_prob*100:.2f}%\n"
        f"Normal Probability: {normal_prob*100:.2f}%"
    )
    pdf.ln(5)

    pdf.set_font("DejaVu", "", 9)
    pdf.multi_cell(0, 5,
        "Red/yellow regions indicate areas of highest AI attention "
        "associated with tuberculosis-related radiographic patterns."
    )

    report_name = f"{report_id}_TB_Report.pdf"
    pdf.output(report_name)

    return report_name
# -------------------------------
# MAIN APP
# -------------------------------
st.title("🫁 Tuberculosis Detection AI")

model = load_tb_model()
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

        heatmap_path = None

        if tb_prob > 0.5:
            msg = "Positive for Tuberculosis"

            heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2")

            if heatmap is not None:

                heatmap = cv2.resize(heatmap, (img.width, img.height))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                superimposed_img = cv2.addWeighted(
                    original_img,
                    0.4,
                    heatmap,
                    0.6,
                    0
                )
                
        # 🔥 SAVE HEATMAP HERE
                heatmap_path = "temp_heatmap.png"
                cv2.imwrite(heatmap_path, superimposed_img)
                superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

                st.subheader("📍 AI Highlighted TB Region")
                st.image(superimposed_img, use_container_width=True)

                heatmap_path = "temp_heatmap.png"
                cv2.imwrite(heatmap_path, superimposed_img)

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

        temp_image_path = "temp_image.png"
        img.save(temp_image_path)
        if heatmap_path:
            st.write("Heatmap file exists:", os.path.exists(heatmap_path))


        report_file = generate_pdf(
            uploaded_file.name,
            tb_prob,
            normal_prob,
            temp_image_path,
            heatmap_path,
            patient_name or "N/A",
            patient_age,
            patient_gender,
            patient_id or "N/A",
            referring_physician or "N/A"
        )

        os.remove(temp_image_path)

        if heatmap_path and os.path.exists(heatmap_path):
            os.remove(heatmap_path)

        with open(report_file, "rb") as f:
            st.download_button(
                "⬇️ Download Report",
                f,
                file_name=report_file,
                mime="application/pdf"
            )
