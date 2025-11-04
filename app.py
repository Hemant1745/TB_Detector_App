# Trigger redeploy on Render (added font fix)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import psycopg2
import bcrypt
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from fpdf import FPDF
import re

# Load environment variables
load_dotenv()

# -------------------------------
# 🎨 Streamlit Config
# -------------------------------
st.set_page_config(page_title="🫁 TB Detection AI", page_icon="🧬", layout="wide")

# -------------------------------
# 🌈 Styling
# -------------------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white;}
h1, h2, h3 {color: #00E6FF; text-align: center;}
.result-card {
    background-color: rgba(255,255,255,0.1);
    padding: 1.5em; border-radius: 15px; text-align: center; margin-top: 1em;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.stButton > button {
    border-radius: 10px;
    background-color: #0077B6;
    color: white;
    border: none;
    font-size: 16px;
    padding: 0.5em 1.5em;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #00B4D8;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🗄️ Database Connection
# -------------------------------
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )

# -------------------------------
# 🔐 Security Helpers
# -------------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# -------------------------------
# 🧠 Load Model (cached)
# -------------------------------
import os
import gdown
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_tb_model():
    model_path = "model/my_model.keras"
    if not os.path.exists(model_path):
        st.info("📥 Downloading AI model...")
        os.makedirs("model", exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1C8fXLwrfV_ZV4xjYwJgdejLzBp7-rboi"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path)
    return model

# -------------------------------
# 🧾 Generate PDF Report with X-ray Thumbnail
# -------------------------------
import re
from fpdf import FPDF

def generate_pdf(username, filename, tb_prob, normal_prob, result_text, image_path):
    pdf = FPDF()
    pdf.add_page()

    # 🧠 Try to load Unicode font safely
    font_path = "fonts/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 14)
    else:
        pdf.set_font("Helvetica", "", 14)

    # 🧹 Clean text to remove emojis or unsupported chars
    def clean_text(txt):
        return re.sub(r"[^\x00-\x7F]+", "", txt)

    pdf.cell(200, 10, txt=clean_text("Tuberculosis Detection Report"), ln=True, align="C")
    pdf.ln(10)

    # 🩻 Add X-ray thumbnail if exists
    if os.path.exists(image_path):
        try:
            pdf.image(image_path, x=60, y=25, w=90)
            pdf.ln(85)
        except Exception:
            pdf.ln(20)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(200, 10, txt=f"Patient/User: {clean_text(username)}", ln=True)
    pdf.cell(200, 10, txt=f"File: {clean_text(filename)}", ln=True)
    pdf.cell(200, 10, txt=f"TB Probability: {tb_prob*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Normal Probability: {normal_prob*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Diagnosis: {clean_text(result_text)}", ln=True)
    pdf.cell(200, 10, txt=f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    report_name = f"report_{username}_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(report_name)
    return report_name

# -------------------------------
# 📝 Register User
# -------------------------------
def register_user():
    st.title("📝 Register New Account")
    username = st.text_input("👤 Username")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Register"):
            if username and email and password:
                hashed_pwd = hash_password(password)
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                                (username, email, hashed_pwd))
                    conn.commit()
                    st.success("✅ Registration successful! You can now log in.")
                except Exception as e:
                    st.error(f"⚠️ Registration failed: {e}")
                finally:
                    cur.close()
                    conn.close()
            else:
                st.warning("Please fill in all fields.")
    with col2:
        if st.button("⬅️ Back to Login"):
            st.session_state["show_register"] = False
            st.rerun()

# -------------------------------
# 🔐 Reset Password
# -------------------------------
def reset_password():
    st.title("🔄 Reset Your Password")
    email = st.text_input("📧 Enter your registered email")
    new_password = st.text_input("🔑 New Password", type="password")
    confirm_password = st.text_input("🔁 Confirm Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Reset Password"):
            if email and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                    return
                hashed_new = hash_password(new_password)
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hashed_new, email))
                    if cur.rowcount == 0:
                        st.error("❌ Email not found.")
                    else:
                        conn.commit()
                        st.success("✅ Password updated successfully. You can now log in.")
                except Exception as e:
                    st.error(f"⚠️ Reset failed: {e}")
                finally:
                    cur.close()
                    conn.close()
            else:
                st.warning("Please enter your email and new password.")
    with col2:
        if st.button("⬅️ Back to Login"):
            st.session_state["show_reset"] = False
            st.rerun()

# -------------------------------
# 🔐 Login Page
# -------------------------------
def login_page():
    st.title("🔐 Login to Tuberculosis Detection AI")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    col1, col2, col3 = st.columns(3)
    with col1:
        login_btn = st.button("Login")
    with col2:
        register_btn = st.button("Register")
    with col3:
        forgot_btn = st.button("Forgot Password")

    if login_btn:
        if username and password:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE username=%s", (username,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result and check_password(password, result[0]):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
        else:
            st.warning("Please enter both fields.")

    if register_btn:
        st.session_state["show_register"] = True
        st.rerun()

    if forgot_btn:
        st.session_state["show_reset"] = True
        st.rerun()

# -------------------------------
# 🧬 TB Detection + History
# -------------------------------
def main_app():
    with st.spinner("Loading AI model... please wait..."):
        model = load_tb_model()
    st.success(f"✅ Model loaded successfully! Welcome, {st.session_state['username']}")

    st.sidebar.title("🧬 Tuberculosis Detection AI")
    st.sidebar.write(f"👤 Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("📜 View History"):
        show_history(st.session_state["username"])
        return
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.title("🫁 Tuberculosis Detection System")
    uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            with st.spinner("Analyzing image..."):
                pred = model.predict(img_array)[0][0]
                tb_prob = float(pred)
                normal_prob = 1 - tb_prob

            if tb_prob > 0.5:
                msg, color = "⚠️ Tuberculosis Detected", "#FF6B6B"
            elif 0.3 <= tb_prob <= 0.5:
                msg, color = "⚠️ Tuberculosis Might Affect", "#F1C40F"
            else:
                msg, color = "✅ Normal Chest X-ray", "#2ECC71"

            st.markdown(f"<div class='result-card'><h2 style='color:{color};'>{msg}</h2><p>🧮 TB Probability: {tb_prob*100:.2f}%</p></div>", unsafe_allow_html=True)
            st.progress(tb_prob)
            st.text(f"Normal Probability: {normal_prob:.4f}")

            # Save analysis in database
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO tb_history (username, filename, tb_probability, normal_probability, result_text)
                    VALUES (%s, %s, %s, %s, %s)
                """, (st.session_state["username"], uploaded_file.name, tb_prob, normal_prob, msg))
                conn.commit()
                cur.close()
                conn.close()
                st.success("📦 Scan result saved to your history.")
            except Exception as e:
                st.warning(f"⚠️ Could not save to history: {e}")

            # Generate PDF with X-ray
            temp_image_path = f"temp_{uploaded_file.name}"
            Image.open(uploaded_file).save(temp_image_path)
            report_file = generate_pdf(st.session_state["username"], uploaded_file.name, tb_prob, normal_prob, msg, temp_image_path)
            os.remove(temp_image_path)

            with open(report_file, "rb") as f:
                st.download_button("⬇️ Download Report", f, file_name=report_file, mime="application/pdf")

            st.markdown(f"🕓 *Analysis completed at:* {datetime.now().strftime('%H:%M:%S')}")

# -------------------------------
# 📜 History Viewer
# -------------------------------
def show_history(username):
    st.title("📜 Past Scan History")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename, tb_probability, normal_probability, result_text, analysis_time FROM tb_history WHERE username=%s ORDER BY analysis_time DESC", (username,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        st.info("No history found yet. Upload a scan to start tracking results.")
        return

    for file, tb, normal, result, time in rows:
        st.markdown(f"""
        <div class='result-card'>
            <h3>{file}</h3>
            <p>🧮 TB Probability: {tb*100:.2f}% | 🩺 Normal: {normal*100:.2f}%</p>
            <p>🕓 {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><b>{result}</b></p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Dashboard"):
        st.rerun()

# -------------------------------
# 🚀 App Controller
# -------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "show_register" not in st.session_state:
    st.session_state["show_register"] = False
if "show_reset" not in st.session_state:
    st.session_state["show_reset"] = False

if st.session_state["show_register"]:
    register_user()
elif st.session_state["show_reset"]:
    reset_password()
elif not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
