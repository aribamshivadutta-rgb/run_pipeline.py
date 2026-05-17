import streamlit as st
import os
import sys
import pandas as pd
import joblib
import re
import difflib
import csv
import subprocess
import hashlib
import random
import uuid
import torch
import torch.nn as nn  # <-- Added explicitly for native layer definitions
import cv2
import numpy as np
from datetime import datetime
from st_supabase_connection import SupabaseConnection
from pdf2image import convert_from_bytes

# Avoid relative import breakages by dynamically adding scripts path to system environment
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

try:
    # Explicitly importing your structural U-Net model from your scripts folder
    from scripts.medical_detector_cnn import MedicalDetectorCNN
except ImportError:
    # Fallback placeholder if layout shifts during runtime context
    MedicalDetectorCNN = None

# ====================================================================
# 1. PORTABLE CORE FILE-SYSTEM PATH ENGINES
# ====================================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")

# Chatbot System Resource Paths
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Computer Vision Network Weights
DETECTOR_WEIGHTS = os.path.join(MODEL_DIR, "medical_detector.pth")
CRNN_WEIGHTS = os.path.join(MODEL_DIR, "MedicalCRNN_v1.pth")  # <-- Native Weights Mapping
TRAFFIC_ROUTER_WEIGHTS = os.path.join(MODEL_DIR, "MedicalTrafficRouter_v1.pkl")
TRAFFIC_VECTORIZER_WEIGHTS = os.path.join(MODEL_DIR, "MedicalTrafficRouter_v1_vectorizer.pkl")

# Subprocess Retraining Anchors
PREPROCESS_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "chat_bot_preprocessing.py")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_lgbm.py")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection", "cold": "upper respiratory infection",
    "flu": "influenza", "sugar": "diabetes", "bp": "hypertension",
    "heart attack": "myocardial infarction", "brain stroke": "cerebrovascular accident"
}

# ====================================================================
# 2. CLOUD DATABASE MANAGEMENT (SUPABASE INTEGRATION)
# ====================================================================
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url="https://cwwoloupweulprxwibmp.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN3d29sb3Vwd2V1bHByeHdpYm1wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg3MDA5NDEsImV4cCI6MjA5NDI3Njk0MX0.ggPfeYBaL7PLiEM8_fYI5fHo48obb5yRum_kR1CORNM"
    )
except Exception as e:
    st.error(f"⚠️ Database Connection Failed: {e}")
    st.stop()


def get_visitor_id():
    if 'visitor_id' not in st.session_state:
        st.session_state.visitor_id = str(uuid.getnode())
    return st.session_state.visitor_id


def generate_permanent_key(email):
    hash_obj = hashlib.sha256(email.strip().lower().encode())
    seed = int(hash_obj.hexdigest(), 16) % 10 ** 8
    random.seed(seed)
    return str(random.randint(100000, 999999))


def save_user_cloud(v_id, email, key):
    try:
        conn.table("user_identities").upsert({
            "visitor_id": v_id, "email": email, "permanent_key": str(key)
        }).execute()
        return True
    except Exception as db_error:
        st.sidebar.error(f"Database Write Error: {db_error}")
        return False


def verify_user_cloud(v_id, input_key):
    try:
        query = conn.table("user_identities").select("*").eq("visitor_id", v_id).eq("permanent_key",
                                                                                    str(input_key)).execute()
        return len(query.data) > 0
    except:
        return False


# ====================================================================
# 3. DETECTOR & RECOGNITION DEEP LEARNING PIPELINE (CRNN ALIGNED)
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def decode(self, nums):
        res = []
        for i, num in enumerate(nums):
            if num != 0 and (i == 0 or num != nums[i - 1]):
                res.append(self.num_to_char.get(num, ""))
        return "".join(res)

    @property
    def vocab_size(self):
        return len(self.chars) + 1


class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(2048, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = x.fc(x)
        return x.log_softmax(2)


class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.text_recognizer = None
        self.encoder = MedicalLabelEncoder()
        self.router = None
        self.vectorizer = None
        self.load_models()

    def load_models(self):
        # 1. Build and Initialize U-Net Architecture
        if MedicalDetectorCNN is not None:
            self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
            if os.path.exists(DETECTOR_WEIGHTS):
                self.detector.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location=self.device))
            self.detector.eval()

        # 2. Build and Initialize Native MedicalCRNN Architecture
        self.text_recognizer = MedicalCRNN(self.encoder.vocab_size).to(self.device)
        if os.path.exists(CRNN_WEIGHTS):
            self.text_recognizer.load_state_dict(torch.load(CRNN_WEIGHTS, map_location=self.device))
        self.text_recognizer.eval()

        # 3. Extract Traffic Controller Serializations
        if os.path.exists(TRAFFIC_ROUTER_WEIGHTS) and os.path.exists(TRAFFIC_VECTORIZER_WEIGHTS):
            self.router = joblib.load(TRAFFIC_ROUTER_WEIGHTS)
            self.vectorizer = joblib.load(TRAFFIC_VECTORIZER_WEIGHTS)

    def process_image(self, image_input, true_label=None):
        """
        Hybrid Vector Handler parsing file string paths OR binary web buffers (Images and PDFs)
        """
        raw_img = None

        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            filename = getattr(image_input, 'name', '').lower()

            if filename.endswith('.pdf'):
                pdf_bytes = image_input.read()
                image_input.seek(0)
                pil_pages = convert_from_bytes(pdf_bytes)

                if len(pil_pages) > 0:
                    rgb_page = np.array(pil_pages[0])
                    raw_img = cv2.cvtColor(rgb_page, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError("The uploaded PDF contains no processable pages.")
            else:
                file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
                image_input.seek(0)
                raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if raw_img is None:
            raise ValueError("File content empty or corrupt array stream presented.")

        # Save an isolation copy for CRNN text compilation before resizing to U-Net proportions
        img_for_crnn = raw_img.copy()

        h, w = raw_img.shape
        img_input = cv2.resize(raw_img, (512, 512)) / 255.0
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # STEP 1: Execute Deep Feature Extraction (U-Net)
        mask = np.zeros((512, 512), dtype=np.uint8)
        if self.detector is not None:
            with torch.no_grad():
                mask_output = self.detector(img_tensor)
                mask = (mask_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # STEP 2: Structural Character Recognition (Live Custom CRNN Alignment)
        ocr_text_output = ""
        if self.text_recognizer is not None:
            # Replicate custom MedicalDataset transformations exactly
            crnn_input = cv2.resize(img_for_crnn, (256, 64))
            crnn_input = (crnn_input / 255.0 - 0.5) / 0.5
            crnn_tensor = torch.from_numpy(crnn_input).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                preds = self.text_recognizer(crnn_tensor)
                best_path = torch.argmax(preds, dim=2).squeeze(0).cpu().numpy()
                ocr_text_output = self.encoder.decode(best_path).strip()

        if not ocr_text_output:
            ocr_text_output = "No readable text extracted."

        # STEP 3: Compute Linear Vector Routing (LightGBM Decision Matrix)
        category_label = "Prescription/Symptom"
        confidence_score = 100.0

        if self.router and self.vectorizer and ocr_text_output != "No readable text extracted.":
            vec_text = self.vectorizer.transform([ocr_text_output])
            pred_label = self.router.predict(vec_text)[0]
            confidence_score = np.max(self.router.predict_proba(vec_text)) * 100
            category_label = "Prescription/Symptom" if pred_label == 0 else "Lab Report"

        # STEP 4: Accuracy Compilation
        accuracy = None
        if true_label is not None and self.router:
            accuracy = 100.0 if pred_label == true_label else 0.0

        return {
            "ocr_text": ocr_text_output,
            "category": category_label,
            "confidence": f"{confidence_score:.2f}%",
            "router_accuracy": accuracy,
            "mask_preview": mask
        }


# ====================================================================
# 4. CHATBOT AND CLASSIFICATION EXPERT LAYER
# ====================================================================
class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.df_full = None
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(LE_PATH)
                self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]
                if os.path.exists(FULL_DATA_PATH):
                    self.df_full = pd.read_csv(FULL_DATA_PATH)
            except Exception as e:
                st.error(f"Resource Load Error: {e}")
        else:
            st.error("⚠️ Model architectural weights not found. Run classification compilation routines.")

    def log_learning_request(self, disease_name):
        if not os.path.exists(REQUESTS_FILE):
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["timestamp", "source_url", "proposed_disease", "symptoms", "status"])
        with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "App", disease_name, "Pending", "Pending"])
        return True

    def execute_verification_cycle(self):
        try:
            st.info("🧠 Recalculating machine learning weights...")
            subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
            subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
            self.load_resources()
            return True, "✅ Update Complete! I have learned the new diseases."
        except Exception as e:
            return False, f"Retraining lifecycle bypassed: {e}"

    def predict(self, user_input):
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            m = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.7)
            if m:
                input_dict[m[0]] = 1
                matched.append(m[0])
            else:
                for k in self.known_symptoms:
                    if t in k.replace("_", " "):
                        input_dict[k] = 1
                        matched.append(k)
                        break
        if not matched:
            return None, [], 0

        pred_id = self.model.predict(pd.DataFrame([input_dict]))[0]
        conf = self.model.predict_proba(pd.DataFrame([input_dict]))[0][pred_id] * 100
        return self.le.inverse_transform([pred_id])[0], list(set(matched)), conf


# ====================================================================
# 5. CORE SYSTEM PRESENTATION WORKSPACE
# ====================================================================
def main():
    st.set_page_config(page_title="Medical AI Chat", page_icon="🛡️", layout="centered")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()
    if 'auth' not in st.session_state:
        st.session_state.auth = False

    v_id = get_visitor_id()

    # --- SIDEBAR CONTROL ROOM ---
    with st.sidebar:
        st.header("🔐 Secure Vault")
        st.caption(f"Hardware ID: `{v_id}`")

        if not st.session_state.auth:
            st.warning("Locked Mode: Chat only.")
            tab_unlock, tab_reg = st.tabs(["Unlock", "Register"])
            with tab_unlock:
                pin = st.text_input("Enter 6-Digit Key", type="password")
                if st.button("Unlock Features"):
                    if verify_user_cloud(v_id, pin):
                        st.session_state.auth = True
                        st.rerun()
                    else:
                        st.error("Invalid Key for this device.")
            with tab_reg:
                mail = st.text_input("Email for Key")
                if st.button("Generate Key"):
                    if "@" in mail:
                        k = generate_permanent_key(mail)
                        if save_user_cloud(v_id, mail, k):
                            st.success(f"Permanent Key: **{k}**")
                    else:
                        st.error("Invalid Email Structure.")
        else:
            st.success("✅ Professional Access Active")
            if st.button("Logout"):
                st.session_state.auth = False
                st.rerun()

            # --- COMPUTER VISION ACCELERATED INFERENCE CORE ---
            st.divider()
            st.subheader("Clinical Data Upload")
            uploaded_file = st.file_uploader("Upload Patient Report", type=["pdf", "png", "jpg", "jpeg"])

            if uploaded_file is not None:
                st.sidebar.success("📦 Scanned file buffered successfully!")

                if 'ocr_pipeline' not in st.session_state:
                    st.session_state.ocr_pipeline = OCRReaderPipeline()

                try:
                    with st.spinner("🔬 Tensor Target Segmentation Active..."):
                        results = st.session_state.ocr_pipeline.process_image(uploaded_file, true_label=0)

                    st.sidebar.success("🎯 Analysis Complete!")

                    # Intercept extracted dynamic CRNN array text and bind to system conversation memory bank
                    st.session_state.extracted_file_text = results["ocr_text"]

                    tab_metrics, tab_mask = st.sidebar.tabs(["Analysis", "U-Net Mask"])
                    with tab_metrics:
                        st.metric("Inferred Category", results["category"])
                        st.metric("Router Confidence", results["confidence"])
                        st.text_area("Extracted Context Matrix", results["ocr_text"])

                    with tab_mask:
                        # 🌟 FIX: Updated use_column_width from boolean to standard string "always"
                        st.image(results["mask_preview"], caption="U-Net Segmented Mask", use_column_width="always")

                except Exception as eval_err:
                    st.sidebar.error(f"Inference Failure: {eval_err}")

    # --- MAIN ENGINE DIALOGUE AREA ---
    st.title("💬 AI Health Assistant")
    if not st.session_state.auth:
        st.caption("🟢 Guest Mode: Symptom analysis is active. Login for report analysis.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can identify health risks. How are you feeling?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Automated Pipeline Interceptor Hook: Evaluates text extracted via side uploads
    if 'extracted_file_text' in st.session_state and st.session_state.extracted_file_text:
        ocr_payload = st.session_state.extracted_file_text
        del st.session_state['extracted_file_text']  # Evacuate index payload to break runtime loop states

        st.session_state.messages.append({"role": "user", "content": f"📋 *[Uploaded Report Data]:* {ocr_payload}"})

        bot = st.session_state.bot
        disease, matched, conf = bot.predict(ocr_payload)

        if matched:
            response_text = f"⚙️ **Automated Report Diagnostics Active:**\n\n" \
                            f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}% confidence)\n" \
                            f"\n**Extracted Matching Features:** {', '.join(matched).replace('_', ' ')}"
        else:
            response_text = f"I detected '{ocr_payload}' in the document, but I couldn't map it cleanly to known symptoms in my classification database."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()

    if prompt := st.chat_input("Enter symptoms (e.g. fever, headache)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        bot = st.session_state.bot
        query_lower = prompt.lower().strip()

        if query_lower == "verify now":
            _, response_text = bot.execute_verification_cycle()
        elif query_lower.startswith("do you know "):
            disease = query_lower[12:].strip("? ")
            bot.log_learning_request(disease)
            response_text = f"📝 Logged: **{disease}**. Type 'verify now' to trigger training."
        else:
            disease, matched, conf = bot.predict(prompt)
            if matched:
                response_text = f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}% confidence)\n"
                response_text += f"\n**Matched Symptoms:** {', '.join(matched).replace('_', ' ')}"
            else:
                response_text = "I couldn't recognize those symptoms. Try 'Do you know [Disease]?' to teach me."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


if __name__ == "__main__":
    main()