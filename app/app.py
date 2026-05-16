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
from datetime import datetime
from st_supabase_connection import SupabaseConnection

# ====================================================================
# 1. CONFIGURATION (PORTABLE CLOUD PATHS)
# ====================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Paths for Models & Data
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Scripts for Retraining
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
# 2. CLOUD DATABASE & IDENTITY (SECURED VIA INHERITANCE)
# ====================================================================
try:
    # SECURED: Pulls implicitly from local secrets configurations or
    # the advanced environment variables panel on the Streamlit web cloud.
    conn = st.connection("supabase", type=SupabaseConnection)
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
        # Graceful logging in case table schemas mismatch on runtime
        st.sidebar.error(f"Write Transaction Rejected: {db_error}")
        return False


def verify_user_cloud(v_id, input_key):
    try:
        query = conn.table("user_identities").select("*").eq("visitor_id", v_id).eq("permanent_key",
                                                                                    str(input_key)).execute()
        return len(query.data) > 0
    except:
        return False


# ====================================================================
# 3. BACKEND LOGIC CLASS
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
        if os.path.exists(MODEL_PATH):
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
            st.error("⚠️ Model architectural weights not found. Run validation/training routines.")

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
            st.info("🧠 Recalculating predictive decision boundaries...")
            subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
            subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
            self.load_resources()
            return True, "✅ Update Complete! System learned the requested vectors."
        except Exception as e:
            return False, f"Retraining lifecycle aborted: {e}"

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
# 4. MAIN APP INTERFACE
# ====================================================================
def main():
    st.set_page_config(page_title="Medical AI Chat", page_icon="🛡️", layout="centered")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()
    if 'auth' not in st.session_state:
        st.session_state.auth = False

    v_id = get_visitor_id()

    # --- SIDEBAR: TIERED ACCESS ---
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
                        st.error("Invalid Email.")
        else:
            st.success("✅ Professional Access Active")
            if st.button("Logout"):
                st.session_state.auth = False
                st.rerun()
            st.divider()
            st.subheader("Clinical Data Upload")
            st.file_uploader("Upload Patient Report", type=["pdf", "png", "jpg"])

    # --- MAIN CHAT AREA ---
    st.title("💬 AI Health Assistant")
    if not st.session_state.auth:
        st.caption("🟢 Guest Mode: Symptom analysis is active. Login for report analysis.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can identify health risks. How are you feeling?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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