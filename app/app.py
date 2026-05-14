import streamlit as st
import os
import sys
import pandas as pd
import joblib
import re
import difflib
import requests
import csv
import subprocess
import hashlib
import random
import uuid
from datetime import datetime
from bs4 import BeautifulSoup

# =======================
# 1. CONFIGURATION (PORTABLE PATHS)
# =======================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Paths for Models & Data
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")

# Specific Files
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Scripts to Trigger
PREPROCESS_SCRIPT = os.path.join(CURRENT_SCRIPT_DIR, "chat_bot_preprocessing.py")
TRAIN_SCRIPT = os.path.join(CURRENT_SCRIPT_DIR, "train_lgbm.py")

APP_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "app_data")
INFO_DB_PATH = os.path.join(APP_DATA_DIR, "who_data_clean.csv")
USER_DB_PATH = os.path.join(APP_DATA_DIR, "user_keys.csv")

os.makedirs(APP_DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "sugar": "diabetes",
    "bp": "hypertension",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


# =======================
# 2. IDENTITY HELPERS
# =======================

def get_visitor_id():
    if 'visitor_id' not in st.session_state:
        st.session_state.visitor_id = str(uuid.getnode())
    return st.session_state.visitor_id


def generate_permanent_key(email):
    hash_obj = hashlib.sha256(email.strip().lower().encode())
    seed = int(hash_obj.hexdigest(), 16) % 10 ** 8
    random.seed(seed)
    return str(random.randint(100000, 999999))


def save_user_key(v_id, email, key):
    new_row = pd.DataFrame([[v_id, email, str(key)]], columns=['visitor_id', 'email', 'permanent_key'])
    if os.path.exists(USER_DB_PATH):
        df = pd.read_csv(USER_DB_PATH)
        df = df[df['email'] != email]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(USER_DB_PATH, index=False)


def verify_user_key(v_id, input_key):
    if not os.path.exists(USER_DB_PATH): return False
    df = pd.read_csv(USER_DB_PATH, dtype={'permanent_key': str})
    match = df[(df['visitor_id'] == v_id) & (df['permanent_key'] == str(input_key))]
    return not match.empty


# =======================
# 3. BACKEND LOGIC CLASS
# =======================
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
                st.error(f"Error loading model files: {e}")
        else:
            st.error(f"⚠️ Model not found at: {MODEL_PATH}")

    def log_learning_request(self, disease_name):
        required_columns = ["timestamp", "source_url", "proposed_disease", "symptoms", "status"]
        if not os.path.exists(REQUESTS_FILE):
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(required_columns)
        try:
            with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(
                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "User App", disease_name, "Pending", "Pending"])
            return True
        except:
            return False

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        return [col.replace("_", " ") for col in self.known_symptoms if subset.iloc[0][col] == 1]

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        found_text = []
        source = "WHO"
        try:
            url = f"https://www.who.int/news-room/fact-sheets/detail/{clean_name.replace(' ', '-')}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(k in h.get_text() for k in ["Prevention", "Treatment"]):
                        for tag in h.find_next_siblings(['p', 'ul'])[:3]: found_text.append(tag.get_text().strip())
                        break
        except:
            pass
        return found_text, source

    def predict(self, user_input):
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            m = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.7)
            if m: input_dict[m[0]] = 1; matched.append(m[0])
        if not matched: return None, [], 0
        input_df = pd.DataFrame([input_dict])
        pred_id = self.model.predict(input_df)[0]
        conf = self.model.predict_proba(input_df)[0][pred_id] * 100
        return self.le.inverse_transform([pred_id])[0], matched, conf


# =======================
# 4. MAIN APP WITH TIERED ACCESS
# =======================
def main():
    st.set_page_config(page_title="AI Health Assistant", page_icon="🛡️", layout="centered")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    v_id = get_visitor_id()

    # --- SIDEBAR: SECURE FEATURES ---
    with st.sidebar:
        st.header("Medical Vault")
        st.caption(f"ID: {v_id}")

        if not st.session_state.authenticated:
            st.warning("Locked Features")
            st.info("Unlock to use personal records or upload clinical reports.")

            tab_login, tab_reg = st.tabs(["Unlock", "Register"])
            with tab_login:
                pin = st.text_input("Enter 6-Digit Key", type="password")
                if st.button("Unlock Now"):
                    if verify_user_key(v_id, pin):
                        st.session_state.authenticated = True
                        st.success("Unlocked!")
                        st.rerun()
                    else:
                        st.error("Invalid Key")

            with tab_reg:
                email = st.text_input("Registration Email")
                if st.button("Get Permanent Key"):
                    if "@" in email:
                        k = generate_permanent_key(email)
                        save_user_key(v_id, email, k)
                        st.success(f"Key: **{k}**")
                    else:
                        st.error("Invalid Email")
        else:
            st.success("✅ Secure Access Active")
            st.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
            st.divider()
            st.subheader("Upload Clinical Data")
            uploaded_file = st.file_uploader("Choose a report (PDF/JPG)", type=["pdf", "jpg", "png"])
            if uploaded_file:
                st.write("File detected. Starting OCR Analysis...")

    # --- MAIN CHAT AREA (PUBLIC ACCESS) ---
    st.title("💬 AI Health Assistant")

    if not st.session_state.authenticated:
        st.caption("🟢 Guest Mode Active: Symptom analysis is available. Login via sidebar for report analysis.")
    else:
        st.caption("🔒 Professional Mode Active: Clinical reporting enabled.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                                      "content": "Hello! I can help identify health risks based on symptoms. How are you feeling today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms (e.g. fever, cough)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        bot = st.session_state.bot
        query_lower = prompt.lower().strip()

        # Chat Logic
        search_term = DISEASE_ALIASES.get(query_lower, query_lower)
        matches = difflib.get_close_matches(search_term, bot.known_diseases, n=1, cutoff=0.85)

        disease_found = matches[0] if matches else None
        if not disease_found:
            disease, matched, conf = bot.predict(query_lower)
            if matched:
                response_text = f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}% confidence)\n"
                disease_found = disease
            else:
                response_text = "I couldn't recognize those symptoms. Please try standard terms like 'headache' or 'fatigue'."
        else:
            response_text = f"✅ Information found for **{disease_found.title()}**.\n"

        if disease_found:
            syms = bot.get_symptoms(disease_found)
            if syms: response_text += f"\n**Common Symptoms:** {', '.join(syms[:6])}"
            adv, src = bot.get_advice(disease_found)
            if adv: response_text += f"\n\n**Suggestions ({src}):**\n- " + "\n- ".join(adv[:3])

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


if __name__ == "__main__":
    main()