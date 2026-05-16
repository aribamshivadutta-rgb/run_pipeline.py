import os
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train_and_validate_router():
    # --- 1. PATH SETUP ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    TRAFFIC_CLEAN_DIR = os.path.join(PROJECT_ROOT, 'data', 'clean', 'traffic_clean')
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    TRAIN_PATH = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_train.csv')
    TEST_PATH = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_test.csv')

    # Fallback to single master file if split execution was bypassed due to file locks
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("⚠️ Split files not found. Falling back to master clean dataset...")
        TRAIN_PATH = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_data_clean.csv')
        TEST_PATH = None

    print(f"Loading training data from: {os.path.basename(TRAIN_PATH)}")
    train_df = pd.read_csv(TRAIN_PATH)
    X_train_text = train_df['text'].astype(str)
    y_train = train_df['label']

    # --- 2. VECTORIZATION (THE PREPROCESSOR) ---
    # character-level n-grams handle variations in handwriting OCR beautifully
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))

    print("Fitting TF-IDF Vectorizer on training text...")
    X_train = vectorizer.fit_transform(X_train_text)

    # --- 3. LIGHTGBM ARCHITECTURE ---
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.08,
        objective='binary',
        random_state=42,
        verbose=-1  # Suppresses excessive tree-splitting warnings
    )

    print("Training Medical Traffic Router (LightGBM)...")
    model.fit(X_train, y_train)

    # --- 4. MODEL EVALUATION (VALIDATION TESTING) ---
    if TEST_PATH and os.path.exists(TEST_PATH):
        print(f"Loading unseen test data from: {os.path.basename(TEST_PATH)}")
        test_df = pd.read_csv(TEST_PATH)
        X_test_text = test_df['text'].astype(str)
        y_test = test_df['label']

        # CRITICAL STEP: Strictly transform (do not fit) test entries
        X_test = vectorizer.transform(X_test_text)

        # Predict on validation data
        y_pred = model.predict(X_test)

        # Metrics Calculations
        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 40)
        print("📊 --- AI ROUTER EVALUATION REPORT ---")
        print(f"Overall Validation Accuracy: {acc * 100:.2f}%")
        print("=" * 40)
        print("\nDetailed Classification Matrix:")
        print(classification_report(y_test, y_pred, target_names=["Prescription/Symptom (0)", "Lab Report (1)"]))
        print("=" * 40 + "\n")
    else:
        print("\n⚠️ Skipping validation split run. Model trained on complete unified dataset.\n")

    # --- 5. SAVE EXPORT ASSETS ---
    model_file = os.path.join(MODEL_SAVE_DIR, 'MedicalTrafficRouter_v1.pkl')
    vec_file = os.path.join(MODEL_SAVE_DIR, 'MedicalTrafficRouter_v1_vectorizer.pkl')

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vec_file)

    print(f"SUCCESS: Model saved as MedicalTrafficRouter_v1.pkl")
    print(f"SUCCESS: Vectorizer saved as MedicalTrafficRouter_v1_vectorizer.pkl")
    print("-" * 40)


if __name__ == "__main__":
    train_and_validate_router()