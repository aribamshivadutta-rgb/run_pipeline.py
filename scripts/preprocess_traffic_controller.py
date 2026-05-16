import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


def preprocess_medical_data():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    TRAFFIC_CLEAN_DIR = os.path.join(PROJECT_ROOT, 'data', 'clean', 'traffic_clean')
    os.makedirs(TRAFFIC_CLEAN_DIR, exist_ok=True)

    all_data = []

    # --- SOURCE 1: mtsamples.csv (The "Stubborn" File) ---
    mtsamples_path = os.path.join(RAW_DIR, 'mtsamples.csv')
    temp_path = os.path.join(RAW_DIR, 'mtsamples_temp.csv')

    if os.path.exists(mtsamples_path):
        try:
            shutil.copy2(mtsamples_path, temp_path)
            mt_df = pd.read_csv(temp_path, encoding='latin1', on_bad_lines='skip', engine='python')

            col = 'description' if 'description' in mt_df.columns else mt_df.columns[0]
            reports = mt_df[col].dropna().unique().tolist()
            for r in reports:
                all_data.append({"text": str(r)[:250], "label": 1})
            print(f"[SUCCESS] Added {len(reports)} entries from MTSamples.")

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"[BYPASS] MTSamples is still locked by Windows. Skipping this source for now.")
            print(f"Details: {e}")

    # --- SOURCE 2: RxHandBD-Raw ---
    rx_path = os.path.join(RAW_DIR, 'RxHandBD-Raw', 'train_labels.csv')
    if os.path.exists(rx_path):
        try:
            rx_df = pd.read_csv(rx_path)
            meds = rx_df.iloc[:, 1].dropna().unique().tolist()
            for m in meds:
                all_data.append({"text": str(m), "label": 0})
            print(f"[SUCCESS] Added {len(meds)} entries from RxHandBD.")
        except Exception as e:
            print(f"[SKIP] RxHandBD error: {e}")

    # --- SOURCE 3: Symptom-severity.csv ---
    severity_path = os.path.join(RAW_DIR, 'Symptom-severity.csv')
    if os.path.exists(severity_path):
        try:
            sev_df = pd.read_csv(severity_path)
            symptoms = sev_df['Symptom'].str.replace('_', ' ').unique().tolist()
            for s in symptoms:
                all_data.append({"text": str(s), "label": 0})
            print(f"[SUCCESS] Added {len(symptoms)} entries from Symptoms.")
        except Exception as e:
            print(f"[SKIP] Symptom data error: {e}")

    # --- DATA EXPORT WITH TRAIN/TEST SPLIT ---
    if all_data:
        full_df = pd.DataFrame(all_data)

        # Save complete dataset for record-keeping
        master_output_file = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_data_clean.csv')
        full_df.to_csv(master_output_file, index=False)

        # Split data into 80% Train and 20% Test
        # 'stratify=y' ensures both files maintain the exact same ratio of Prescriptions to Reports
        X = full_df['text']
        y = full_df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Reconstruct into clean dataframes
        train_df = pd.DataFrame({"text": X_train, "label": y_train})
        test_df = pd.DataFrame({"text": X_test, "label": y_test})

        # File paths
        train_file = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_train.csv')
        test_file = os.path.join(TRAFFIC_CLEAN_DIR, 'traffic_test.csv')

        # Export individual files
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print("-" * 40)
        print(f"TOTAL MASTER ENTRIES: {len(full_df)}")
        print(f"TRAIN SPLIT SAVED ({len(train_df)} rows): {train_file}")
        print(f"TEST SPLIT SAVED ({len(test_df)} rows): {test_file}")
        print("-" * 40)
    else:
        print("[CRITICAL] All files are locked or missing. Please restart your PC.")


if __name__ == "__main__":
    preprocess_medical_data()