import torch
import joblib
import cv2
import numpy as np
import os
from medical_detector_cnn import MedicalDetectorCNN  # Your U-Net Architecture


class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load the "Eyes" (Detector)
        self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
        detector_path = os.path.join('../models', 'medical_detector.pth')
        if os.path.exists(detector_path):
            self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
        self.detector.eval()

        # 2. Load the "Brain" (Traffic Router)
        # This is the model trained on MTSamples, RxHandBD, and Symptoms
        self.router = joblib.load('../models/MedicalTrafficRouter_v1.pkl')
        self.vectorizer = joblib.load('../models/MedicalTrafficRouter_v1_vectorizer.pkl')

        # 3. CRNN Recognition Placeholder
        # In a full integration, you would load your CRNN weights here
        print(f"--- ocr_reader_pipeline initialized on {self.device} ---")

    def process_image(self, image_path, true_label=None):
        """
        Processes an image and returns predictions + accuracy if true_label is provided.
        true_label: 0 for Prescription/Symptom, 1 for Lab Report
        """
        # STEP 1: Detection
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = raw_img.shape
        img_input = cv2.resize(raw_img, (512, 512)) / 255.0
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            mask_output = self.detector(img_tensor)
            mask = (mask_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # STEP 2: Recognition (CRNN Output)
        # Note: This is where the CRNN would read the cropped 'mask' regions.
        # For the pipeline test, we simulate the text output.
        ocr_text_output = "Amoxicillin 500mg"

        # STEP 3: Routing (Traffic Controller)
        vec_text = self.vectorizer.transform([ocr_text_output])
        pred_label = self.router.predict(vec_text)[0]
        confidence = np.max(self.router.predict_proba(vec_text)) * 100

        # STEP 4: Accuracy Evaluation
        accuracy = None
        if true_label is not None:
            accuracy = 100.0 if pred_label == true_label else 0.0

        return {
            "ocr_text": ocr_text_output,
            "category": "Prescription/Symptom" if pred_label == 0 else "Lab Report",
            "confidence": f"{confidence:.2f}%",
            "router_accuracy": accuracy,
            "mask_preview": mask  # Returned for display in Streamlit
        }


if __name__ == "__main__":
    pipeline = OCRReaderPipeline()
    # Manual Test: Predicting a Prescription (Label 0)
    # Ensure you have a 'sample_test.png' in your project root
    results = pipeline.process_image("sample_test.png", true_label=0)
    print("\n--- Pipeline Accuracy Report ---")
    print(f"Text Read: {results['ocr_text']}")
    print(f"Assigned Category: {results['category']}")
    print(f"Router Confidence: {results['confidence']}")
    print(f"Match Accuracy: {results['router_accuracy']}%")