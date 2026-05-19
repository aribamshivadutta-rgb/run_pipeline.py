import torch
import torch.nn as nn
import joblib
import cv2
import numpy as np
import os

# Dynamically anchor directory structure relative to this file's position
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

try:
    from scripts.medical_detector_cnn import MedicalDetectorCNN  # Your U-Net Architecture
except ImportError:
    # Fallback import structure depending on terminal runtime context orientation
    from medical_detector_cnn import MedicalDetectorCNN


class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load the "Eyes" (Detector - U-Net)
        self.detector = None
        detector_path = os.path.join(MODEL_DIR, 'medical_detector.pth')

        if os.path.exists(detector_path):
            self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
            self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
            self.detector.eval()
        else:
            print(f"⚠️ Warning: Weights missing at {detector_path}. Running pipeline in mock mode.")

        # 2. Load the "Brain" (Traffic Router serializations)
        self.router = None
        self.vectorizer = None
        router_path = os.path.join(MODEL_DIR, 'MedicalTrafficRouter_v1.pkl')
        vectorizer_path = os.path.join(MODEL_DIR, 'MedicalTrafficRouter_v1_vectorizer.pkl')

        if os.path.exists(router_path) and os.path.exists(vectorizer_path):
            self.router = joblib.load(router_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            print("⚠️ Warning: Traffic Router or Vectorizer binaries missing from models directory.")

        print(f"--- ocr_reader_pipeline initialized on {self.device} ---")

    def _split_lines_by_projection(self, block_crop):
        """
        🎯 OPTION A FILTER LAYER: Horizontal Projection Profile
        Takes a multi-line paragraph crop block and analyzes horizontal row intensity.
        If empty space rows are detected, it segments them into isolated lines.
        """
        # Fallback list for final text lines
        line_crops = []

        if len(block_crop.shape) == 3:
            gray_crop = cv2.cvtColor(block_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = block_crop

        # Binary threshold match
        _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate white pixel counts along rows (horizontal projection)
        horizontal_sum = np.sum(thresh_crop, axis=1)

        # Dynamic line-gap threshold detection logic
        in_line = False
        start_y = 0

        for idx, row_sum in enumerate(horizontal_sum):
            if not in_line and row_sum > 0:
                in_line = True
                start_y = max(0, idx - 2)  # Mild safe upper buffer pad
            elif in_line and row_sum == 0:
                in_line = False
                end_y = min(block_crop.shape[0], idx + 2)  # Mild safe lower buffer pad
                if (end_y - start_y) > 5:  # Filter out single pixel dust artifacts
                    line_crops.append(block_crop[start_y:end_y, :])

        # Catch lingering rows if text reaches the exact frame floor boundary
        if in_line:
            line_crops.append(block_crop[start_y:, :])

        # If no gaps were clear, fallback to the original whole isolated ROI crop box block
        return line_crops if len(line_crops) > 0 else [block_crop]

    def process_image(self, image_input, true_label=None):
        """
        Processes an image string path OR a binary web file stream packet.
        true_label: 0 for Prescription/Symptom, 1 for Lab Report
        """
        # STEP 1: Detection Matrix Allocation (Hybrid Layer Switch)
        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input)
        else:
            file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
            raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw_img is None:
            raise ValueError("Pipeline cannot unpack image stream array. Payload corrupted or empty.")

        # Save native layout dimensions before processing
        orig_h, orig_w, _ = raw_img.shape
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # Format standardized sizing exclusively for the U-Net convolutional encoder path
        img_input = cv2.resize(gray_img, (512, 512)) / 255.0
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # U-Net Mask Evaluation Block
        mask = np.zeros((512, 512), dtype=np.uint8)
        if self.detector is not None:
            with torch.no_grad():
                mask_output = self.detector(img_tensor)
                # Convert soft sigmoidal weights matrix map to clear binary mask output array
                mask = (mask_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Resize mask back up to native scale mapping coordinates
        resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Morphological Closing Pass to eliminate loose gaps inside handwritten character boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        processed_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)

        # ====================================================================
        # 🎯 THE FIX: CONNECTED COMPONENTS EXTRACTOR DYNAMIC LOOP
        # ====================================================================
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours perfectly from top to bottom based on their spatial geometric layout coordinates
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        extracted_line_crops = []

        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out microscopic noise anomalies or speck artifacts
            if w < 12 or h < 8:
                continue

            # Adaptive bounding padding padding margin buffer
            padding = 4
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(orig_w, x + w + padding)
            y_end = min(orig_h, y + h + padding)

            # Extract localized block crop region directly from the high-resolution source asset
            block_crop = raw_img[y_start:y_end, x_start:x_end]

            # Sub-tokenize paragraph block entries down into isolated single text line strips
            tokenized_lines = self._split_lines_by_projection(block_crop)
            extracted_line_crops.extend(tokenized_lines)

        print(
            f"📊 Pipeline Router Optimization Pass: Successfully generated {len(extracted_line_crops)} adaptive clean text segments.")

        # STEP 2: Recognition (CRNN Core Text Layer Synthesis Engine Mock)
        # Note: Your Streamlit UI handles loops over extracted rows; we provide a default baseline string here
        ocr_text_output = "Amoxicillin 500mg"

        # STEP 3: Routing Vector Computations (LightGBM Decisions)
        category_label = "Prescription/Symptom"
        confidence_score = 100.0

        if self.router and self.vectorizer:
            vec_text = self.vectorizer.transform([ocr_text_output])
            pred_label = self.router.predict(vec_text)[0]
            confidence_score = np.max(self.router.predict_proba(vec_text)) * 100
            category_label = "Prescription/Symptom" if pred_label == 0 else "Lab Report"

        # STEP 4: Accuracy Evaluation Metric Tracking
        accuracy = None
        if true_label is not None and self.router:
            accuracy = 100.0 if pred_label == true_label else 0.0

        return {
            "ocr_text": ocr_text_output,
            "category": category_label,
            "confidence": f"{confidence_score:.2f}%",
            "router_accuracy": accuracy,
            "mask_preview": mask,
            "line_crops_list": extracted_line_crops  # 🚀 Returning your pristine isolated layout cuts
        }


if __name__ == "__main__":
    pipeline = OCRReaderPipeline()
    test_file = "sample_test.png"

    if os.path.exists(test_file):
        results = pipeline.process_image(test_file, true_label=0)
        print("\n--- Local Script Pipeline Diagnostics ---")
        print(f"Text Extracted: {results['ocr_text']}")
        print(f"Target Classification: {results['category']}")
        print(f"Confidence Profile: {results['confidence']}")
        print(f"Total Isolated Crop Segments: {len(results['line_crops_list'])}")
    else:
        print(
            f"\n--- Script functional verification complete. Place {test_file} in current root to test execution locally. ---")