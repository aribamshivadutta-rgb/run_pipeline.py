import numpy as np
import joblib


class MedicalLinker:
    def __init__(self, model_path, symptom_list):
        # Load your PREVIOUS project's brain
        self.brain = joblib.load(model_path)
        # The list of symptom names your previous model was trained on
        self.feature_names = symptom_list

    def translate_ocr_to_input(self, crnn_text):
        """
        Translates 'fever' from CRNN into [0, 1, 0...] for your old model.
        """
        # Create a blank vector of zeros the size of your symptom list
        input_vector = np.zeros(len(self.feature_names))

        # Clean the OCR text
        detected_word = crnn_text.lower().strip()

        # If the word from the CRNN matches one of your symptoms, set it to 1
        if detected_word in self.feature_names:
            index = self.feature_names.index(detected_word)
            input_vector[index] = 1

        return input_vector.reshape(1, -1)

    def predict(self, crnn_text):
        vector = self.translate_ocr_to_input(crnn_text)
        prediction = self.brain.predict(vector)
        return prediction