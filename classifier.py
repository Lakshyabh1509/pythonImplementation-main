import numpy as np
import os.path
import random
import pickle

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available.")

class HeadingClassifier:
    def __init__(self, model_path=None, pickle_model_path=None):
        """
        Initialize the classifier with either an ONNX model or a pickled sklearn model.
        Falls back to rule-based classification if no models are available.
        
        Args:
            model_path: Path to the ONNX model
            pickle_model_path: Path to the pickled sklearn model
        """
        self.model_path = model_path
        self.pickle_model_path = pickle_model_path or './model_trained.pkl'
        
        # Check if models exist
        self.onnx_model_exists = model_path and os.path.exists(model_path)
        self.pickle_model_exists = os.path.exists(self.pickle_model_path)
        
        # Try to load the ONNX model if available
        self.use_onnx = ONNX_AVAILABLE and self.onnx_model_exists
        self.use_sklearn = False
        
        if self.use_onnx:
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                print(f"Initialized classifier with ONNX model: {model_path}")
            except Exception as e:
                self.use_onnx = False
                print(f"Failed to load ONNX model: {e}")
                print("Trying sklearn model...")
                
        # Try to load the sklearn model if ONNX is not available or failed
        if not self.use_onnx and self.pickle_model_exists:
            try:
                with open(self.pickle_model_path, 'rb') as f:
                    self.sklearn_model_data = pickle.load(f)
                self.use_sklearn = True
                print(f"Initialized classifier with sklearn model: {self.pickle_model_path}")
            except Exception as e:
                print(f"Failed to load sklearn model: {e}")
                print("Falling back to rule-based classification.")
        
        # Print final classification method
        if self.use_onnx:
            print("Using ONNX model for classification")
        elif self.use_sklearn:
            print("Using sklearn model for classification")
        else:
            print("Using rule-based classification")

    def serialize(self, line):
        """Format text line with features for model input"""
        return f"[SIZE={int(line['font_size'])}] [X={int(line['x'])}] [BOLD={line['is_bold']}] {line['text']}"

    def extract_features(self, lines):
        """Extract features from serialized text lines"""
        features = []
        
        for i, text in enumerate(lines):
            # Extract features from the serialized text
            is_bold = "BOLD=True" in text
            is_bold_value = 1 if is_bold else 0
            
            try:
                # Extract font size
                size_part = text.split("SIZE=")[1].split("]")[0]
                font_size = float(size_part)
            except (IndexError, ValueError):
                font_size = 11  # Default font size
            
            # Extract indentation
            try:
                x_pos = text.split("X=")[1].split("]")[0]
                indent = float(x_pos)
            except (IndexError, ValueError):
                indent = 0
                
            # Extract text (after the features)
            try:
                actual_text = text.split("] ")[3]
            except (IndexError, ValueError):
                actual_text = ""
                
            # Compute additional features
            text_length = len(actual_text)
            has_number_prefix = 1 if actual_text.strip().split('.')[0].isdigit() else 0
            
            # Collect features in the same order as training
            features.append([
                font_size,
                indent,
                is_bold_value,
                text_length,
                has_number_prefix
            ])
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)

    def predict(self, lines):
        """
        Predict heading classifications for text lines
        
        Returns:
            - predictions: array of class indices (0: TITLE, 1: H1, 2: H2, 3: H3, 4: BODY)
            - confidence: array of confidence scores
        """
        # Extract features from the serialized text lines
        features_array = self.extract_features(lines)
        
        # Use ONNX model if available
        if self.use_onnx:
            try:
                # Run inference
                ort_inputs = {self.input_name: features_array}
                ort_outputs = self.session.run(None, ort_inputs)
                
                # Get predictions and confidences
                if len(ort_outputs) == 1:
                    # If model outputs class scores
                    logits = ort_outputs[0]
                    predictions = np.argmax(logits, axis=1)
                    confidences = np.max(logits, axis=1)
                else:
                    # If model directly outputs predictions
                    predictions = ort_outputs[0]
                    confidences = np.ones_like(predictions) * 0.9  # Default confidence
                
                return predictions, confidences
            except Exception as e:
                print(f"ONNX inference failed: {e}")
                print("Trying sklearn model...")
                self.use_onnx = False
        
        # Use sklearn model if available
        if self.use_sklearn:
            try:
                # Get the model pipeline
                pipeline = self.sklearn_model_data['model']
                
                # Run inference
                probabilities = pipeline.predict_proba(features_array)
                predictions = pipeline.predict(features_array)
                confidences = np.max(probabilities, axis=1)
                
                return predictions, confidences
            except Exception as e:
                print(f"sklearn inference failed: {e}")
                print("Falling back to rule-based classification.")
        
        # Fallback to rule-based classification
        predictions = []
        confidences = []
        
        for i, (font_size, indent, is_bold, text_length, _) in enumerate(features_array):
            # Simplified rule-based classification
            # First line is usually title
            if i == 0:
                pred_class = 0  # TITLE
                confidence = 0.9
            elif is_bold and font_size > 14:
                pred_class = 0  # TITLE
                confidence = 0.85
            elif is_bold and font_size > 12:
                pred_class = 1  # H1
                confidence = 0.8
            elif is_bold or font_size > 11:
                pred_class = 2  # H2
                confidence = 0.7
            elif indent > 20:
                pred_class = 3  # H3
                confidence = 0.6
            else:
                pred_class = 4  # BODY
                confidence = 0.9
            
            # Add some randomness for variety in the mock predictions
            if random.random() < 0.2:
                pred_class = random.randint(1, 4)
                confidence = random.uniform(0.5, 0.8)
            
            predictions.append(pred_class)
            confidences.append(confidence)
            
        return np.array(predictions), np.array(confidences)