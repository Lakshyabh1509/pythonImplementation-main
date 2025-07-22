"""
This script trains a model for heading classification using the prepared training data.
It creates and saves the model in ONNX format for production use.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import onnxruntime as ort
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_model(data_file):
    """
    Train a model using the prepared training data.
    
    Args:
        data_file: Path to the training data file (.npy)
    
    Returns:
        The trained model and a summary of its performance
    """
    # Load training data
    print(f"Loading training data from {data_file}...")
    data = np.load(data_file, allow_pickle=True).item()
    
    X = data['features']
    y = data['labels']
    feature_names = data['feature_names']
    label_map = data['label_map']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train model pipeline
    print("Training model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    class_names = [label_map[i] for i in sorted(label_map.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return pipeline, feature_names

def save_onnx_model(model, feature_names, output_path):
    """
    Convert and save the trained model to ONNX format.
    
    Args:
        model: The trained sklearn model
        feature_names: Names of the features
        output_path: Path to save the ONNX model
    """
    print(f"Converting model to ONNX format...")
    
    # Define input type (assuming float32 for all features)
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Save the model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify the model
    try:
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        print(f"Model verification successful. Input name: {input_name}")
    except Exception as e:
        print(f"Model verification failed: {e}")

if __name__ == "__main__":
    data_file = './training_data.npy'
    model_output = './model_trained.onnx'
    
    # Check if training data exists
    if not os.path.exists(data_file):
        print(f"Error: Training data file {data_file} not found.")
        print("Please run prepare_training_data.py first.")
        exit(1)
    
    # Train the model
    model, feature_names = train_model(data_file)
    
    # Save the model
    save_onnx_model(model, feature_names, model_output)
