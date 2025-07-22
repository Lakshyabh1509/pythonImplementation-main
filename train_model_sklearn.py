"""
This script trains a model for heading classification using the prepared training data
and saves it as a pickle file for easy loading.
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_file):
    """
    Train a model using the prepared training data.
    
    Args:
        data_file: Path to the training data file (.npy)
    
    Returns:
        The trained model and a summary of its performance
    """
    try:
        # Load training data
        print(f"Loading training data from {data_file}...")
        data = np.load(data_file, allow_pickle=True).item()
        
        X = data['features']
        y = data['labels']
        feature_names = data['feature_names']
        label_map = data['label_map']
        
        # Print label distribution in the data
        print("Label distribution in full dataset:")
        unique_values, counts = np.unique(y, return_counts=True)
        for value, count in zip(unique_values, counts):
            class_name = label_map.get(value, f"Unknown-{value}")
            print(f"  {class_name} (class {value}): {count} samples ({count/len(y)*100:.1f}%)")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create and train model pipeline
        print("Training model...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training preparation: {e}")
        return None, None, None
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Find unique classes in all the data
    unique_train = set(np.unique(y_train))
    unique_test = set(np.unique(y_test))
    unique_pred = set(np.unique(y_pred))
    
    all_classes = sorted(unique_train.union(unique_test).union(unique_pred))
    print(f"All unique classes: {all_classes}")
    print(f"Classes in training: {sorted(unique_train)}")
    print(f"Classes in test: {sorted(unique_test)}")
    print(f"Classes in predictions: {sorted(unique_pred)}")
    
    # Print detailed classification report without target_names
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print class distribution
    print("\nClass distribution in test set:")
    for label in sorted(unique_test):
        count = np.sum(y_test == label)
        class_name = label_map.get(label, f"Unknown-{label}")
        print(f"  {class_name} (class {label}): {count} samples")
    
    return pipeline, feature_names, label_map

def save_model(model, feature_names, label_map, output_path):
    """
    Save the trained model as a pickle file.
    
    Args:
        model: The trained sklearn model
        feature_names: Names of the features
        label_map: Mapping of class indices to names
        output_path: Path to save the pickle model
    """
    print(f"Saving model to {output_path}...")
    
    # Create a dictionary with all model components
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'label_map': label_map
    }
    
    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    data_file = './training_data.npy'
    model_output = './model_trained.pkl'
    
    # Check if training data exists
    if not os.path.exists(data_file):
        print(f"Error: Training data file {data_file} not found.")
        print("Please run prepare_training_data.py first.")
        exit(1)
    
    try:
        # Train the model
        model, feature_names, label_map = train_model(data_file)
        
        if model is None:
            print("Model training failed. Cannot save the model.")
            exit(1)
        
        # Save the model
        save_model(model, feature_names, label_map, model_output)
        
        print("Script completed successfully!")
    except Exception as e:
        print(f"An error occurred during script execution: {e}")
        exit(1)
