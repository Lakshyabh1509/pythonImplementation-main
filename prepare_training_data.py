"""
This script prepares training data for heading classification by:
1. Reading PDFs and extracting text lines with features
2. Reading corresponding JSON files with heading classifications
3. Matching the extracted lines with their classifications
4. Creating a labeled dataset for model training
"""

import os
import json
import pandas as pd
import numpy as np
from extract import extract_lines

def load_json_outline(json_path):
    """Load the JSON file with the heading outline."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of text -> heading level
    heading_map = {}
    for item in data.get('outline', []):
        text = item['text']
        level = item['level']
        page = item['page'] - 1  # JSON uses 1-indexed pages, our extraction uses 0-indexed
        
        # Create a unique key to handle duplicates text across pages
        key = f"{text}_{page}"
        heading_map[key] = level
    
    # Also map the title
    title = data.get('title', '')
    if title:
        heading_map[f"{title}_0"] = "TITLE"
        
    return heading_map

def create_training_dataset(pdf_dir, json_dir, output_file):
    """
    Create a training dataset from PDFs and their corresponding JSON outlines.
    
    Args:
        pdf_dir: Directory containing PDF files
        json_dir: Directory containing JSON outline files
        output_file: Path to save the training dataset
    """
    # Lists to store features and labels
    all_features = []
    all_labels = []
    label_map = {"TITLE": 0, "H1": 1, "H2": 2, "H3": 3, "BODY": 4}
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        json_path = os.path.join(json_dir, f"{base_name}.json")
        
        # Skip if JSON doesn't exist
        if not os.path.exists(json_path):
            print(f"Warning: No JSON file found for {pdf_file}, skipping...")
            continue
            
        print(f"Processing {pdf_file}...")
        
        # Load heading classifications from JSON
        heading_map = load_json_outline(json_path)
        
        # Extract lines from PDF
        lines = list(extract_lines(pdf_path))
        
        # Match lines with classifications and extract features
        for line in lines:
            text = line['text']
            page = line['page']
            
            # Create the same key as in the heading map
            key = f"{text}_{page}"
            
            # Determine the label (default to BODY)
            label = "BODY"
            if key in heading_map:
                label = heading_map[key]
                
            # Extract features
            features = {
                'font_size': line['font_size'],
                'x_position': line['x'],
                'is_bold': 1 if line['is_bold'] else 0,
                'text_length': len(text),
                'has_number_prefix': 1 if text.strip().split('.')[0].isdigit() else 0
            }
            
            # Add to our dataset
            all_features.append(features)
            all_labels.append(label_map.get(label, 4))  # Default to BODY (4) if not found
    
    # Create DataFrame
    feature_df = pd.DataFrame(all_features)
    
    # Save to output file
    output = {
        'features': feature_df.to_numpy(),
        'labels': np.array(all_labels),
        'feature_names': list(feature_df.columns),
        'label_map': {v: k for k, v in label_map.items()}
    }
    
    np.save(output_file, output)
    print(f"Saved training dataset to {output_file}")
    print(f"Dataset contains {len(all_labels)} samples")
    
    # Display class distribution
    classes, counts = np.unique(all_labels, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"Class {label_map.get(cls, 'Unknown')}: {count} samples")
    
    return output

if __name__ == "__main__":
    pdf_dir = './test_pdfs'
    json_dir = './outputs'
    output_file = './training_data.npy'
    
    dataset = create_training_dataset(pdf_dir, json_dir, output_file)
