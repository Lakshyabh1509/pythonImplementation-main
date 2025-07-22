# PDF Heading Classification System

A production-ready system for extracting and classifying headings from PDF documents using machine learning.

## Production Features

- **High Performance PDF Processing**
  - Parallel processing of multiple PDFs
  - Batch processing for memory efficiency
  - Optimized text extraction and cleaning

- **Machine Learning Integration**
  - Supports both ONNX and sklearn models
  - Automated model training pipeline
  - Feature engineering for heading detection

- **Production Ready**
  - Comprehensive error handling
  - Performance monitoring
  - Resource management
  - Unit testing
  - Configuration management

## Features
- Extracts text lines with font and position info from PDFs
- Matches lines with heading classifications from JSON
- Prepares training data for ML models
- Includes scripts for training and using ONNX models

## File Overview
- `extract.py`: Extracts text lines and font info from PDFs
- `prepare_training_data.py`: Prepares labeled training data from PDFs and JSON
- `train_model.py` / `train_model_sklearn.py`: Train models using the prepared data
- `classifier.py`: Classifies new PDF lines using trained models
- `model.onnx`: Example trained model
- `requirements.txt`: Python dependencies
- `test_pdfs/`: Sample PDFs for testing
- `outputs/`: JSON files with heading classifications

## Setup & Installation
1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For ONNX model usage, install onnxruntime:
   ```bash
   pip install onnxruntime
   ```

## Usage
- Prepare training data:
  ```bash
  python prepare_training_data.py
  ```
- Train a model:
  ```bash
  python train_model.py
  ```
- Classify new PDFs:
  ```bash
  python classifier.py
  ```

## Notes
- Place your PDFs in `test_pdfs/` and corresponding JSONs in `outputs/`.
- The scripts are modular and can be extended for other ML frameworks.

## License
MIT
