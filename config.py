"""Configuration settings for the PDF processing pipeline"""

import os

# PDF Processing Settings
BATCH_SIZE = 5
MAX_WORKERS = None  # None = use CPU count

# Model Settings
MODEL_PATH = "model.onnx"
PICKLE_MODEL_PATH = "model_trained.pkl"

# File Paths
TEST_PDFS_DIR = "test_pdfs"
OUTPUT_DIR = "outputs"
TRAINING_DATA_FILE = "training_data.npy"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = "pdf_processor.log"

# Feature Settings
FONT_SIZE_THRESHOLD = 12
BOLD_KEYWORDS = ["bold", "black"]

# Create directories if they don't exist
for directory in [TEST_PDFS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
