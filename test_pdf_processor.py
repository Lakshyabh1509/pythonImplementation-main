"""Unit tests for the PDF processing pipeline"""

import unittest
import os
import tempfile
from extract import extract_lines, process_pdfs_parallel
from classifier import HeadingClassifier
import numpy as np

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_pdf_path = os.path.join("test_pdfs", "file01.pdf")
        self.model_path = "model.onnx"
    
    def test_extract_lines(self):
        """Test PDF text extraction"""
        if os.path.exists(self.test_pdf_path):
            lines = list(extract_lines(self.test_pdf_path))
            self.assertGreater(len(lines), 0)
            self.assertTrue(all(isinstance(line, dict) for line in lines))
            
            # Check required keys
            required_keys = {"text", "page", "font_size", "is_bold", "x", "y"}
            self.assertTrue(all(required_keys.issubset(line.keys()) for line in lines))
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        classifier = HeadingClassifier()
        self.assertIsNotNone(classifier)
    
    def test_parallel_processing(self):
        """Test parallel PDF processing"""
        if os.path.exists("test_pdfs"):
            pdf_files = [f for f in os.listdir("test_pdfs") if f.endswith('.pdf')]
            if pdf_files:
                pdf_paths = [os.path.join("test_pdfs", f) for f in pdf_files]
                results = process_pdfs_parallel(pdf_paths)
                self.assertIsNotNone(results)

if __name__ == '__main__':
    unittest.main()
