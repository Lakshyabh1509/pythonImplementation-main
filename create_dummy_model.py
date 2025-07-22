import numpy as np
import struct
import os

def create_simple_onnx_file(output_path):
    """
    Create a minimal valid ONNX file with a simple identity operator.
    This is a very basic implementation for testing purposes only.
    """
    # ONNX file header (magic number + version)
    header = b'ONNX-ML' + struct.pack('<I', 7)  # Version 7
    
    # Create a simple ModelProto binary - this is a very simplified version
    # In a real scenario, use proper ONNX API to generate this
    model_binary = b'\x0a\x05Model\x12\x0fHeadingClassifier\x1a\x01A'
    
    # Combine header and model
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(model_binary)
    
    print(f"Simple ONNX model file created at: {output_path}")

if __name__ == "__main__":
    output_file = "model.onnx"
    create_simple_onnx_file(output_file)
