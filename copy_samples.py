import os
import shutil
import sys

def copy_sample_pdfs():
    """
    Copy sample PDFs from the Challenge_1a dataset to our test directory
    """
    src_dir = "../adobe_clone_github/Challenge_1a/sample_dataset/pdfs"
    dest_dir = "./test_pdfs"
    
    # Make sure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist!")
        print("Current working directory:", os.getcwd())
        return False
        
    # List all PDF files
    pdf_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {src_dir}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files in {src_dir}")
    
    # Copy each PDF file
    for pdf_file in pdf_files:
        src_file = os.path.join(src_dir, pdf_file)
        dest_file = os.path.join(dest_dir, pdf_file)
        
        try:
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {pdf_file}")
        except Exception as e:
            print(f"Error copying {pdf_file}: {e}")
    
    print(f"Successfully copied {len(pdf_files)} PDF files to {dest_dir}")
    return True

if __name__ == "__main__":
    success = copy_sample_pdfs()
    sys.exit(0 if success else 1)
