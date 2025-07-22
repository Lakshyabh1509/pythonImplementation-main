import os, concurrent.futures, argparse
from extract import extract_lines
from classifier import HeadingClassifier
from outline import build_outline, write_json

def process_pdf(pdf_path, model_path, pickle_model_path, output_dir, label_map):
    try:
        print(f"Processing: {pdf_path}")
        
        # First check if the PDF exists
        if not os.path.exists(pdf_path):
            print(f"ERROR: PDF file not found: {pdf_path}")
            return
        
        # Check if output directory exists
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract lines
        try:
            lines = list(extract_lines(pdf_path))
            print(f"  Extracted {len(lines)} lines from {pdf_path}")
        except Exception as e:
            print(f"  ERROR extracting lines from {pdf_path}: {str(e)}")
            return
        
        # Initialize classifier
        clf = HeadingClassifier(model_path, pickle_model_path)
        
        # Classify lines
        try:
            serialized = [clf.serialize(line) for line in lines]
            preds, confs = clf.predict(serialized)
            print(f"  Classified {len(preds)} lines")
        except Exception as e:
            print(f"  ERROR classifying lines: {str(e)}")
            return
        
        # Get title and build outline
        title = next((line["text"] for line, pred in zip(lines, preds) if pred == 0), lines[0]["text"])
        outline = build_outline(lines, preds, confs, label_map)
        print(f"  Generated outline with {len(outline)} headings")
        
        # Save JSON in output directory
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")
        
        try:
            write_json(output_file, title, outline)
            print(f"  Output saved to: {output_file}")
        except Exception as e:
            print(f"  ERROR saving output to {output_file}: {str(e)}")
    except Exception as e:
        print(f"ERROR processing {pdf_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Extract PDF outlines with heading classification')
    parser.add_argument('--input', '-i', required=False, default='../adobe_clone_github/Challenge_1a/sample_dataset/pdfs', help='Input directory with PDFs')
    parser.add_argument('--output', '-o', required=False, default='./outputs', help='Output directory for JSON files')
    parser.add_argument('--model', '-m', required=False, default='./model.onnx', help='Path to ONNX model file')
    parser.add_argument('--sklearn-model', '-s', required=False, default='./model_trained.pkl', help='Path to pickled sklearn model file')
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = []
    if os.path.isdir(args.input):
        pdf_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(".pdf")]
    elif os.path.isfile(args.input) and args.input.lower().endswith(".pdf"):
        pdf_files = [args.input]
    else:
        print(f"Error: Invalid input path: {args.input}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Define label mapping for the model
    label_map = {0: "TITLE", 1: "H1", 2: "H2", 3: "H3", 4: "BODY"}
    
    # Process PDFs sequentially for debugging
    print("Processing PDFs sequentially for debugging:")
    for pdf_file in pdf_files:
        process_pdf(pdf_file, args.model, args.sklearn_model, args.output, label_map)
    
    # Check if output files were created
    output_files = [f for f in os.listdir(args.output) if f.endswith('.json')]
    print(f"Output files found: {len(output_files)}")
    if output_files:
        print("Output files:")
        for output_file in output_files:
            print(f"  - {output_file}")
    else:
        print("No output files were created!")
        print(f"Output directory: {os.path.abspath(args.output)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()