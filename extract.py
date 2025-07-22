try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz  # Alternative import name
    except ImportError:
        from PyMuPDF import fitz  # Another alternative
import re

def extract_lines(pdf_path):
    """Extract text lines with font and positioning information from a PDF."""
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        # Extract text with details about fonts and positions
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line and len(line["spans"]) > 0:
                        # Combine spans into a single line
                        text = " ".join([span["text"] for span in line["spans"] if span["text"].strip()])
                        if not text.strip():
                            continue
                            
                        # Get font information from the first span (most significant part of the line)
                        first_span = line["spans"][0]
                        font_size = first_span["size"]
                        is_bold = "bold" in first_span["font"].lower() or "black" in first_span["font"].lower()
                        
                        # Clean up text
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        yield {
                            "text": text,
                            "page": page_num,
                            "font_size": font_size,
                            "is_bold": is_bold,
                            "x": line["bbox"][0],  # Left position (indentation)
                            "y": line["bbox"][1],  # Top position
                        }
    doc.close()
