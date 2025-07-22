try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz  # Alternative import name
    except ImportError:
        from PyMuPDF import fitz  # Another alternative
import re

import time
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

def measure_performance(func):
    """Decorator to measure and log function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_performance
def extract_lines(pdf_path):
    """Extract text lines with font and positioning information from a PDF."""
    doc = fitz.open(pdf_path)
    try:
        # Pre-compile regex pattern for better performance
        space_pattern = re.compile(r'\s+')
        
        # Process pages in batches for better memory management
        batch_size = 5
        pages = list(doc)
        
        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i:i + batch_size]
            
            for page_num, page in enumerate(batch_pages, start=i):
                # Get all blocks at once
                blocks = page.get_text("dict")["blocks"]
                
                # Filter blocks with lines first
                lines_blocks = [b for b in blocks if "lines" in b]
                
                for block in lines_blocks:
                    for line in block["lines"]:
                        if not ("spans" in line and line["spans"]):
                            continue
                            
                        spans = line["spans"]
                        first_span = spans[0]
                        
                        # More efficient text combination
                        text = " ".join(span["text"] for span in spans if span["text"].strip())
                        if not text:
                            continue
                            
                        # Clean text using pre-compiled pattern
                        text = space_pattern.sub(' ', text).strip()
                        
                        yield {
                            "text": text,
                            "page": page_num,
                            "font_size": first_span["size"],
                            "is_bold": any(word in first_span["font"].lower() 
                                         for word in ("bold", "black")),
                            "x": line["bbox"][0],
                            "y": line["bbox"][1],
                        }
    finally:
        doc.close()

@measure_performance
def process_pdfs_parallel(pdf_paths):
    """Process multiple PDFs in parallel"""
    max_workers = min(multiprocessing.cpu_count(), len(pdf_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_lines, pdf_paths))
    return results
