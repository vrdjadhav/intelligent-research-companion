import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    """
    Takes a Streamlit uploaded file object, reads it using PyMuPDF,
    and returns the combined raw text from all pages with explicit debugging.
    """
    text = ""
    try:
        # Reset file pointer to the beginning to make sure we read from byte 0
        uploaded_file.seek(0)
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        print(f"\n[DEBUG] Successfully read {len(file_bytes)} bytes from uploaded file.")
        
        # Open PDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        print(f"[DEBUG] PyMuPDF opened document. Total pages detected: {len(doc)}")
        
        if len(doc) == 0:
            print("[DEBUG] WARNING: This PDF has 0 pages.")
            return None
            
        # Extract text page by page
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            print(f"[DEBUG] Page {page_num}: Extracted {len(page_text)} characters.")
            text += page_text + "\n"
            
        doc.close()
        
        final_text = text.strip()
        print(f"[DEBUG] Extraction complete. Total final character length: {len(final_text)}\n")
        
        return final_text
        
    except Exception as e:
        print(f"\n[DEBUG] CRITICAL ERROR during PDF extraction: {e}\n")
        return None