import pdfplumber
import re
import pandas as pd

def extract_shot_descriptions_from_pdf(pdf_path, start_page=10, end_page=100):
    """
    Extract shot IDs, titles, and descriptions from a PDF file.

    Parameters:
    - pdf_path (str): Path to the PDF file
    - start_page (int): Page number to start reading from (default skips preface)
    - end_page (int): Page number to stop reading

    Returns:
    - DataFrame: Structured data containing id, shot_title, and description
    """
    full_text = ""

    # Step 1: Concatenate text from the specified page range
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[start_page:end_page]:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Step 2: Use regex to extract segments like "1.1 TITLE" + description block
    pattern = r"(\d+\.\d+)\s+([A-Z][A-Z\s\-]+)\n(.*?)(?=\n\d+\.\d+\s+[A-Z])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)

    # Step 3: Convert to structured DataFrame
    extracted_data = []
    for shot_id, title, desc in matches:
        extracted_data.append({
            "id": shot_id.strip(),
            "shot_title": title.title().strip(),
            "description": desc.strip().replace("\n", " ")
        })

    return pd.DataFrame(extracted_data)
