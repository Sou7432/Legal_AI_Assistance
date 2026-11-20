"""
=============================================================
PART 1 — PDF Extraction & Cleaning
=============================================================
File: src/part1_pdf_extraction/extract_pdfs.py

This script:
  1. Reads all PDFs from data/raw_pdfs/
  2. Extracts text using PyMuPDF (fast) with pdfplumber fallback
  3. Detects scanned pages and applies Tesseract OCR
  4. Cleans headers, footers, and page numbers
  5. Splits text into legal sections (preserving section numbers)
  6. Saves cleaned sections as JSON files in data/cleaned_sections/
  7. Verifies the extraction quality

HOW TO RUN:
    python src/part1_pdf_extraction/extract_pdfs.py

DEPENDENCIES:
    pip install PyMuPDF pdfplumber pytesseract pdf2image Pillow
    Install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki
    Set TESSERACT_PATH in .env  (e.g. C:/Program Files/Tesseract-OCR/tesseract.exe)
=============================================================
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import pymupdf as fitz         # PyMuPDF (v2.x renamed fitz → pymupdf)
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RAW_PDF_DIR       = Path("data/raw_pdfs")
EXTRACTED_TEXT_DIR = Path("data/extracted_text")
CLEANED_SECTION_DIR = Path("data/cleaned_sections")

# Tesseract path (Windows) — set in .env or change here
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Directories to create
for d in [EXTRACTED_TEXT_DIR, CLEANED_SECTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 & 2: Extract text from PDF pages
# ─────────────────────────────────────────────
def extract_with_pymupdf(pdf_path: Path) -> list[dict]:
    """
    Fast extraction using PyMuPDF.
    Returns a list of dicts: {page_number, text, is_scanned}.
    A page is considered 'scanned' if it has very little selectable text
    but contains images — meaning we need OCR.
    """
    pages = []
    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")           # extract selectable text
        image_count = len(page.get_images())   # count embedded images

        # Heuristic: if text is very short but images exist → likely scanned
        is_scanned = (len(text.strip()) < 50 and image_count > 0)

        pages.append({
            "page_number": page_num,
            "text": text,
            "is_scanned": is_scanned
        })

    doc.close()
    return pages


# ─────────────────────────────────────────────
# STEP 3: OCR for scanned pages
# ─────────────────────────────────────────────
def ocr_page(pdf_path: Path, page_number: int) -> str:
    """
    Convert a single PDF page to an image and run Tesseract OCR.
    page_number is 1-indexed.
    """
    # pdf2image uses 0-indexed first_page / last_page
    images = convert_from_path(
        str(pdf_path),
        dpi=300,
        first_page=page_number,
        last_page=page_number
    )
    if not images:
        return ""

    img = images[0]
    # lang='eng' works for English legal text; add '+hin' for Hindi if needed
    text = pytesseract.image_to_string(img, lang="eng")
    return text


def extract_all_pages(pdf_path: Path) -> list[dict]:
    """
    Full extraction pipeline per PDF:
    - Use PyMuPDF for digital text
    - Use OCR for scanned pages
    """
    log.info(f"Extracting: {pdf_path.name}")
    pages = extract_with_pymupdf(pdf_path)

    for page in pages:
        if page["is_scanned"]:
            log.info(f"  → Page {page['page_number']} is scanned. Running OCR…")
            page["text"] = ocr_page(pdf_path, page["page_number"])
            page["ocr_applied"] = True
        else:
            page["ocr_applied"] = False

    return pages


# ─────────────────────────────────────────────
# STEP 4: Text cleaning
# ─────────────────────────────────────────────
# Common patterns found in Indian legal PDFs
HEADER_FOOTER_PATTERNS = [
    r"^Page\s+\d+\s+of\s+\d+",       # "Page 3 of 45"
    r"^\d+\s*$",                       # standalone page numbers
    r"^Ministry of Law.*$",            # ministry headers
    r"^Government of India.*$",
    r"^THE GAZETTE OF INDIA.*$",
    r"^Registered No\..*$",
    r"^©.*$",                          # copyright lines
    r"^www\..*$",                      # website footers
    r"^\s*[-–—]+\s*$",                 # divider lines
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                     for p in HEADER_FOOTER_PATTERNS]


def clean_page_text(text: str) -> str:
    """
    Remove headers, footers, page numbers, and formatting artifacts.
    Preserve section numbers, sub-sections, article references, and legal text.
    """
    # Remove header/footer lines
    for pattern in COMPILED_PATTERNS:
        text = pattern.sub("", text)

    # Normalize excessive whitespace while keeping paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)         # multiple spaces → one space
    text = re.sub(r"\n[ \t]+", "\n", text)         # leading whitespace on lines

    # Fix common OCR errors in legal text
    text = text.replace("§", "Section ")           # § → "Section"
    text = text.replace("Sec.", "Section ")
    text = re.sub(r"\bS\.\s*(\d)", r"Section \1", text)  # "S. 34" → "Section 34"

    return text.strip()


# ─────────────────────────────────────────────
# STEP 5 & 6: Section splitting
# ─────────────────────────────────────────────
# Patterns that mark the START of a legal section / article
SECTION_START_PATTERNS = [
    # "Section 34." or "34." or "Section 34A"
    r"^(?:Section|Sec\.?|SECTION)\s+(\d+[A-Z]?(?:\s*[-–]\s*\d+[A-Z]?)?)[.\s—]",
    r"^(\d+[A-Z]?)\.\s+[A-Z]",         # "34. Definitions"
    r"^Article\s+(\d+[A-Z]?)[.\s—]",   # "Article 21"
    r"^CHAPTER\s+[IVXLCDM\d]+",        # Roman numeral chapters
    r"^Part\s+[IVXLCDM\d]+",           # "Part II"
    r"^Schedule\s+[IVXLCDM\d]+",       # Schedules
]

SECTION_REGEX = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                 for p in SECTION_START_PATTERNS]


def split_into_sections(full_text: str, law_name: str) -> list[dict]:
    """
    Split the full cleaned document text into individual legal sections.
    
    Each section dict contains:
    - law_name: name of the parent Act
    - section_id: e.g. "Section 34", "Article 21"
    - section_title: title / heading if present
    - section_text: full text of the section
    """
    # Find all section start positions
    breakpoints = []
    for regex in SECTION_REGEX:
        for match in regex.finditer(full_text):
            breakpoints.append((match.start(), match.group(0).strip()))

    # Sort by position and deduplicate nearby matches
    breakpoints.sort(key=lambda x: x[0])
    deduped = []
    for pos, label in breakpoints:
        if not deduped or pos - deduped[-1][0] > 10:
            deduped.append((pos, label))

    sections = []
    for i, (start_pos, label) in enumerate(deduped):
        end_pos = deduped[i + 1][0] if i + 1 < len(deduped) else len(full_text)
        raw_section_text = full_text[start_pos:end_pos].strip()

        # Extract section ID from label
        section_id = label.strip().rstrip(".").strip()

        # Try to extract first line as title
        lines = raw_section_text.split("\n")
        section_title = lines[0].strip() if lines else section_id
        section_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw_section_text

        sections.append({
            "law_name": law_name,
            "section_id": section_id,
            "section_title": section_title,
            "section_text": raw_section_text,  # keep full text for RAG
            "char_count": len(raw_section_text)
        })

    # If no sections found, treat entire doc as one section
    if not sections:
        log.warning(f"No section boundaries found in '{law_name}'. Storing as one block.")
        sections.append({
            "law_name": law_name,
            "section_id": "FULL_DOCUMENT",
            "section_title": law_name,
            "section_text": full_text,
            "char_count": len(full_text)
        })

    return sections


# ─────────────────────────────────────────────
# STEP 7: Save to JSON
# ─────────────────────────────────────────────
def save_sections_to_json(sections: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    log.info(f"  ✓ Saved {len(sections)} sections → {output_path}")


def save_raw_text(pages: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n--- PAGE {page['page_number']} ---\n")
            f.write(page["text"])
    log.info(f"  ✓ Raw text saved → {output_path}")


# ─────────────────────────────────────────────
# STEP 8: Verification
# ─────────────────────────────────────────────
def verify_extraction(sections: list[dict]) -> dict:
    """
    Quality checks on the extracted sections:
    - Minimum section count
    - No empty sections
    - Sections contain recognizable legal keywords
    """
    LEGAL_KEYWORDS = [
        "shall", "provided", "notwithstanding", "section", "sub-section",
        "clause", "article", "act", "offence", "penalty", "court", "person"
    ]

    total = len(sections)
    empty = sum(1 for s in sections if s["char_count"] < 30)
    with_keywords = sum(
        1 for s in sections
        if any(kw in s["section_text"].lower() for kw in LEGAL_KEYWORDS)
    )

    report = {
        "total_sections": total,
        "empty_sections": empty,
        "sections_with_legal_keywords": with_keywords,
        "keyword_coverage_pct": round(with_keywords / max(total, 1) * 100, 1)
    }
    log.info(f"  Verification: {report}")
    return report


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def process_single_pdf(pdf_path: Path) -> dict:
    """Full pipeline for one PDF file."""
    law_name = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    # 1–3: Extract (with OCR for scanned pages)
    pages = extract_all_pages(pdf_path)

    # Save raw extracted text for reference
    raw_text_path = EXTRACTED_TEXT_DIR / f"{pdf_path.stem}_raw.txt"
    save_raw_text(pages, raw_text_path)

    # 4: Clean each page
    full_text = "\n\n".join(clean_page_text(p["text"]) for p in pages)

    # 5–6: Split into sections
    sections = split_into_sections(full_text, law_name)

    # 7: Save cleaned sections
    output_path = CLEANED_SECTION_DIR / f"{pdf_path.stem}_sections.json"
    save_sections_to_json(sections, output_path)

    # 8: Verify
    verification = verify_extraction(sections)

    return {
        "pdf": pdf_path.name,
        "law_name": law_name,
        "pages_processed": len(pages),
        "ocr_pages": sum(1 for p in pages if p.get("ocr_applied")),
        "sections_found": len(sections),
        "verification": verification,
        "output_file": str(output_path)
    }


def main():
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDFs found in {RAW_PDF_DIR}. Please add your legal PDFs there.")
        return

    log.info(f"Found {len(pdf_files)} PDF(s) to process.")
    summary = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            result = process_single_pdf(pdf_path)
            summary.append(result)
        except Exception as e:
            log.error(f"Failed to process {pdf_path.name}: {e}")

    # Save overall summary
    summary_path = EXTRACTED_TEXT_DIR / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*60}")
    log.info(f"EXTRACTION COMPLETE")
    log.info(f"  PDFs processed : {len(summary)}")
    log.info(f"  Summary saved  : {summary_path}")
    log.info(f"  Sections stored: {CLEANED_SECTION_DIR}")


if __name__ == "__main__":
    main()
