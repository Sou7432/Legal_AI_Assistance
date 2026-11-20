"""
Quick validation script -- run this after installing requirements.txt.
It checks every package and prints a clear PASS/FAIL per module.

Usage:
    python check_setup.py
"""

import sys
import os

REQUIRED = [
    ("fitz",                 "PyMuPDF",             "pip install PyMuPDF"),
    ("pdfplumber",           "pdfplumber",           "pip install pdfplumber"),
    ("pytesseract",          "pytesseract",          "pip install pytesseract"),
    ("pdf2image",            "pdf2image",            "pip install pdf2image"),
    ("PIL",                  "Pillow",               "pip install Pillow"),
    ("tqdm",                 "tqdm",                 "pip install tqdm"),
    ("dotenv",               "python-dotenv",        "pip install python-dotenv"),
    ("requests",             "requests",             "pip install requests"),
    ("regex",                "regex",                "pip install regex"),
    ("jsonlines",            "jsonlines",            "pip install jsonlines"),
    ("tabulate",             "tabulate",             "pip install tabulate"),
    ("pandas",               "pandas",               "pip install pandas"),
    ("numpy",                "numpy",                "pip install numpy"),
    ("chromadb",             "chromadb",             "pip install chromadb"),
    ("sentence_transformers","sentence-transformers","pip install sentence-transformers"),
    ("transformers",         "transformers",         "pip install transformers"),
    ("datasets",             "datasets",             "pip install datasets"),
    ("peft",                 "peft",                 "pip install peft"),
    ("accelerate",           "accelerate",           "pip install accelerate"),
    ("trl",                  "trl",                  "pip install trl"),
    ("evaluate",             "evaluate",             "pip install evaluate"),
    ("rouge_score",          "rouge-score",          "pip install rouge-score"),
    ("sklearn",              "scikit-learn",         "pip install scikit-learn"),
    ("mysql.connector",      "mysql-connector-python","pip install mysql-connector-python"),
    ("ollama",               "ollama",               "pip install ollama"),
]

OPTIONAL = [
    ("bitsandbytes", "bitsandbytes (QLoRA 4-bit, needs NVIDIA GPU)", "pip install bitsandbytes"),
    ("bert_score",   "bert-score",                                   "pip install bert-score"),
]


def check(module, label, install_cmd, required=True):
    try:
        __import__(module)
        print(f"  [PASS]  {label}")
        return True
    except ImportError:
        tag = "[FAIL]" if required else "[SKIP]"
        note = f"         Fix: {install_cmd}" if required else "         (optional - only for GPU training)"
        print(f"  {tag}  {label}")
        print(note)
        return False


print()
print("=" * 55)
print("AI Legal Assistant -- Setup Check")
print(f"Python {sys.version}")
print("=" * 55)
print()

print("Required packages:")
failures = sum(1 for m, l, c in REQUIRED if not check(m, l, c, required=True))

print()
print("Optional packages:")
for m, l, c in OPTIONAL:
    check(m, l, c, required=False)

print()
print("=" * 55)
if failures == 0:
    print("ALL REQUIRED PACKAGES INSTALLED. Project is ready to run!")
else:
    print(f"{failures} required package(s) missing.")
    print("Run:  pip install -r requirements.txt")
print("=" * 55)
print()

# Check Ollama is running
print("Checking Ollama service...")
try:
    import requests
    r = requests.get("http://localhost:11434/api/version", timeout=3)
    if r.status_code == 200:
        print(f"  [PASS] Ollama is running: {r.json()}")
    else:
        raise Exception("bad status")
except Exception:
    print("  [WARN] Ollama is NOT running.")
    print("         Start it with:  ollama serve")
    print("         Then pull a model:  ollama pull mistral")

# Check Tesseract
print()
print("Checking Tesseract OCR...")
try:
    import pytesseract
    tess_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
        ver = pytesseract.get_tesseract_version()
        print(f"  [PASS] Tesseract found: version {ver}")
    else:
        print(f"  [WARN] Tesseract not found at: {tess_path}")
        print("         Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("         Then set TESSERACT_PATH in your .env file")
except Exception as e:
    print(f"  [WARN] Tesseract check failed: {e}")

# Check GPU
print()
print("Checking GPU (for model training)...")
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [PASS] GPU found: {gpu} ({vram:.1f} GB VRAM)")
        if vram < 8:
            print("  [WARN] Less than 8GB VRAM. Use Phi-3-mini (3.8B) instead of Mistral-7B")
        elif vram < 16:
            print("  [WARN] 8-16GB VRAM. QLoRA will work, use batch_size=2 in train.py")
        else:
            print("  [PASS] Sufficient VRAM for QLoRA fine-tuning of Mistral-7B")
    else:
        print("  [WARN] No CUDA GPU detected.")
        print("         Training requires an NVIDIA GPU.")
        print("         Dataset generation (Part 1 & 2) still works without a GPU.")
except ImportError:
    print("  [WARN] torch not found -- GPU check skipped")

# Check PDF folder
print()
from pathlib import Path
pdf_dir = Path("data/raw_pdfs")
pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
print("Checking PDFs in data/raw_pdfs/ ...")
if pdfs:
    print(f"  [PASS] {len(pdfs)} PDF(s) found:")
    for p in pdfs:
        print(f"         - {p.name}")
else:
    print("  [WARN] No PDFs found in data/raw_pdfs/")
    print("         Copy your Indian legal PDF files there before running Part 1.")

print()
print("NEXT STEPS:")
print("  1. Copy your legal PDFs to:  data\\raw_pdfs\\")
print("  2. Install Ollama from:       https://ollama.com")
print("  3. Run:  ollama pull mistral")
print("  4. Run:  python src\\part1_pdf_extraction\\extract_pdfs.py")
print()
