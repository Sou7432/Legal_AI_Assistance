"""
=============================================================
PART 3 — Dataset Preparation for Fine-Tuning
=============================================================
File: src/part3_tokenization/prepare_dataset.py

NOTE: Run from the project root directory:
    python -m src.part3_tokenization.prepare_dataset
  or:
    python src/part3_tokenization/prepare_dataset.py

This script:
  1. Converts each training row into an instruction-tuning JSONL format
  2. Tokenizes the dataset using the target model's tokenizer
  3. Checks token lengths to ensure they fit within model context window
  4. Prepares Hugging Face Dataset objects for training

HOW TO RUN:
    python src/part3_tokenization/prepare_dataset.py

CONCEPTS EXPLAINED:
  ─ RAG Knowledge Base:
    The original legal PDFs stored as vector embeddings in ChromaDB.
    Used at INFERENCE time to retrieve relevant legal sections.
    NOT used during fine-tuning directly.

  ─ Supervised Fine-Tuning (SFT) Dataset:
    Our 10,000-row CSV. Each row has an input (problem) and
    expected output (the 7 structured fields). Used to TRAIN the model.

  ─ Model Evaluation Dataset:
    The 10% test split. Used ONLY to measure accuracy AFTER training.
    The model never sees this data during training.
=============================================================
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR  = Path("data/dataset")
TRAIN_CSV    = DATASET_DIR / "train.csv"
VAL_CSV      = DATASET_DIR / "val.csv"
TEST_CSV     = DATASET_DIR / "test.csv"
JSONL_DIR    = DATASET_DIR / "jsonl"
JSONL_DIR.mkdir(parents=True, exist_ok=True)

# Change to your chosen model — same as the one you will fine-tune
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH   = 2048    # max token length the model supports (Mistral = 32768, but 2048 is practical)

# ─────────────────────────────────────────────
# INSTRUCTION FORMAT — imported from shared constants
# ─────────────────────────────────────────────
try:
    from src.constants import INSTRUCTION
except ImportError:
    from constants import INSTRUCTION


def row_to_jsonl(row: dict) -> dict:
    """
    Convert a single dataset row into instruction-tuning format.

    Format:
    {
        "instruction": <system instruction>,
        "input": <user's problem>,
        "output": <JSON with the 7 answer fields>
    }

    This is the standard format used by most SFT trainers (e.g., Alpaca-style).
    The TRL SFTTrainer will convert this to chat format automatically.
    """
    output_dict = {
        "law":                   row.get("law", "To be determined"),
        "section":               row.get("section", "To be determined"),
        "solution":              row.get("solution", ""),
        "solution_based_on_law": row.get("solution_based_on_law", ""),
        "risk_category":         row.get("risk_category", "Other"),
        "applicable_conditions": row.get("applicable_conditions", ""),
        "expected_answer":       row.get("expected_answer", "")
    }
    return {
        "instruction": INSTRUCTION,
        "input": row.get("problem", ""),
        "output": json.dumps(output_dict, ensure_ascii=False)
    }


def format_as_chat_prompt(example: dict) -> str:
    """
    Format as a Mistral/Llama instruct prompt for the tokenizer.
    This is the full text the model sees during training.
    """
    return (
        f"[INST] {example['instruction']}\n\n"
        f"User Problem: {example['input']} [/INST]\n"
        f"{example['output']}"
    )


# ─────────────────────────────────────────────
# CONVERT CSVs TO JSONL
# ─────────────────────────────────────────────
def convert_csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    jsonl_rows = []
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            jrow = row_to_jsonl(row.to_dict())
            jsonl_rows.append(jrow)
            f.write(json.dumps(jrow, ensure_ascii=False) + "\n")
    log.info(f"  ✓ {len(jsonl_rows)} rows → {jsonl_path}")
    return jsonl_rows


# ─────────────────────────────────────────────
# TOKENIZATION & LENGTH CHECK
# ─────────────────────────────────────────────
def check_token_lengths(jsonl_rows: list[dict], tokenizer) -> dict:
    """
    Tokenize all examples and report length statistics.
    Examples that exceed MAX_LENGTH will be truncated during training —
    too many long examples means the model might lose the output tail.
    """
    lengths = []
    over_limit = 0

    for row in tqdm(jsonl_rows, desc="Tokenizing length check"):
        text = format_as_chat_prompt(row)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        length = len(tokens)
        lengths.append(length)
        if length > MAX_LENGTH:
            over_limit += 1

    stats = {
        "total_examples":      len(lengths),
        "max_token_length":    max(lengths),
        "min_token_length":    min(lengths),
        "avg_token_length":    round(sum(lengths) / len(lengths), 1),
        "over_limit_count":    over_limit,
        "over_limit_pct":      round(over_limit / len(lengths) * 100, 2)
    }
    return stats


# ─────────────────────────────────────────────
# HUGGING FACE DATASET PREPARATION
# ─────────────────────────────────────────────
def tokenize_function(examples: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenize the formatted prompt for model training.
    Labels = input_ids (the model predicts the entire prompt+output sequence).
    Padding is done during the DataCollator step, not here.
    """
    texts = [
        format_as_chat_prompt({"instruction": inst, "input": inp, "output": out})
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,          # dynamic padding via DataCollator
        return_tensors=None     # return plain lists
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_hf_datasets(
    train_jsonl: list[dict],
    val_jsonl: list[dict],
    tokenizer,
    max_length: int
) -> DatasetDict:
    """
    Convert JSONL rows into Hugging Face Dataset objects with tokenization.
    Returns a DatasetDict with 'train' and 'validation' splits.
    """
    train_hf = Dataset.from_list(train_jsonl)
    val_hf   = Dataset.from_list(val_jsonl)

    # Tokenize
    train_tokenized = train_hf.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_hf.column_names,
        desc="Tokenizing train"
    )
    val_tokenized = val_hf.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_hf.column_names,
        desc="Tokenizing validation"
    )

    return DatasetDict({
        "train":      train_tokenized,
        "validation": val_tokenized
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. Convert CSVs → JSONL
    log.info("Converting CSVs to instruction-tuning JSONL format…")
    train_rows = convert_csv_to_jsonl(TRAIN_CSV, JSONL_DIR / "train.jsonl")
    val_rows   = convert_csv_to_jsonl(VAL_CSV,   JSONL_DIR / "val.jsonl")
    _          = convert_csv_to_jsonl(TEST_CSV,  JSONL_DIR / "test.jsonl")

    # 2. Load tokenizer
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    log.info("(First run will download from Hugging Face — needs internet)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Check token lengths
    log.info("Checking token lengths for training examples…")
    stats = check_token_lengths(train_rows + val_rows, tokenizer)
    log.info(f"\nToken Length Statistics:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    if stats["over_limit_pct"] > 10:
        log.warning(
            f"{stats['over_limit_pct']}% of examples exceed MAX_LENGTH={MAX_LENGTH}. "
            f"Consider increasing MAX_LENGTH or truncating longer examples."
        )

    # 4. Save tokenization stats
    stats_path = JSONL_DIR / "token_length_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved → {stats_path}")

    # 5. Prepare Hugging Face Datasets (tokenized, ready for SFTTrainer)
    log.info("\nPreparing Hugging Face Datasets…")
    dataset_dict = prepare_hf_datasets(train_rows, val_rows, tokenizer, MAX_LENGTH)

    # Save to disk for fast reuse during training
    hf_dataset_path = DATASET_DIR / "hf_tokenized"
    dataset_dict.save_to_disk(str(hf_dataset_path))
    log.info(f"\n✓ HF Datasets saved → {hf_dataset_path}")
    log.info(f"  Train: {len(dataset_dict['train'])} examples")
    log.info(f"  Validation: {len(dataset_dict['validation'])} examples")
    log.info("\nPart 3 complete. Ready for fine-tuning (Part 4).")


if __name__ == "__main__":
    main()
