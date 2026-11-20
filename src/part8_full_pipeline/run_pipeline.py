"""
=============================================================
PART 8 — Full Pipeline Orchestrator
=============================================================
File: src/part8_full_pipeline/run_pipeline.py

This is the master script that ties all parts together.

ARCHITECTURE:
  Legal PDFs
    ↓ Part 1: Extract & clean text, split into sections
  cleaned_sections/ (JSON)
    ↓ Part 5: Build ChromaDB vector index (RAG knowledge base)
  chroma_db/
    ↓ Part 2: Generate 10,000-row supervised dataset
  data/dataset/legal_dataset_10000.csv
    ↓ Part 3: Convert to instruction format, tokenize
  data/dataset/hf_tokenized/
    ↓ Part 4: QLoRA fine-tuning on Mistral-7B
  models/legal_llm/final/
    ↓ Part 6: Evaluate on test set
  reports/accuracy_report.json
    ↓ Part 7: Analyze errors, generate improvement plan
  reports/improvement/
    ↓ Part 8: Run the final AI Legal Assistant
  LIVE SYSTEM

HOW TO RUN:
    # Run everything from scratch:
    python src/part8_full_pipeline/run_pipeline.py --all

    # Or run specific stages:
    python src/part8_full_pipeline/run_pipeline.py --stage extract
    python src/part8_full_pipeline/run_pipeline.py --stage generate
    python src/part8_full_pipeline/run_pipeline.py --stage tokenize
    python src/part8_full_pipeline/run_pipeline.py --stage train
    python src/part8_full_pipeline/run_pipeline.py --stage rag
    python src/part8_full_pipeline/run_pipeline.py --stage evaluate
    python src/part8_full_pipeline/run_pipeline.py --stage improve
    python src/part8_full_pipeline/run_pipeline.py --stage demo
=============================================================
"""

import sys
import logging
import argparse
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STAGE RUNNERS
# ─────────────────────────────────────────────
def run_extract():
    log.info("=" * 55)
    log.info("STAGE 1: PDF Extraction")
    log.info("=" * 55)
    from src.part1_pdf_extraction.extract_pdfs import main
    main()


def run_generate():
    log.info("=" * 55)
    log.info("STAGE 2: Dataset Generation (10,000 rows)")
    log.info("=" * 55)
    from src.part2_dataset_creation.generate_dataset import main
    main()


def run_tokenize():
    log.info("=" * 55)
    log.info("STAGE 3: Tokenization & HF Dataset Preparation")
    log.info("=" * 55)
    from src.part3_tokenization.prepare_dataset import main
    main()


def run_train():
    log.info("=" * 55)
    log.info("STAGE 4: QLoRA Fine-Tuning")
    log.info("=" * 55)
    from src.part4_fine_tuning.train import main
    main()


def run_rag():
    log.info("=" * 55)
    log.info("STAGE 5: Build ChromaDB RAG Index")
    log.info("=" * 55)
    # Build the index from extracted sections
    import sys
    sys.argv = ["rag_pipeline.py", "--build-index"]
    from src.part5_rag.rag_pipeline import main
    main()


def run_evaluate():
    log.info("=" * 55)
    log.info("STAGE 6: Model Evaluation")
    log.info("=" * 55)
    import sys
    sys.argv = ["evaluate.py", "--model-path", "models/legal_llm/final", "--use-rag"]
    from src.part6_evaluation.evaluate import main
    main()


def run_improve():
    log.info("=" * 55)
    log.info("STAGE 7: Error Analysis & Improvement Plan")
    log.info("=" * 55)
    from src.part7_improvement.analyze_errors import main
    main()


# ─────────────────────────────────────────────
# INTERACTIVE DEMO
# ─────────────────────────────────────────────
def run_demo():
    """
    Interactive demo: enter a legal problem and get a full analysis.
    This is the final AI Legal Assistant in action.
    """
    import torch
    from src.part4_fine_tuning.train import load_for_inference
    from src.part5_rag.rag_pipeline import load_chroma_collection, generate_rag_answer, EMBEDDING_MODEL

    log.info("=" * 55)
    log.info("AI LEGAL ASSISTANT — DEMO")
    log.info("=" * 55)

    model_path = "models/legal_llm/final"
    log.info(f"Loading fine-tuned model from {model_path}…")
    model, tokenizer = load_for_inference(model_path)

    log.info("Loading RAG knowledge base…")
    _, collection, embedder = load_chroma_collection(EMBEDDING_MODEL)

    print("\n" + "=" * 55)
    print("Indian AI Legal Assistant")
    print("Type your legal problem in everyday language.")
    print("Type 'exit' to quit.")
    print("=" * 55)

    while True:
        print()
        problem = input("Your legal problem: ").strip()
        if problem.lower() in ("exit", "quit", "q"):
            break
        if not problem:
            continue

        print("\nAnalyzing your legal problem…")
        answer = generate_rag_answer(problem, collection, embedder, model, tokenizer)

        print("\n" + "-" * 50)
        print("LEGAL ANALYSIS")
        print("-" * 50)

        if answer.get("parse_error"):
            print("Raw output:", answer.get("raw_output", "No output generated"))
        else:
            fields = [
                ("Applicable Law",    "law"),
                ("Section",           "section"),
                ("Solution",          "solution"),
                ("Legal Basis",       "solution_based_on_law"),
                ("Risk Category",     "risk_category"),
                ("Conditions",        "applicable_conditions"),
                ("Full Answer",       "expected_answer"),
            ]
            for label, key in fields:
                value = answer.get(key, "N/A")
                print(f"\n{label}:")
                print(f"  {value}")

            sources = answer.get("retrieved_sources", [])
            if sources:
                print(f"\nSources Retrieved from Legal Database:")
                for src in sources:
                    print(f"  • {src['law']} — {src['section']} (similarity: {src['similarity']})")

        print("-" * 50)

    print("\nThank you for using the AI Legal Assistant.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
STAGES = {
    "extract":  run_extract,
    "generate": run_generate,
    "tokenize": run_tokenize,
    "train":    run_train,
    "rag":      run_rag,
    "evaluate": run_evaluate,
    "improve":  run_improve,
    "demo":     run_demo
}

FULL_PIPELINE = ["extract", "generate", "tokenize", "train", "rag", "evaluate", "improve"]


def main():
    parser = argparse.ArgumentParser(
        description="AI Legal Assistant — Full Pipeline Orchestrator"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run the complete pipeline from PDF extraction to evaluation"
    )
    parser.add_argument(
        "--stage", type=str, choices=list(STAGES.keys()),
        help="Run a specific pipeline stage"
    )
    args = parser.parse_args()

    if args.all:
        log.info("Running FULL PIPELINE…")
        for stage_name in FULL_PIPELINE:
            STAGES[stage_name]()
        log.info("\n" + "=" * 55)
        log.info("FULL PIPELINE COMPLETE")
        log.info("Run 'python run_pipeline.py --stage demo' to test the assistant")
    elif args.stage:
        STAGES[args.stage]()
    else:
        parser.print_help()
        print(f"\nAvailable stages: {', '.join(STAGES.keys())}")
        print("Run --all for the complete pipeline")


if __name__ == "__main__":
    # Add project root to path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
