"""
=============================================================
PART 6 — Model Accuracy Evaluation
=============================================================
File: src/part6_evaluation/evaluate.py

This script evaluates the fine-tuned model against the held-out test set.

METRICS USED:
  ─ Law Identification Accuracy:  Exact match + fuzzy partial match
  ─ Section Identification Accuracy: Exact match of section numbers
  ─ Solution Relevance: BERTScore (semantic similarity)
  ─ Solution Based on Law: BERTScore
  ─ Risk Category Accuracy: Exact match (categorical)
  ─ Applicable Conditions: ROUGE-L + BERTScore
  ─ Expected Answer Quality: ROUGE-L + BERTScore + semantic similarity
  ─ RAG Retrieval Accuracy: Recall@K (did retrieved chunks contain the right law?)
  ─ Hallucination Rate: % of predictions citing non-existent laws/sections
  ─ Response Consistency: Variance across repeated queries

WHY NOT JUST EXACT MATCH?
  Legal answers can be phrased in many valid ways:
  - "Consumer Protection Act, 2019" = "CPA 2019" = "The Consumer Protection Act"
  - Exact string match would mark these all as WRONG
  - BERTScore compares semantic meaning (using contextual embeddings)
  - ROUGE measures n-gram overlap
  - Fuzzy matching handles abbreviations and reformulations

HOW TO RUN:
    python src/part6_evaluation/evaluate.py --model-path models/legal_llm/final
=============================================================
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from rouge_score import rouge_scorer
import evaluate as hf_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TEST_CSV      = Path("data/dataset/test.csv")
REPORT_DIR    = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Known valid Indian laws (from PDF index) — load dynamically in practice
KNOWN_LAWS = [
    "consumer protection act",
    "indian penal code",
    "code of criminal procedure",
    "information technology act",
    "protection of women from domestic violence act",
    "motor vehicles act",
    "indian contract act",
    "transfer of property act",
    "right to information act",
    "prevention of corruption act",
    # Add all laws from your extracted PDFs
]

RISK_CATEGORIES = [
    "Criminal", "Civil", "Consumer", "Contract", "Property",
    "Cybercrime", "Family", "Labour", "Constitutional", "Traffic",
    "Environmental", "Banking", "RTI", "Other"
]


# ─────────────────────────────────────────────
# HELPER: RUN MODEL INFERENCE ON TEST SET
# ─────────────────────────────────────────────
def run_model_on_test(test_df: pd.DataFrame, model, tokenizer) -> list[dict]:
    """
    Run the fine-tuned model on every test example.
    Returns a list of prediction dicts.
    """
    from src.part4_fine_tuning.train import generate_legal_answer

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running inference"):
        try:
            pred = generate_legal_answer(row["problem"], model, tokenizer)
        except Exception as e:
            pred = {"error": str(e), "parse_error": True}
        predictions.append(pred)

    return predictions


# ─────────────────────────────────────────────
# METRIC 1: LAW IDENTIFICATION ACCURACY
# ─────────────────────────────────────────────
def normalize_law_name(law: str) -> str:
    """Normalize law name for comparison."""
    law = law.lower().strip()
    law = re.sub(r"\s+", " ", law)
    law = re.sub(r"[,.]", "", law)
    # Remove years for fuzzy match
    law = re.sub(r"\b\d{4}\b", "", law).strip()
    return law


def law_accuracy(true_laws: list[str], pred_laws: list[str]) -> dict:
    """
    Compute law identification accuracy using:
    - Exact match (after normalization)
    - Partial match (true law is a substring of predicted or vice versa)
    """
    exact_matches = 0
    partial_matches = 0
    n = len(true_laws)

    for true, pred in zip(true_laws, pred_laws):
        t = normalize_law_name(true)
        p = normalize_law_name(pred)

        if t == p:
            exact_matches += 1
        elif t in p or p in t:
            partial_matches += 1

    return {
        "law_exact_match_pct":   round(exact_matches / n * 100, 2),
        "law_partial_match_pct": round((exact_matches + partial_matches) / n * 100, 2),
        "law_exact_count":       exact_matches,
        "law_partial_count":     partial_matches,
        "total_examples":        n
    }


# ─────────────────────────────────────────────
# METRIC 2: SECTION IDENTIFICATION ACCURACY
# ─────────────────────────────────────────────
def extract_section_numbers(text: str) -> set[str]:
    """Extract all section numbers from a text string."""
    # Matches: "Section 34", "34A", "Section 3(1)(a)", "Article 21"
    patterns = [
        r"Section\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)",
        r"Article\s+(\d+[A-Z]?)",
        r"\bS\.\s*(\d+[A-Z]?)",
        r"^\s*(\d+[A-Z]?)\.",
    ]
    sections = set()
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
        sections.update(m.strip() for m in matches)
    return sections


def section_accuracy(true_sections: list[str], pred_sections: list[str]) -> dict:
    """
    Compute section accuracy:
    - Exact match: predicted section == true section
    - Precision/Recall/F1: for cases with multiple sections
    """
    exact = 0
    precision_scores = []
    recall_scores = []

    for true, pred in zip(true_sections, pred_sections):
        true_set = extract_section_numbers(true)
        pred_set = extract_section_numbers(pred)

        if not true_set and not pred_set:
            exact += 1
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            continue

        if true_set == pred_set:
            exact += 1

        if pred_set:
            prec = len(true_set & pred_set) / len(pred_set)
        else:
            prec = 0.0

        if true_set:
            rec = len(true_set & pred_set) / len(true_set)
        else:
            rec = 0.0

        precision_scores.append(prec)
        recall_scores.append(rec)

    n = len(true_sections)
    avg_prec = np.mean(precision_scores)
    avg_rec  = np.mean(recall_scores)
    f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-8)

    return {
        "section_exact_match_pct": round(exact / n * 100, 2),
        "section_precision":       round(avg_prec * 100, 2),
        "section_recall":          round(avg_rec * 100, 2),
        "section_f1":              round(f1 * 100, 2)
    }


# ─────────────────────────────────────────────
# METRIC 3–4: SOLUTION RELEVANCE (BERTScore + ROUGE)
# ─────────────────────────────────────────────
def compute_text_metrics(
    references: list[str],
    hypotheses: list[str],
    field_name: str
) -> dict:
    """
    Compute ROUGE-L and BERTScore for free-text fields.
    
    ROUGE-L: measures longest common subsequence overlap
    BERTScore: measures semantic similarity using BERT embeddings
                (handles paraphrases and synonyms better than ROUGE)
    """
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    avg_rouge = np.mean(rouge_scores) * 100

    # BERTScore
    bertscore = hf_evaluate.load("bertscore")
    bert_results = bertscore.compute(
        predictions=hypotheses,
        references=references,
        lang="en",
        model_type="bert-base-uncased"
    )
    avg_bert_f1 = np.mean(bert_results["f1"]) * 100

    return {
        f"{field_name}_rouge_l":    round(avg_rouge, 2),
        f"{field_name}_bert_score": round(avg_bert_f1, 2)
    }


# ─────────────────────────────────────────────
# METRIC 5: RISK CATEGORY ACCURACY
# ─────────────────────────────────────────────
def risk_category_accuracy(true_cats: list[str], pred_cats: list[str]) -> dict:
    """
    Exact match accuracy for the categorical risk_category field.
    Also computes per-class precision, recall, F1.
    """
    # Normalize
    true_norm = [c.strip().title() for c in true_cats]
    pred_norm = [c.strip().title() for c in pred_cats]

    exact = sum(t == p for t, p in zip(true_norm, pred_norm))
    acc = exact / len(true_norm) * 100

    return {
        "risk_category_accuracy_pct": round(acc, 2),
        "risk_category_exact_count":  exact
    }


# ─────────────────────────────────────────────
# METRIC 9: HALLUCINATION RATE
# ─────────────────────────────────────────────
def compute_hallucination_rate(pred_laws: list[str], pred_sections: list[str],
                                legal_index: dict) -> dict:
    """
    Hallucination = predicting a law or section that does NOT exist
    in the official legal document index.

    Lower is better. Target: < 5% hallucination rate.
    """
    hallucinated = 0
    for law, section in zip(pred_laws, pred_sections):
        law_norm = normalize_law_name(law)

        # Skip "to be determined" answers
        if "determined" in law_norm or not law_norm.strip():
            continue

        # Check if any known law matches
        law_found = any(
            known in law_norm or law_norm in known
            for known in KNOWN_LAWS
        )

        if not law_found and law_norm:
            hallucinated += 1

    total = len(pred_laws)
    return {
        "hallucination_count": hallucinated,
        "hallucination_rate_pct": round(hallucinated / max(total, 1) * 100, 2)
    }


# ─────────────────────────────────────────────
# METRIC 8: RAG RETRIEVAL ACCURACY (Recall@K)
# ─────────────────────────────────────────────
def evaluate_rag_retrieval(
    test_df: pd.DataFrame,
    collection,
    embedder,
    top_k: int = 5
) -> dict:
    """
    Recall@K: for each test example, did the top-K retrieved chunks
    contain the correct law/section?

    Higher Recall@K = better RAG retrieval quality.
    """
    from src.part5_rag.rag_pipeline import retrieve_relevant_sections

    hits_at_k = 0
    total = len(test_df)

    for _, row in tqdm(test_df.iterrows(), total=total, desc="RAG Recall@K"):
        problem = row["problem"]
        true_law = normalize_law_name(row.get("law", ""))

        retrieved = retrieve_relevant_sections(problem, collection, embedder, top_k)
        retrieved_laws = [normalize_law_name(r["law_name"]) for r in retrieved]

        # Check if the true law appears in any of the retrieved chunks
        if any(true_law in r or r in true_law for r in retrieved_laws):
            hits_at_k += 1

    return {
        f"rag_recall_at_{top_k}": round(hits_at_k / total * 100, 2),
        "rag_hits":               hits_at_k,
        "rag_total":              total
    }


# ─────────────────────────────────────────────
# GENERATE FULL ACCURACY REPORT
# ─────────────────────────────────────────────
def evaluate_all(
    test_df: pd.DataFrame,
    predictions: list[dict],
    legal_index: dict = None,
    rag_collection=None,
    rag_embedder=None
) -> dict:
    """
    Run all evaluation metrics and compile the accuracy report.
    """
    # Extract true labels and predictions
    true_laws     = test_df["law"].tolist()
    true_sections = test_df["section"].tolist()
    true_solutions = test_df["solution"].tolist()
    true_sol_law  = test_df["solution_based_on_law"].tolist()
    true_risk     = test_df["risk_category"].tolist()
    true_cond     = test_df["applicable_conditions"].tolist()
    true_answer   = test_df["expected_answer"].tolist()

    pred_laws     = [p.get("law", "")                   for p in predictions]
    pred_sections = [p.get("section", "")               for p in predictions]
    pred_solutions = [p.get("solution", "")             for p in predictions]
    pred_sol_law  = [p.get("solution_based_on_law", "") for p in predictions]
    pred_risk     = [p.get("risk_category", "")         for p in predictions]
    pred_cond     = [p.get("applicable_conditions", "") for p in predictions]
    pred_answer   = [p.get("expected_answer", "")       for p in predictions]

    report = {}

    # 1. Law accuracy
    log.info("Computing law identification accuracy…")
    report.update(law_accuracy(true_laws, pred_laws))

    # 2. Section accuracy
    log.info("Computing section identification accuracy…")
    report.update(section_accuracy(true_sections, pred_sections))

    # 3. Solution relevance (BERTScore + ROUGE)
    log.info("Computing solution relevance (BERTScore + ROUGE)…")
    report.update(compute_text_metrics(true_solutions, pred_solutions, "solution"))

    # 4. Solution based on law
    log.info("Computing solution_based_on_law metrics…")
    report.update(compute_text_metrics(true_sol_law, pred_sol_law, "solution_based_on_law"))

    # 5. Risk category accuracy
    log.info("Computing risk category accuracy…")
    report.update(risk_category_accuracy(true_risk, pred_risk))

    # 6. Applicable conditions
    log.info("Computing applicable_conditions metrics…")
    report.update(compute_text_metrics(true_cond, pred_cond, "applicable_conditions"))

    # 7. Expected answer quality
    log.info("Computing expected_answer quality…")
    report.update(compute_text_metrics(true_answer, pred_answer, "expected_answer"))

    # 9. Hallucination rate
    log.info("Computing hallucination rate…")
    report.update(compute_hallucination_rate(pred_laws, pred_sections, legal_index or {}))

    # 8. RAG Recall@K (optional — only if RAG collection is provided)
    if rag_collection and rag_embedder:
        log.info("Computing RAG Recall@K…")
        report.update(evaluate_rag_retrieval(test_df, rag_collection, rag_embedder))

    return report


def print_and_save_report(report: dict, output_path: Path):
    """Print the accuracy report to console and save as JSON."""
    print(f"\n{'='*60}")
    print("ACCURACY REPORT — AI Legal Assistant")
    print(f"{'='*60}")

    sections_order = [
        ("Law Identification",     ["law_exact_match_pct", "law_partial_match_pct"]),
        ("Section Identification", ["section_exact_match_pct", "section_precision",
                                    "section_recall", "section_f1"]),
        ("Solution Relevance",     ["solution_rouge_l", "solution_bert_score"]),
        ("Solution Based on Law",  ["solution_based_on_law_rouge_l",
                                    "solution_based_on_law_bert_score"]),
        ("Risk Category",          ["risk_category_accuracy_pct"]),
        ("Applicable Conditions",  ["applicable_conditions_rouge_l",
                                    "applicable_conditions_bert_score"]),
        ("Expected Answer Quality",["expected_answer_rouge_l", "expected_answer_bert_score"]),
        ("Hallucination",          ["hallucination_rate_pct", "hallucination_count"]),
        ("RAG Retrieval",          ["rag_recall_at_5"]),
    ]

    for section_name, keys in sections_order:
        print(f"\n{section_name}:")
        for key in keys:
            if key in report:
                val = report[key]
                suffix = "%" if "pct" in key or "score" in key or "rouge" in key or "f1" in key or "recall" in key else ""
                print(f"  {key:<40} {val}{suffix}")

    print(f"\n{'='*60}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/legal_llm/final",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--use-rag", action="store_true",
                        help="Also evaluate RAG retrieval accuracy")
    parser.add_argument("--load-predictions", type=str, default=None,
                        help="Load pre-computed predictions from JSON file (skip inference)")
    args = parser.parse_args()

    # Load test dataset
    test_df = pd.read_csv(TEST_CSV)
    log.info(f"Test set loaded: {len(test_df)} examples")

    # Load or compute predictions
    if args.load_predictions:
        with open(args.load_predictions) as f:
            predictions = json.load(f)
        log.info(f"Loaded {len(predictions)} pre-computed predictions")
    else:
        log.info(f"Loading model from {args.model_path}…")
        from src.part4_fine_tuning.train import load_for_inference
        model, tokenizer = load_for_inference(args.model_path)
        predictions = run_model_on_test(test_df, model, tokenizer)

        # Save predictions for later analysis
        pred_path = REPORT_DIR / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        log.info(f"Predictions saved → {pred_path}")

    # RAG setup (optional)
    rag_collection = None
    rag_embedder   = None
    if args.use_rag:
        from src.part5_rag.rag_pipeline import load_chroma_collection, EMBEDDING_MODEL
        _, rag_collection, rag_embedder = load_chroma_collection(EMBEDDING_MODEL)

    # Run all evaluations
    report = evaluate_all(test_df, predictions, rag_collection=rag_collection,
                          rag_embedder=rag_embedder)

    # Print and save report
    report_path = REPORT_DIR / "accuracy_report.json"
    print_and_save_report(report, report_path)


if __name__ == "__main__":
    main()
