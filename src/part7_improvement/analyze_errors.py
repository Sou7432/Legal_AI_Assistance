"""
=============================================================
PART 7 — Model Improvement
=============================================================
File: src/part7_improvement/analyze_errors.py

This script:
  1. Loads predictions and ground truth from Part 6
  2. Identifies incorrect predictions (wrong law, wrong section)
  3. Groups errors by domain / risk category
  4. Suggests additional training examples for weak domains
  5. Compares base model vs fine-tuned model vs RAG-enhanced model
  6. Outputs an improvement action plan

HOW TO RUN:
    python src/part7_improvement/analyze_errors.py
=============================================================
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from tabulate import tabulate  # pip install tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TEST_CSV        = Path("data/dataset/test.csv")
PREDICTIONS_JSON = Path("reports/predictions.json")
REPORT_DIR      = Path("reports")
IMPROVEMENT_DIR = Path("reports/improvement")
IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_evaluation_data():
    test_df = pd.read_csv(TEST_CSV)
    with open(PREDICTIONS_JSON) as f:
        predictions = json.load(f)
    return test_df, predictions


# ─────────────────────────────────────────────
# 2. IDENTIFY INCORRECT PREDICTIONS
# ─────────────────────────────────────────────
def categorize_errors(test_df: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """
    Compare each prediction against ground truth.
    Label each row as: correct / wrong_law / wrong_section / wrong_risk / wrong_all
    """
    errors = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = predictions[i] if i < len(predictions) else {}
        true_law  = row.get("law", "").lower().strip()
        pred_law  = pred.get("law", "").lower().strip()
        true_sec  = row.get("section", "").lower().strip()
        pred_sec  = pred.get("section", "").lower().strip()
        true_risk = row.get("risk_category", "").strip()
        pred_risk = pred.get("risk_category", "").strip()

        law_correct  = true_law in pred_law or pred_law in true_law
        sec_correct  = true_sec in pred_sec or pred_sec in true_sec
        risk_correct = true_risk.lower() == pred_risk.lower()

        errors.append({
            "index":          i,
            "problem":        row.get("problem", ""),
            "true_law":       row.get("law", ""),
            "pred_law":       pred.get("law", ""),
            "true_section":   row.get("section", ""),
            "pred_section":   pred.get("section", ""),
            "true_risk":      true_risk,
            "pred_risk":      pred_risk,
            "law_correct":    law_correct,
            "section_correct": sec_correct,
            "risk_correct":   risk_correct,
            "fully_correct":  law_correct and sec_correct and risk_correct,
            "error_type": (
                "correct" if law_correct and sec_correct and risk_correct
                else "wrong_law" if not law_correct
                else "wrong_section" if not sec_correct
                else "wrong_risk"
            )
        })

    return pd.DataFrame(errors)


# ─────────────────────────────────────────────
# 3. ANALYZE ERRORS BY DOMAIN
# ─────────────────────────────────────────────
def analyze_errors_by_domain(error_df: pd.DataFrame) -> dict:
    """
    Group incorrect predictions by risk_category to find weak domains.
    """
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": []})

    for _, row in error_df.iterrows():
        domain = row["true_risk"] or "Unknown"
        domain_stats[domain]["total"] += 1
        if row["fully_correct"]:
            domain_stats[domain]["correct"] += 1
        else:
            domain_stats[domain]["errors"].append({
                "problem":      row["problem"][:100],
                "true_law":     row["true_law"],
                "pred_law":     row["pred_law"],
                "error_type":   row["error_type"]
            })

    # Compute accuracy per domain
    domain_accuracy = {}
    for domain, stats in domain_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        domain_accuracy[domain] = {
            "total":           total,
            "correct":         correct,
            "accuracy_pct":    round(correct / max(total, 1) * 100, 1),
            "error_count":     total - correct,
            "sample_errors":   stats["errors"][:3]  # first 3 error examples
        }

    return domain_accuracy


# ─────────────────────────────────────────────
# 4. IDENTIFY MOST COMMON WRONG PREDICTIONS
# ─────────────────────────────────────────────
def analyze_law_confusion(error_df: pd.DataFrame) -> dict:
    """
    Find which laws are most commonly confused with each other.
    E.g., model predicts "Indian Penal Code" when answer is "IT Act"
    """
    wrong_laws = error_df[~error_df["law_correct"]]
    confusion_pairs = Counter()

    for _, row in wrong_laws.iterrows():
        pair = (row["true_law"][:50], row["pred_law"][:50])
        confusion_pairs[pair] += 1

    return dict(confusion_pairs.most_common(20))


# ─────────────────────────────────────────────
# 5. COMPARE BASE vs FINE-TUNED vs RAG MODEL
# ─────────────────────────────────────────────
def compare_model_versions(
    base_report: dict,
    finetuned_report: dict,
    rag_report: dict
) -> pd.DataFrame:
    """
    Side-by-side comparison of three model versions.
    Call this after running Part 6 evaluation with each model variant.
    """
    metrics = [
        "law_exact_match_pct",
        "section_exact_match_pct",
        "solution_bert_score",
        "risk_category_accuracy_pct",
        "expected_answer_bert_score",
        "hallucination_rate_pct"
    ]

    rows = []
    for metric in metrics:
        rows.append({
            "Metric":       metric,
            "Base Model":   base_report.get(metric, "N/A"),
            "Fine-tuned":   finetuned_report.get(metric, "N/A"),
            "RAG-enhanced": rag_report.get(metric, "N/A")
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. GENERATE IMPROVEMENT PLAN
# ─────────────────────────────────────────────
def generate_improvement_plan(domain_accuracy: dict) -> str:
    """
    Generate actionable recommendations based on error analysis.
    """
    # Sort by accuracy (ascending = weakest domains first)
    sorted_domains = sorted(domain_accuracy.items(), key=lambda x: x[1]["accuracy_pct"])

    plan_lines = [
        "# AI Legal Assistant — Improvement Action Plan",
        "",
        "## Weakest Domains (prioritize these for more training examples)",
        ""
    ]

    for domain, stats in sorted_domains[:8]:
        plan_lines.append(
            f"### {domain} — {stats['accuracy_pct']}% accuracy "
            f"({stats['error_count']} errors / {stats['total']} total)"
        )
        plan_lines.append(
            f"**Action**: Add {max(50, stats['error_count'] * 3)} more training "
            f"examples covering edge cases in this domain."
        )
        if stats["sample_errors"]:
            plan_lines.append("**Sample Error:**")
            err = stats["sample_errors"][0]
            plan_lines.append(
                f"- Problem: {err['problem']}\n"
                f"- True law: {err['true_law']}\n"
                f"- Predicted: {err['pred_law']}\n"
            )

    plan_lines += [
        "## General Recommendations",
        "",
        "1. **Reduce hallucinations**: Increase RAG top_k from 5 to 8. "
           "This provides more legal context for the LLM.",
        "2. **Wrong law confusion**: Review law_confusion_pairs.json. "
           "Add contrastive examples to help the model distinguish similar laws.",
        "3. **Dataset quality**: Review errors manually. Remove mislabeled rows.",
        "4. **More training epochs**: If validation loss is still decreasing at "
           "epoch 3, train for 1–2 more epochs.",
        "5. **Increase LoRA rank**: Change LORA_R from 16 to 32 for more capacity "
           "if you have enough VRAM.",
        "6. **Domain-weighted sampling**: If some domains have very few examples, "
           "oversample them during training (use WeightedRandomSampler).",
        "7. **Legal BERT embedding**: Switch ChromaDB embedding model to "
           "'nlpaueb/legal-bert-base-uncased' for better legal concept matching.",
    ]

    return "\n".join(plan_lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    test_df, predictions = load_evaluation_data()
    log.info(f"Loaded {len(test_df)} test examples and {len(predictions)} predictions")

    # 1. Categorize errors
    error_df = categorize_errors(test_df, predictions)
    overall_accuracy = error_df["fully_correct"].mean() * 100
    log.info(f"Overall accuracy: {overall_accuracy:.1f}%")

    # Save error analysis
    error_df.to_csv(IMPROVEMENT_DIR / "error_analysis.csv", index=False)

    # 2. Analyze by domain
    domain_accuracy = analyze_errors_by_domain(error_df)
    with open(IMPROVEMENT_DIR / "domain_accuracy.json", "w") as f:
        # Remove sample_errors for JSON (too verbose)
        slim = {k: {ek: ev for ek, ev in v.items() if ek != "sample_errors"}
                for k, v in domain_accuracy.items()}
        json.dump(slim, f, indent=2)

    # Print domain table
    table_data = [
        [d, s["total"], s["correct"], f"{s['accuracy_pct']}%"]
        for d, s in sorted(domain_accuracy.items(), key=lambda x: x[1]["accuracy_pct"])
    ]
    print("\nDomain Accuracy Breakdown:")
    print(tabulate(table_data, headers=["Domain", "Total", "Correct", "Accuracy"]))

    # 3. Law confusion analysis
    confusion = analyze_law_confusion(error_df)
    with open(IMPROVEMENT_DIR / "law_confusion_pairs.json", "w") as f:
        # Convert tuple keys to strings
        str_confusion = {f"{k[0]} → {k[1]}": v for k, v in confusion.items()}
        json.dump(str_confusion, f, indent=2)
    log.info(f"Top 5 law confusions:")
    for pair, count in list(confusion.items())[:5]:
        log.info(f"  True: '{pair[0]}' | Predicted: '{pair[1]}' | Count: {count}")

    # 4. Generate improvement plan
    plan_text = generate_improvement_plan(domain_accuracy)
    plan_path = IMPROVEMENT_DIR / "improvement_plan.md"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_text)
    log.info(f"\nImprovement plan saved → {plan_path}")

    print(f"\n{'='*55}")
    print("IMPROVEMENT ANALYSIS COMPLETE")
    print(f"  Error analysis  → {IMPROVEMENT_DIR}/error_analysis.csv")
    print(f"  Domain accuracy → {IMPROVEMENT_DIR}/domain_accuracy.json")
    print(f"  Law confusion   → {IMPROVEMENT_DIR}/law_confusion_pairs.json")
    print(f"  Action plan     → {IMPROVEMENT_DIR}/improvement_plan.md")


if __name__ == "__main__":
    main()
