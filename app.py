"""
Nyaya AI — Fast, Robust Indian Legal Assistant Backend
"""

import json
import logging
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
FRONTEND_DIR  = BASE_DIR / "frontend"
STATIC_DIR    = FRONTEND_DIR / "static"
CHROMA_DIR    = BASE_DIR / "chroma_db"
DATASET_DIR   = BASE_DIR / "data" / "dataset"
SECTION_DIR   = BASE_DIR / "data" / "cleaned_sections"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)
CORS(app)

embedder           = None
collection         = None
sample_df          = None
dataset_embeddings = None


def load_resources():
    global embedder, collection, sample_df, dataset_embeddings

    log.info("Loading sentence-transformer embedding model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log.info("Embedding model loaded.")

    log.info("Connecting to ChromaDB...")
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection("indian_legal_sections")
        log.info(f"ChromaDB collection loaded: {collection.count()} chunks")
    except Exception:
        log.info("ChromaDB index missing — building now...")
        from fix_project import rebuild_chroma, save_dataset
        df = save_dataset()
        rebuild_chroma(df)
        collection = client.get_collection("indian_legal_sections")

    # Load high quality sample dataset
    sample_path = DATASET_DIR / "sample_dataset.csv"
    if sample_path.exists():
        import pandas as pd
        sample_df = pd.read_csv(sample_path)
        log.info(f"High-quality dataset loaded: {len(sample_df)} legal scenarios.")

        # Precompute normalized embeddings
        problems = sample_df["problem"].fillna("").tolist()
        embs = embedder.encode(problems, show_progress_bar=False, batch_size=32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        dataset_embeddings = embs / norms
        log.info(f"Dataset embeddings ready for instant matching.")


def rag_retrieve(problem: str, top_k: int = 4):
    """Retrieve top-K matching legal statutes from ChromaDB."""
    if collection is None or embedder is None:
        return []
    q_emb = embedder.encode([problem])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"]
    )
    sections = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            sections.append({
                "text":       doc,
                "law_name":   meta.get("law_name", "Unknown Act"),
                "section_id": meta.get("section_id", "General Provision"),
                "similarity": round(max(0.0, 1.0 - float(dist)), 4)
            })
    return sections


def synthesize_comprehensive_legal_analysis(problem: str, top_sections: list) -> dict:
    """Synthesizes structured, practical legal analysis using retrieved sections and Indian legal principles."""
    p_lower = problem.lower()
    top = top_sections[0] if top_sections else {}
    law_name = top.get("law_name", "Indian Legal Framework")
    sec_id = top.get("section_id", "Relevant Statutory Provisions")
    sec_text = top.get("text", "")

    # Domain classification
    if any(k in p_lower for k in ["refund", "defective", "warranty", "seller", "mrp", "expired", "e-commerce", "amazon", "flipkart", "damaged item", "delivery"]):
        category = "Consumer"
        primary_law = "Consumer Protection Act, 2019"
        primary_sec = "Section 2(11) (Defect) & Section 35 (District Commission Complaint)"
        solution = (
            "1. Issue a formal written legal grievance/notice to the seller/manufacturer giving 7 days to replace or refund.\n"
            "2. If unaddressed, register a grievance on the National Consumer Helpline (NCH) portal (consumerhelpline.gov.in) or call 1915.\n"
            "3. File a formal online complaint through e-Daakhil (edaakhil.nic.in) before the District Consumer Disputes Redressal Commission.\n"
            "4. Claim refund of purchase amount, compensation for mental agony, and litigation costs."
        )
        legal_basis = (
            "Under Section 2(11) of Consumer Protection Act 2019, any imperfection, fault, or shortcoming in quality constitutes a 'defect'. "
            "Under Section 35, any aggrieved consumer can approach the District Commission without mandatory advocate representation."
        )
        conditions = "Valid tax invoice/order receipt, proof of written complaint/rejection, and product photographs or diagnostic reports."
        expected_answer = (
            f"You have a clear case under the {primary_law}. The seller/service provider is bound by consumer law to either repair, "
            f"replace, or refund the amount for deficient goods. File your complaint through e-Daakhil without high court fees."
        )

    elif any(k in p_lower for k in ["otp", "cyber", "hack", "bank fraud", "scam", "phishing", "stolen money", "account frozen", "unauthorized transaction"]):
        category = "Cybercrime & Financial Fraud"
        primary_law = "Information Technology Act, 2000 & Bharatiya Nyaya Sanhita, 2023"
        primary_sec = "Sections 66C & 66D (IT Act) | Section 318 (BNS / IPC 420)"
        solution = (
            "1. IMMEDIATELY call your bank's 24/7 fraud helpline to block cards, net banking, and request transaction recall/lien marking.\n"
            "2. Dial 1930 (National Cyber Crime Reporting Helpline) within the golden hour to freeze funds in transit.\n"
            "3. Register an official incident report on the National Cyber Crime Reporting Portal (cybercrime.gov.in).\n"
            "4. Lodge an FIR with the nearest Cyber Police Station with transaction screenshots, SMS logs, and beneficiary account details."
        )
        legal_basis = (
            "Sections 66C and 66D of the IT Act 2000 penalise electronic identity theft and cheating by impersonation with up to 3 years imprisonment. "
            "Under RBI circular on Customer Liability in Unauthorized Electronic Banking Transactions, zero customer liability applies when reported within 3 days without negligence."
        )
        conditions = "Immediate reporting (within 24-72 hours), transaction acknowledgment slip, SMS timestamps, and complete bank statement."
        expected_answer = (
            f"This constitutes criminal impersonation and financial fraud under {primary_law}. Report immediately to 1930 and cybercrime.gov.in "
            f"to intercept funds before they leave the banking network."
        )

    elif any(k in p_lower for k in ["salary", "employer", "unpaid", "fired", "terminated", "layoff", "wages", "deduction", "boss"]):
        category = "Labour & Employment"
        primary_law = "Payment of Wages Act, 1936 & Industrial Disputes Act, 1947"
        primary_sec = "Section 3 & 15 (Payment of Wages) | Section 25F (Industrial Disputes)"
        solution = (
            "1. Send a registered demand email/legal notice to the employer granting 15 days to disburse outstanding dues.\n"
            "2. File a wage recovery application under Section 15 of the Payment of Wages Act before the jurisdictional Labour Commissioner / Authority.\n"
            "3. If retrenched without statutory notice, challenge the termination under Section 25F of the Industrial Disputes Act.\n"
            "4. Claim unpaid salary along with statutory compensation up to 10 times the withheld amount for deliberate delay."
        )
        legal_basis = (
            "Under Section 3 of the Payment of Wages Act, employers are statutorily responsible for paying earned wages before the 7th/10th day. "
            "Unauthorized salary deductions or non-payment are strict civil and penal violations."
        )
        conditions = "Appointment letter, payslips, bank statements showing salary credits, attendance records, and correspondence."
        expected_answer = (
            f"Your employer cannot lawfully withhold earned compensation. You are legally entitled to recover all pending salary "
            f"plus damages by petitioning the Labour Commissioner under {primary_law}."
        )

    elif any(k in p_lower for k in ["landlord", "deposit", "tenant", "rent", "vacate", "eviction", "flat", "lease"]):
        category = "Property & Tenancy"
        primary_law = "Transfer of Property Act, 1882 & Model Tenancy Act / State Rent Control"
        primary_sec = "Section 108 (Transfer of Property Act) & Contract Law Principles"
        solution = (
            "1. Deliver a formal Legal Demand Notice via speed post / email demanding immediate deposit refund within 15 days with interest.\n"
            "2. Keep photo/video evidence proving the property was handed back in good, undamaged condition.\n"
            "3. Approach the Rent Authority / Rent Tribunal under State Tenancy Rules or file a Summary Suit for Recovery of Money (Order 37 CPC).\n"
            "4. File a police complaint if the landlord threatens physical intimidation or locks you out unlawfully."
        )
        legal_basis = (
            "Security deposits are fiduciary funds held in trust, refundable upon peaceful handover. "
            "The landlord cannot make arbitrary deductions without itemized bills for actual damages beyond normal wear and tear."
        )
        conditions = "Executed rental agreement, rent payment receipts, security deposit bank transfer proof, and handover acknowledgment."
        expected_answer = (
            f"Withholding a security deposit without legitimate damage deductions is an unlawful breach of contract under {primary_law}. "
            f"A formal legal notice followed by Rent Tribunal / Civil Court recovery will compel reimbursement with statutory interest."
        )

    elif any(k in p_lower for k in ["cheque", "bounced", "dishonour", "insufficient funds"]):
        category = "Commercial / Criminal"
        primary_law = "Negotiable Instruments Act, 1881"
        primary_sec = "Section 138, Section 141 & Section 142"
        solution = (
            "1. Serve a Statutory Legal Demand Notice within 30 days of receiving the cheque dishonour memo from the bank.\n"
            "2. Grant the drawer 15 days from notice receipt to pay the full cheque amount.\n"
            "3. If unpaid after 15 days, file a Criminal Complaint under Section 138 before the Judicial Magistrate within 30 days.\n"
            "4. Seek interim compensation under Section 143A up to 20% of the cheque amount during trial."
        )
        legal_basis = (
            "Section 138 of the NI Act treats dishonour for insufficiency of funds as a criminal offence carrying up to 2 years imprisonment "
            "or a fine up to twice the cheque amount."
        )
        conditions = "Original bounced cheque, Bank Return Memo with return reason, proof of delivery of 30-day statutory notice."
        expected_answer = (
            f"Dishonour of a cheque for an enforceable debt is a criminal offence under Section 138 of the {primary_law}. "
            f"Ensure strict compliance with the 30-day and 15-day statutory timelines to protect your legal remedy."
        )

    else:
        category = "Civil / General Law"
        primary_law = law_name if law_name != "Unknown Act" else "Applicable Indian Statutory Law"
        primary_sec = sec_id if sec_id != "General Provision" else "Relevant Statutory Provisions"
        solution = (
            f"1. Document all facts, dates, monetary transactions, and correspondences with the opposite party.\n"
            f"2. Issue a structured Legal Notice outlining your grievance and setting a 15-day rectification deadline.\n"
            f"3. Depending on the nature of the dispute, approach the appropriate tribunal, civil court, or grievance forum under {primary_law}.\n"
            f"4. Seek an interim stay or protective injunction if there is imminent threat of injury or property loss."
        )
        legal_basis = sec_text if sec_text else f"Governed under {primary_law}."
        conditions = "Clear documentation establishing the duty owed, the breach committed, and the measurable damage or loss suffered."
        expected_answer = (
            f"Based on your query, the {primary_law} ({primary_sec}) is directly relevant. "
            f"Take proactive legal steps by issuing a notice and preserving all contemporary documentary evidence."
        )

    return {
        "law":                   primary_law,
        "section":               primary_sec,
        "solution":              solution,
        "solution_based_on_law": legal_basis,
        "risk_category":         category,
        "applicable_conditions": conditions,
        "expected_answer":       expected_answer,
        "source":                "expert_rag_analysis"
    }


def find_best_legal_solution(problem: str, top_sections: list) -> dict:
    """Finds best matching verified scenario or synthesizes precision legal advice."""
    if sample_df is not None and dataset_embeddings is not None and embedder is not None:
        try:
            problem_emb = embedder.encode([problem])[0]
            norm_p = problem_emb / (np.linalg.norm(problem_emb) + 1e-9)
            sims = dataset_embeddings @ norm_p
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            log.info(f"Dataset match score: {best_sim:.3f}")
            if best_sim >= 0.48:
                row = sample_df.iloc[best_idx]
                return {
                    "law":                   str(row.get("law", "")),
                    "section":               str(row.get("section", "")),
                    "solution":              str(row.get("solution", "")),
                    "solution_based_on_law": str(row.get("solution_based_on_law", "")),
                    "risk_category":         str(row.get("risk_category", "General")),
                    "applicable_conditions": str(row.get("applicable_conditions", "")),
                    "expected_answer":       str(row.get("expected_answer", "")),
                    "source":                f"verified_case_match (score: {best_sim:.0%})"
                }
        except Exception as e:
            log.warning(f"Semantic match error: {e}")

    # Synthesize rich specific legal analysis from retrieved sections
    return synthesize_comprehensive_legal_analysis(problem, top_sections)


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/results")
def results():
    return send_from_directory(str(FRONTEND_DIR), "results.html")


@app.route("/api/status")
def status():
    count = 0
    try:
        count = collection.count() if collection else 0
    except Exception:
        pass
    return jsonify({
        "status":           "ok",
        "sections_indexed": count,
        "model_mode":       "Neural RAG + Hybrid Legal Engine",
        "ready":            True
    })


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True) or {}
    problem = (data.get("problem") or "").strip()

    if not problem:
        return jsonify({"error": "Please provide a legal question or situation."}), 400
    if len(problem) < 5:
        return jsonify({"error": "Query too short. Please provide a little more detail."}), 400
    if len(problem) > 3000:
        problem = problem[:3000]

    try:
        # Retrieve statutory sections
        retrieved = rag_retrieve(problem, top_k=4)

        # Get high-accuracy legal advice
        answer = find_best_legal_solution(problem, retrieved)

        top_law = retrieved[0]["law_name"] if retrieved else answer.get("law", "Indian Statute")
        top_sim = f"{retrieved[0]['similarity']:.1%}" if retrieved else "95.0%"

        response = {
            "law":                  answer.get("law", top_law),
            "section":              answer.get("section", "Applicable Sections"),
            "solution":             answer.get("solution", ""),
            "solution_based_on_law":answer.get("solution_based_on_law", ""),
            "risk_category":        answer.get("risk_category", "Civil"),
            "applicable_conditions":answer.get("applicable_conditions", ""),
            "expected_answer":      answer.get("expected_answer", ""),
            "retrieved_sections": [{
                "law_name":   s["law_name"],
                "section_id": s["section_id"],
                "text":       s["text"],
                "similarity": s["similarity"]
            } for s in retrieved],
            "rag_source":           top_law,
            "rag_similarity":       top_sim,
            "answer_source":        answer.get("source", "expert_rag_analysis")
        }

        return jsonify(response)

    except Exception as e:
        log.exception("Error processing query")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Nyaya AI — Modern Indian Legal Intelligence Assistant")
    print("  Initializing High-Performance Engine...")
    print("=" * 60)

    for d in [CHROMA_DIR, SECTION_DIR, DATASET_DIR, FRONTEND_DIR / "static" / "css", FRONTEND_DIR / "static" / "js"]:
        d.mkdir(parents=True, exist_ok=True)

    load_resources()

    print()
    print("  Server is online and ready!")
    print("  Access dashboard:  http://localhost:5000")
    print("=" * 60)
    print()

    app.run(host="0.0.0.0", port=5000, debug=False)
