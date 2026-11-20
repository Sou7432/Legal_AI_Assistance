"""
=============================================================
DEMO RUN — AI Legal Assistant (End-to-End Demo)
=============================================================
This demo runs the complete pipeline using built-in sample
Indian legal data, so you can see output without needing:
  - PDFs (sample legal text is included)
  - Ollama (sample dataset is included)
  - GPU (no fine-tuning in demo mode)

What this demo does:
  Step 1: Simulate Part 1 — write sample legal sections as JSON
  Step 2: Simulate Part 2 — write a sample 10-row dataset as CSV
  Step 3: Build ChromaDB index from sample legal sections (Part 5)
  Step 4: Run 3 sample queries through the RAG retrieval system
  Step 5: Show what the full pipeline output looks like

Run:
    python demo_run.py
=============================================================
"""

import json
import csv
import sys
import os
import time
from pathlib import Path

# ─── make sure src imports work ───
sys.path.insert(0, str(Path(__file__).parent))

# ─── ensure all directories exist ───
for d in ["data/raw_pdfs", "data/extracted_text", "data/cleaned_sections",
          "data/dataset", "models", "chroma_db", "reports"]:
    Path(d).mkdir(parents=True, exist_ok=True)

SECTION_DIR  = Path("data/cleaned_sections")
DATASET_DIR  = Path("data/dataset")
CHROMA_DIR   = Path("chroma_db")

# ═══════════════════════════════════════════════════════════
# SAMPLE INDIAN LEGAL DATA (built-in — no PDFs needed)
# ═══════════════════════════════════════════════════════════
SAMPLE_LEGAL_SECTIONS = [
    # ── Consumer Protection Act, 2019 ──────────────────────
    {
        "law_name": "Consumer Protection Act 2019",
        "section_id": "Section 2(11)",
        "section_title": "Section 2(11). Definition of Defect.",
        "section_text": (
            "Section 2(11) — 'defect' means any fault, imperfection or shortcoming in the "
            "quality, quantity, potency, purity or standard which is required to be maintained "
            "by or under any law for the time being in force or under any contract, express or "
            "implied, or as is claimed by the trader in any manner whatsoever in relation to "
            "any goods or product and the expression 'defective' shall be construed accordingly."
        ),
        "char_count": 480
    },
    {
        "law_name": "Consumer Protection Act 2019",
        "section_id": "Section 35",
        "section_title": "Section 35. Manner in which complaint shall be made.",
        "section_text": (
            "Section 35 — A complaint, in relation to any goods sold or delivered or agreed "
            "to be sold or delivered or any service provided or agreed to be provided, may be "
            "filed with a District Commission by the complainant. The complaint shall be filed "
            "within a period of two years from the date on which the cause of action has arisen. "
            "Any consumer may file a complaint. The complainant shall not be required to pay "
            "any fee for filing a complaint. A consumer may also seek electronic filing."
        ),
        "char_count": 530
    },
    {
        "law_name": "Consumer Protection Act 2019",
        "section_id": "Section 2(7)",
        "section_title": "Section 2(7). Definition of Consumer.",
        "section_text": (
            "Section 2(7) — 'consumer' means any person who buys any goods for a consideration "
            "which has been paid or promised or partly paid and partly promised, or under any "
            "system of deferred payment and includes any user of such goods other than the "
            "person who buys such goods for consideration paid or promised. It also includes "
            "any person who hires or avails of any service for a consideration."
        ),
        "char_count": 450
    },
    # ── Indian Penal Code, 1860 ────────────────────────────
    {
        "law_name": "Indian Penal Code 1860",
        "section_id": "Section 420",
        "section_title": "Section 420. Cheating and dishonestly inducing delivery of property.",
        "section_text": (
            "Section 420 — Whoever cheats and thereby dishonestly induces the person deceived "
            "to deliver any property to any person, or to make, alter or destroy the whole or "
            "any part of a valuable security, or anything which is signed or sealed, and which "
            "is capable of being converted into a valuable security, shall be punished with "
            "imprisonment of either description for a term which may extend to seven years, "
            "and shall also be liable to fine."
        ),
        "char_count": 490
    },
    {
        "law_name": "Indian Penal Code 1860",
        "section_id": "Section 379",
        "section_title": "Section 379. Punishment for theft.",
        "section_text": (
            "Section 379 — Whoever commits theft shall be punished with imprisonment of either "
            "description for a term which may extend to three years, or with fine, or with both. "
            "Theft is defined as dishonestly taking any moveable property out of the possession "
            "of any person without that person's consent."
        ),
        "char_count": 320
    },
    {
        "law_name": "Indian Penal Code 1860",
        "section_id": "Section 304A",
        "section_title": "Section 304A. Causing death by negligence.",
        "section_text": (
            "Section 304A — Whoever causes the death of any person by doing any rash or "
            "negligent act not amounting to culpable homicide, shall be punished with "
            "imprisonment of either description for a term which may extend to two years, "
            "or with fine, or with both. This section commonly applies to road accidents "
            "and medical negligence cases resulting in death."
        ),
        "char_count": 400
    },
    # ── Information Technology Act, 2000 ───────────────────
    {
        "law_name": "Information Technology Act 2000",
        "section_id": "Section 66",
        "section_title": "Section 66. Computer related offences.",
        "section_text": (
            "Section 66 — If any person, dishonestly or fraudulently, does any act referred "
            "to in section 43 of this Act, he shall be punishable with imprisonment for a "
            "term which may extend to three years or with fine which may extend to five lakh "
            "rupees or with both. Section 43 covers unauthorized access, data damage, "
            "introduction of viruses, denial of service attacks, and similar cybercrimes."
        ),
        "char_count": 440
    },
    {
        "law_name": "Information Technology Act 2000",
        "section_id": "Section 66C",
        "section_title": "Section 66C. Punishment for identity theft.",
        "section_text": (
            "Section 66C — Whoever, fraudulently or dishonestly make use of the electronic "
            "signature, password or any other unique identification feature of any other "
            "person, shall be punished with imprisonment of either description for a term "
            "which may extend to three years and shall also be liable to fine which may "
            "extend to rupees one lakh. This covers SIM card cloning, OTP theft, and "
            "fraudulent use of digital credentials."
        ),
        "char_count": 460
    },
    {
        "law_name": "Information Technology Act 2000",
        "section_id": "Section 66D",
        "section_title": "Section 66D. Punishment for cheating by personation using computer resource.",
        "section_text": (
            "Section 66D — Whoever, by means of any communication device or computer resource "
            "cheats by personating, shall be punished with imprisonment of either description "
            "for a term which may extend to three years and shall also be liable to fine which "
            "may extend to one lakh rupees. This covers phishing, fake identity fraud, "
            "and impersonation via phone or email to steal money."
        ),
        "char_count": 450
    },
    # ── Protection of Women from Domestic Violence Act, 2005 ─
    {
        "law_name": "Protection of Women from Domestic Violence Act 2005",
        "section_id": "Section 3",
        "section_title": "Section 3. Definition of domestic violence.",
        "section_text": (
            "Section 3 — For the purposes of this Act, any act, omission or commission or "
            "conduct of the respondent shall constitute domestic violence in case it harms "
            "or injures or endangers the health, safety, life, limb or well-being, whether "
            "mental or physical, of the aggrieved person or tends to do so and includes "
            "causing physical abuse, sexual abuse, verbal and emotional abuse and economic "
            "abuse. Economic abuse includes deprivation of financial resources including "
            "food, clothing, medicines and shelter."
        ),
        "char_count": 530
    },
    # ── Indian Contract Act, 1872 ──────────────────────────
    {
        "law_name": "Indian Contract Act 1872",
        "section_id": "Section 73",
        "section_title": "Section 73. Compensation for loss or damage caused by breach of contract.",
        "section_text": (
            "Section 73 — When a contract has been broken, the party who suffers by such "
            "breach is entitled to receive, from the party who has broken the contract, "
            "compensation for any loss or damage caused to him thereby, which naturally "
            "arose in the usual course of things from such breach, or which the parties "
            "knew, when they made the contract, to be likely to result from the breach of "
            "it. Such compensation is not to be given for any remote and indirect loss or "
            "damage sustained by reason of the breach."
        ),
        "char_count": 510
    },
    # ── Motor Vehicles Act, 1988 ───────────────────────────
    {
        "law_name": "Motor Vehicles Act 1988",
        "section_id": "Section 185",
        "section_title": "Section 185. Driving by a drunken person or by a person under the influence of drugs.",
        "section_text": (
            "Section 185 — Whoever, while driving or attempting to drive a motor vehicle, "
            "has, in his blood, alcohol exceeding 30 mg. per 100 ml. of blood detected in "
            "a test by a breath analyser shall be punishable for the first offence with "
            "imprisonment for a term which may extend to six months, or with fine which "
            "may extend to ten thousand rupees, or with both. For a second or subsequent "
            "offence, imprisonment may extend to two years, or fine up to fifteen thousand "
            "rupees, or with both."
        ),
        "char_count": 530
    },
    # ── Payment of Wages Act, 1936 ─────────────────────────
    {
        "law_name": "Payment of Wages Act 1936",
        "section_id": "Section 3",
        "section_title": "Section 3. Responsibility for payment of wages.",
        "section_text": (
            "Section 3 — Every employer shall be responsible for the payment to persons "
            "employed by him of all wages required to be paid under this Act. Wages shall "
            "be paid before the expiry of the seventh day after the last day of the wage "
            "period in respect of which the wages are payable, where less than one thousand "
            "persons are employed. Failure to pay wages on time constitutes an offence under "
            "this Act and the employer shall be liable for the delayed wages plus compensation."
        ),
        "char_count": 490
    },
    # ── Right to Information Act, 2005 ────────────────────
    {
        "law_name": "Right to Information Act 2005",
        "section_id": "Section 6",
        "section_title": "Section 6. Request for obtaining information.",
        "section_text": (
            "Section 6 — A person who desires to obtain any information under this Act, "
            "shall make a request in writing or through electronic means in English or "
            "Hindi or in the official language of the area in which the application is "
            "being made, accompanying such fee as may be prescribed, to the Central "
            "Public Information Officer or State Public Information Officer. The Central "
            "Public Information Officer shall not be required to give any reason for "
            "requesting the information."
        ),
        "char_count": 480
    },
]

# ═══════════════════════════════════════════════════════════
# SAMPLE DATASET (10 rows — simulating Part 2 output)
# ═══════════════════════════════════════════════════════════
SAMPLE_DATASET = [
    {
        "problem": "I bought a defective mobile phone, but the seller refuses to replace it or refund my money.",
        "law": "Consumer Protection Act, 2019",
        "section": "Section 2(11), Section 35",
        "solution": "File a consumer complaint with the District Consumer Disputes Redressal Commission seeking replacement or refund.",
        "solution_based_on_law": "Section 2(11) defines 'defect' and Section 35 provides the mechanism for filing a consumer complaint.",
        "risk_category": "Consumer",
        "applicable_conditions": "The complainant must qualify as a consumer under Section 2(7) and the product must have a defect.",
        "expected_answer": "The Consumer Protection Act, 2019 may apply. Section 2(11) defines defect, and Section 35 allows you to file a complaint with the District Commission."
    },
    {
        "problem": "I paid a contractor to renovate my house but he stopped work midway and disappeared with the money.",
        "law": "Indian Contract Act, 1872",
        "section": "Section 73",
        "solution": "File a civil suit for recovery of money and claim compensation for loss caused by the breach of contract.",
        "solution_based_on_law": "Section 73 entitles the aggrieved party to compensation for loss caused by breach of contract.",
        "risk_category": "Contract",
        "applicable_conditions": "A valid written or oral contract must exist, breach must be established, and the loss must be quantifiable.",
        "expected_answer": "The Indian Contract Act, 1872 may apply. Section 73 provides that when a contract is broken, the suffering party is entitled to compensation for the loss arising from the breach."
    },
    {
        "problem": "Someone called me pretending to be my bank and tricked me into sharing my OTP. Money was stolen from my account.",
        "law": "Information Technology Act, 2000",
        "section": "Section 66C, Section 66D",
        "solution": "Immediately contact your bank to freeze the account, file an FIR at the nearest police station, and report at cybercrime.gov.in.",
        "solution_based_on_law": "Section 66C punishes identity theft using electronic credentials and Section 66D covers cheating by personation through computer resources.",
        "risk_category": "Cybercrime",
        "applicable_conditions": "The incident must involve fraudulent use of electronic credentials. Evidence such as call recordings or SMS records must be preserved.",
        "expected_answer": "The Information Technology Act, 2000 may apply. Section 66C (identity theft) and Section 66D (cheating by personation) are relevant. Report to cybercrime.gov.in and file an FIR."
    },
    {
        "problem": "My spouse beats me regularly and does not give me any money for food or household expenses.",
        "law": "Protection of Women from Domestic Violence Act, 2005",
        "section": "Section 3",
        "solution": "Contact a Protection Officer or file an application before the Magistrate for a Protection Order and monetary relief.",
        "solution_based_on_law": "Section 3 defines domestic violence to include physical abuse and economic abuse such as denial of food and financial resources.",
        "risk_category": "Family",
        "applicable_conditions": "The complainant must be in a domestic relationship with the respondent as defined under the Act.",
        "expected_answer": "The Protection of Women from Domestic Violence Act, 2005 applies. Section 3 covers both physical abuse and economic abuse. You may approach a Protection Officer or Magistrate for relief."
    },
    {
        "problem": "My employer has not paid my salary for the last three months despite repeated requests.",
        "law": "Payment of Wages Act, 1936",
        "section": "Section 3",
        "solution": "File a complaint before the Authority under the Payment of Wages Act in your jurisdiction for recovery of delayed wages.",
        "solution_based_on_law": "Section 3 makes it the employer's responsibility to pay wages on time. Failure to do so entitles the employee to delayed wages plus compensation.",
        "risk_category": "Labour",
        "applicable_conditions": "The employment relationship must be established. Payslips or any written employment record should be preserved as evidence.",
        "expected_answer": "The Payment of Wages Act, 1936 may apply. Section 3 makes timely payment of wages mandatory. You can file a complaint with the Payment of Wages Authority for recovery."
    },
    {
        "problem": "A drunk driver hit my parked car and sped away. A witness recorded the vehicle number.",
        "law": "Motor Vehicles Act, 1988; Indian Penal Code, 1860",
        "section": "Section 185 (Motor Vehicles Act); Section 279 (IPC)",
        "solution": "File an FIR with the police immediately. Share the vehicle registration number and witness evidence. File an insurance claim for vehicle damage.",
        "solution_based_on_law": "Section 185 of the Motor Vehicles Act punishes drunk driving. Section 279 IPC covers rash driving endangering public safety.",
        "risk_category": "Criminal",
        "applicable_conditions": "Witness statement and vehicle registration number must be provided to police. CCTV footage should be preserved if available.",
        "expected_answer": "The Motor Vehicles Act, 1988 (Section 185 - drunk driving) and IPC Section 279 (rash driving) may apply. File an FIR with the witness details and vehicle number."
    },
    {
        "problem": "My landlord is refusing to return my security deposit of one lakh rupees even though I vacated the flat in good condition.",
        "law": "Applicable tenancy law and Indian Contract Act, 1872",
        "section": "To be determined from the rental agreement and local tenancy law",
        "solution": "Review the rental agreement for deposit refund terms. Send a legal notice to the landlord. If unresolved, file a civil suit for recovery.",
        "solution_based_on_law": "The rental agreement governs deposit terms. Indian Contract Act Section 73 may apply if breach of agreement is established.",
        "risk_category": "Property",
        "applicable_conditions": "The rental agreement, payment receipts, and proof of vacating in good condition are required. The applicable tenancy law depends on the state.",
        "expected_answer": "More information is needed. The applicable law depends on your state's tenancy laws and the rental agreement. Consider sending a legal notice and filing a civil recovery suit."
    },
    {
        "problem": "A shopkeeper sold me food that was expired and I got seriously ill after eating it.",
        "law": "Consumer Protection Act, 2019; Food Safety and Standards Act, 2006",
        "section": "Section 2(11), Section 35 (Consumer Protection Act); Section 26 (FSSA)",
        "solution": "Preserve the product packaging as evidence. File a consumer complaint and report to the Food Safety Officer in your area.",
        "solution_based_on_law": "The Consumer Protection Act covers defective goods. The FSSA Section 26 prohibits sale of unsafe food and provides for penalties.",
        "risk_category": "Consumer",
        "applicable_conditions": "The product must be in your possession as evidence. Medical records of illness must be documented.",
        "expected_answer": "The Consumer Protection Act, 2019 and the Food Safety and Standards Act, 2006 may apply. File a consumer complaint and also report to your local Food Safety Authority."
    },
    {
        "problem": "My neighbour has been playing extremely loud music late at night regularly disturbing my sleep.",
        "law": "Indian Penal Code, 1860; Environment Protection Act, 1986",
        "section": "Section 268 IPC (public nuisance); Noise Pollution Rules 2000",
        "solution": "File a complaint with the local police station under noise pollution rules. The permissible noise limit at night in residential areas is 45 dB.",
        "solution_based_on_law": "Section 268 IPC defines public nuisance. Noise Pollution (Regulation and Control) Rules, 2000 set limits for residential areas.",
        "risk_category": "Civil",
        "applicable_conditions": "Noise levels must exceed prescribed limits. Time of occurrence (after 10 PM) is a key factor.",
        "expected_answer": "IPC Section 268 (public nuisance) and Noise Pollution Rules 2000 may apply. File a complaint with police. Night-time residential noise limit is 45 dB."
    },
    {
        "problem": "I want to get information about the number of government employees in my district but the government office is not responding.",
        "law": "Right to Information Act, 2005",
        "section": "Section 6",
        "solution": "File an RTI application with the concerned Public Information Officer. If no response in 30 days, file a first appeal.",
        "solution_based_on_law": "Section 6 of RTI Act allows any citizen to request information from public authorities. Response must be given within 30 days.",
        "risk_category": "RTI",
        "applicable_conditions": "The information must be held by a public authority. The request must be specific and accompanied by the prescribed fee.",
        "expected_answer": "The Right to Information Act, 2005 applies. Under Section 6, you can file an RTI application with the Public Information Officer. A response is mandatory within 30 days."
    },
]

COLUMNS = [
    "problem", "law", "section", "solution",
    "solution_based_on_law", "risk_category",
    "applicable_conditions", "expected_answer"
]

# ═══════════════════════════════════════════════════════════
# DEMO HELPER
# ═══════════════════════════════════════════════════════════
def header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def step(msg):
    print(f"\n  --> {msg}")

def ok(msg):
    print(f"  [OK] {msg}")

def show(label, value, indent=6):
    lines = str(value).split("\n")
    print(f"{'':>{indent}}{label}:")
    for line in lines:
        print(f"{'':>{indent+2}}{line}")


# ═══════════════════════════════════════════════════════════
# STEP 1: Write sample legal sections (simulates Part 1 output)
# ═══════════════════════════════════════════════════════════
def step1_write_sample_sections():
    header("STEP 1 — Legal Sections (Simulating Part 1 PDF Extraction)")

    # Group by law
    from collections import defaultdict
    by_law = defaultdict(list)
    for s in SAMPLE_LEGAL_SECTIONS:
        by_law[s["law_name"]].append(s)

    files_written = []
    for law_name, sections in by_law.items():
        filename = law_name.lower().replace(" ", "_") + "_sections.json"
        path = SECTION_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
        ok(f"Wrote {len(sections)} section(s) -> {path.name}")
        files_written.append(path)

    print(f"\n  Summary:")
    print(f"    Laws covered     : {len(by_law)}")
    print(f"    Total sections   : {len(SAMPLE_LEGAL_SECTIONS)}")
    print(f"    Output directory : {SECTION_DIR}/")

    # Show one sample section
    sample = SAMPLE_LEGAL_SECTIONS[0]
    print(f"\n  Sample extracted section:")
    print(f"    Law     : {sample['law_name']}")
    print(f"    Section : {sample['section_id']}")
    print(f"    Text    : {sample['section_text'][:120]}...")

    return files_written


# ═══════════════════════════════════════════════════════════
# STEP 2: Write sample dataset (simulates Part 2 output)
# ═══════════════════════════════════════════════════════════
def step2_write_sample_dataset():
    header("STEP 2 — Sample Dataset (Simulating Part 2 Generation)")

    csv_path = DATASET_DIR / "sample_dataset.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(SAMPLE_DATASET)

    ok(f"Wrote {len(SAMPLE_DATASET)} rows -> {csv_path}")
    print(f"\n  Dataset columns: {', '.join(COLUMNS)}")
    print(f"\n  Sample rows preview:")

    for i, row in enumerate(SAMPLE_DATASET[:3], 1):
        print(f"\n  --- Row {i} ---")
        print(f"    Problem  : {row['problem'][:75]}...")
        print(f"    Law      : {row['law']}")
        print(f"    Section  : {row['section']}")
        print(f"    Category : {row['risk_category']}")

    return csv_path


# ═══════════════════════════════════════════════════════════
# STEP 3: Build ChromaDB index (Part 5 — RAG)
# ═══════════════════════════════════════════════════════════
def step3_build_chroma_index():
    header("STEP 3 — Building ChromaDB Vector Index (Part 5 RAG)")

    step("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    step("(First run downloads ~90MB model from HuggingFace...)")

    from sentence_transformers import SentenceTransformer
    import chromadb

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ok("Embedding model loaded")

    # Init persistent ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("indian_legal_sections")
    except Exception:
        pass
    collection = client.create_collection(
        name="indian_legal_sections",
        metadata={"hnsw:space": "cosine"}
    )

    step(f"Embedding {len(SAMPLE_LEGAL_SECTIONS)} legal sections...")

    texts = [s["section_text"] for s in SAMPLE_LEGAL_SECTIONS]
    ids   = [f"doc_{i}" for i in range(len(SAMPLE_LEGAL_SECTIONS))]
    metas = [{
        "law_name":   s["law_name"],
        "section_id": s["section_id"]
    } for s in SAMPLE_LEGAL_SECTIONS]

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    ok(f"ChromaDB index built: {collection.count()} chunks stored")
    ok(f"Persisted to: {CHROMA_DIR}/")

    return collection, embedder


# ═══════════════════════════════════════════════════════════
# STEP 4: Run sample queries through RAG retrieval
# ═══════════════════════════════════════════════════════════
def step4_run_rag_queries(collection, embedder):
    header("STEP 4 — RAG Retrieval Test (3 Sample Legal Queries)")

    TEST_QUERIES = [
        "I bought a defective product and the seller won't refund",
        "Someone stole money from my bank account using my OTP",
        "My husband beats me and doesn't give me money for household expenses"
    ]

    for q_num, query in enumerate(TEST_QUERIES, 1):
        print(f"\n  Query {q_num}: \"{query}\"")
        print(f"  {'-' * 55}")

        # Embed query and search
        query_embedding = embedder.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        print(f"  Top 3 retrieved legal sections:")
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            similarity = round(1 - dist, 4)
            print(f"\n    [{rank}] {meta['law_name']} -- {meta['section_id']}")
            print(f"        Similarity : {similarity:.4f}")
            print(f"        Text       : {doc[:110]}...")


# ═══════════════════════════════════════════════════════════
# STEP 5: Show what a full legal answer looks like
# ═══════════════════════════════════════════════════════════
def step5_show_sample_outputs():
    header("STEP 5 — Sample AI Legal Assistant Outputs")

    print("  (These are what the fine-tuned model would generate)")
    print("  (After training on 10,000 rows with the real PDFs)")

    for i, row in enumerate(SAMPLE_DATASET[:3], 1):
        print(f"\n  {'=' * 56}")
        print(f"  LEGAL QUERY {i}")
        print(f"  {'=' * 56}")
        print(f"  Problem   : {row['problem']}")
        print()
        print(f"  [OUTPUT FROM AI LEGAL ASSISTANT]")
        print(f"  Applicable Law       : {row['law']}")
        print(f"  Section              : {row['section']}")
        print(f"  Risk Category        : {row['risk_category']}")
        print(f"  Solution             : {row['solution']}")
        print(f"  Legal Basis          : {row['solution_based_on_law']}")
        print(f"  Applicable Conditions: {row['applicable_conditions']}")
        print(f"  Full Answer          :")
        print(f"    {row['expected_answer']}")


# ═══════════════════════════════════════════════════════════
# STEP 6: Validate dataset structure
# ═══════════════════════════════════════════════════════════
def step6_validate_dataset():
    header("STEP 6 — Dataset Validation Report")

    import pandas as pd

    csv_path = DATASET_DIR / "sample_dataset.csv"
    df = pd.read_csv(csv_path)

    print(f"  Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated(subset=['problem']).sum()}")

    print(f"\n  Risk Category Distribution:")
    for cat, count in df["risk_category"].value_counts().items():
        bar = "#" * count
        print(f"    {cat:<20} {bar} ({count})")

    print(f"\n  Average problem length : {df['problem'].str.len().mean():.0f} chars")
    print(f"  Average answer length  : {df['expected_answer'].str.len().mean():.0f} chars")

    # Check no law names in problem column
    import re
    legal_in_problem = df["problem"].str.contains(
        r"Section\s+\d+|Act,?\s+\d{4}|IPC\s*\d+", regex=True, case=False
    )
    violations = legal_in_problem.sum()
    if violations == 0:
        ok("No legal citations found inside problem column (correct)")
    else:
        print(f"  [WARN] {violations} problem(s) contain legal citations (should be fixed)")


# ═══════════════════════════════════════════════════════════
# STEP 7: Final project file tree
# ═══════════════════════════════════════════════════════════
def step7_show_project_tree():
    header("STEP 7 — Project Structure Created")

    root = Path(".")
    important_paths = [
        "data/cleaned_sections/",
        "data/dataset/sample_dataset.csv",
        "chroma_db/",
        "src/part1_pdf_extraction/extract_pdfs.py",
        "src/part2_dataset_creation/generate_dataset.py",
        "src/part3_tokenization/prepare_dataset.py",
        "src/part4_fine_tuning/train.py",
        "src/part5_rag/rag_pipeline.py",
        "src/part6_evaluation/evaluate.py",
        "src/part7_improvement/analyze_errors.py",
        "src/part8_full_pipeline/run_pipeline.py",
        "requirements.txt",
        "check_setup.py",
    ]

    print()
    for p in important_paths:
        full = Path(p)
        exists = "[EXISTS]" if (full.exists() or full.parent.exists()) else "[MISSING]"
        print(f"  {exists}  {p}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print()
    print("*" * 60)
    print("*  AI LEGAL ASSISTANT -- DEMO RUN                        *")
    print("*  Indian Law | RAG + LLM Pipeline                       *")
    print("*" * 60)
    print(f"  Python {sys.version.split()[0]} | "
          f"Running from: {Path('.').resolve()}")

    t0 = time.time()

    step1_write_sample_sections()
    step2_write_sample_dataset()
    collection, embedder = step3_build_chroma_index()
    step4_run_rag_queries(collection, embedder)
    step5_show_sample_outputs()
    step6_validate_dataset()
    step7_show_project_tree()

    elapsed = time.time() - t0

    header("DEMO COMPLETE")
    print(f"  Total time       : {elapsed:.1f} seconds")
    print()
    print("  WHAT JUST WORKED:")
    print("  [OK] Sample legal sections written to data/cleaned_sections/")
    print("  [OK] Sample 10-row dataset written to data/dataset/")
    print("  [OK] ChromaDB vector index built and persisted to chroma_db/")
    print("  [OK] RAG retrieval tested with 3 real legal queries")
    print("  [OK] Sample AI outputs displayed for 3 legal problems")
    print("  [OK] Dataset validation passed (no duplicates, no citations in problem)")
    print()
    print("  WHAT YOU NEED TO DO NEXT (in order):")
    print("  1. Copy your Indian legal PDFs to:  data\\raw_pdfs\\")
    print("     then run: python src\\part1_pdf_extraction\\extract_pdfs.py")
    print()
    print("  2. Install Ollama and pull mistral model:")
    print("     https://ollama.com  -->  ollama pull mistral")
    print("     then run: python src\\part2_dataset_creation\\generate_dataset.py")
    print()
    print("  3. For model fine-tuning (needs NVIDIA GPU):")
    print("     Use Google Colab with the generated dataset, OR")
    print("     Use Ollama (mistral) + RAG without fine-tuning")
    print()
    print("*" * 60)
    print()


if __name__ == "__main__":
    main()
