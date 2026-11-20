"""
=============================================================
PART 5 — RAG System with ChromaDB
=============================================================
File: src/part5_rag/rag_pipeline.py

This script:
  1. Loads the cleaned legal sections from Part 1
  2. Splits them into smaller chunks for embedding
  3. Generates sentence embeddings using a legal-domain model
  4. Stores embeddings + metadata in ChromaDB (persistent on disk)
  5. Provides a retrieval function to find relevant legal sections
  6. Passes retrieved context to the fine-tuned LLM
  7. Generates the final grounded legal answer

WHY RAG REDUCES HALLUCINATIONS:
  Without RAG: The LLM relies entirely on its training memory.
  It can "hallucinate" section numbers or laws that don't exist.

  With RAG:
  1. The user's problem is used to search ChromaDB
  2. The actual legal text is retrieved from the PDFs
  3. This real text is passed to the LLM as context
  4. The LLM then answers based on the actual retrieved law —
     it cannot fabricate a section that isn't in the retrieved text.

HOW TO RUN:
    python src/part5_rag/rag_pipeline.py --query "I bought a defective phone"

DEPENDENCIES:
    pip install chromadb sentence-transformers
=============================================================
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CLEANED_SECTION_DIR = Path("data/cleaned_sections")
CHROMA_DB_DIR       = Path("chroma_db")

# Embedding model: legal-bert is fine-tuned on legal text
# Options:
#   "sentence-transformers/all-MiniLM-L6-v2"       ← general purpose, fast
#   "nlpaueb/legal-bert-base-uncased"               ← legal domain, better quality
#   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  ← multilingual
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME     = "indian_legal_sections"
CHUNK_SIZE          = 512     # characters per chunk (for long sections)
CHUNK_OVERLAP       = 50      # overlap between chunks to preserve context
TOP_K               = 5       # number of relevant chunks to retrieve per query


# ─────────────────────────────────────────────
# STEP 1: CHUNKING
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.

    WHY OVERLAP?
    Legal sections can reference other sections. Overlapping ensures
    that cross-references at chunk boundaries are not split.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward by (chunk_size - overlap)

    return chunks


def prepare_documents_for_chroma(sections_dir: Path) -> list[dict]:
    """
    Load all legal sections, chunk them, and prepare documents for ChromaDB.
    Each document = one chunk with metadata (law_name, section_id, chunk_index).
    """
    documents = []
    doc_id = 0

    for json_file in sections_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            sections = json.load(f)

        for section in sections:
            law_name   = section.get("law_name", "Unknown")
            section_id = section.get("section_id", "UNKNOWN")
            full_text  = section.get("section_text", "")

            if not full_text.strip():
                continue

            chunks = chunk_text(full_text)
            for chunk_idx, chunk in enumerate(chunks):
                documents.append({
                    "id":       f"doc_{doc_id}",
                    "text":     chunk,
                    "metadata": {
                        "law_name":    law_name,
                        "section_id":  section_id,
                        "chunk_index": chunk_idx,
                        "source_file": json_file.name
                    }
                })
                doc_id += 1

    log.info(f"Prepared {len(documents)} chunks from {sections_dir}")
    return documents


# ─────────────────────────────────────────────
# STEP 2–3: GENERATE EMBEDDINGS & STORE
# ─────────────────────────────────────────────
def build_chroma_index(documents: list[dict], embedding_model_name: str):
    """
    Generate embeddings for all document chunks and store in ChromaDB.
    ChromaDB persists to disk — you only need to build this ONCE.
    """
    log.info(f"Loading embedding model: {embedding_model_name}")
    embedder = SentenceTransformer(embedding_model_name)

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text embeddings
    )

    # Batch embed and store (ChromaDB prefers batches of ~100)
    BATCH = 100
    for i in tqdm(range(0, len(documents), BATCH), desc="Storing in ChromaDB"):
        batch = documents[i:i + BATCH]
        texts    = [d["text"]     for d in batch]
        ids      = [d["id"]       for d in batch]
        metas    = [d["metadata"] for d in batch]

        # Generate embeddings
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas
        )

    log.info(f"✓ ChromaDB index built. {len(documents)} chunks stored in '{COLLECTION_NAME}'")
    return client, collection, embedder


# ─────────────────────────────────────────────
# STEP 4: RETRIEVAL
# ─────────────────────────────────────────────
def load_chroma_collection(embedding_model_name: str):
    """Load an existing ChromaDB collection (after index is built)."""
    embedder = SentenceTransformer(embedding_model_name)
    client   = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    log.info(f"Loaded ChromaDB collection '{COLLECTION_NAME}' "
             f"({collection.count()} chunks)")
    return client, collection, embedder


def retrieve_relevant_sections(
    query: str,
    collection,
    embedder,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Retrieve the top-k most relevant legal sections for a user query.
    
    HOW IT WORKS:
    1. The user query is converted to an embedding vector
    2. ChromaDB performs cosine similarity search against all stored embeddings
    3. The top-k most similar legal chunks are returned
    4. These chunks contain actual legal text from the official PDFs
    """
    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text":        doc,
            "law_name":    meta.get("law_name", ""),
            "section_id":  meta.get("section_id", ""),
            "similarity":  round(1 - dist, 4)  # convert distance to similarity
        })

    return retrieved


# ─────────────────────────────────────────────
# STEP 5–6: RAG ANSWER GENERATION
# ─────────────────────────────────────────────
def build_rag_prompt(problem: str, retrieved_sections: list[dict]) -> str:
    """
    Build a prompt that includes the retrieved legal context + user problem.
    The LLM will use this real legal text to generate its answer,
    preventing fabricated citations.
    """
    context_parts = []
    for i, sec in enumerate(retrieved_sections, 1):
        context_parts.append(
            f"[Source {i}] {sec['law_name']} — {sec['section_id']}\n"
            f"{sec['text']}"
        )
    legal_context = "\n\n".join(context_parts)

    prompt = f"""You are an expert Indian legal assistant.
Use ONLY the following retrieved legal sections to answer the user's problem.
Do NOT use any legal knowledge outside the provided sources.
If the retrieved sections do not cover the problem, say so clearly.

RETRIEVED LEGAL SOURCES:
{legal_context}

USER PROBLEM:
{problem}

Based ONLY on the retrieved legal sources above, provide:
1. Applicable Law and Section
2. Practical Solution
3. Legal Basis
4. Risk Category
5. Applicable Conditions
6. Final Answer

Output as a valid JSON object with keys:
law, section, solution, solution_based_on_law, risk_category, applicable_conditions, expected_answer
"""
    return prompt


def generate_rag_answer(
    problem: str,
    collection,
    embedder,
    llm_model,
    llm_tokenizer,
    top_k: int = TOP_K
) -> dict:
    """
    Full RAG pipeline: retrieve → augment prompt → generate answer.
    
    This is the COMBINED system:
    RAG (ChromaDB retrieval) + Fine-tuned LLM = grounded, accurate answers.
    """
    import torch
    import json

    # Step 4: Retrieve relevant legal sections
    retrieved = retrieve_relevant_sections(problem, collection, embedder, top_k)

    if not retrieved:
        log.warning("No relevant sections retrieved from ChromaDB.")

    # Step 5: Build the RAG-augmented prompt
    rag_prompt = build_rag_prompt(problem, retrieved)

    # Step 6: Generate answer using the fine-tuned LLM
    prompt_text = f"[INST] {rag_prompt} [/INST]\n"
    inputs = llm_tokenizer(prompt_text, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1
        )

    output_text = llm_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Parse JSON response
    try:
        answer = json.loads(output_text.strip())
    except json.JSONDecodeError:
        answer = {"raw_output": output_text, "parse_error": True}

    # Attach the sources used
    answer["retrieved_sources"] = [
        {"law": s["law_name"], "section": s["section_id"], "similarity": s["similarity"]}
        for s in retrieved
    ]

    return answer


# ─────────────────────────────────────────────
# MAIN — Build index or run a test query
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true",
                        help="Build ChromaDB index from cleaned sections")
    parser.add_argument("--query", type=str, default=None,
                        help="Test a retrieval query")
    args = parser.parse_args()

    if args.build_index or not Path(CHROMA_DB_DIR).exists():
        log.info("Building ChromaDB index from legal sections…")
        documents = prepare_documents_for_chroma(CLEANED_SECTION_DIR)
        if not documents:
            log.error("No legal sections found. Run Part 1 first.")
            return
        client, collection, embedder = build_chroma_index(documents, EMBEDDING_MODEL)
    else:
        client, collection, embedder = load_chroma_collection(EMBEDDING_MODEL)

    if args.query:
        log.info(f"\nQuery: '{args.query}'")
        results = retrieve_relevant_sections(args.query, collection, embedder)
        print(f"\n{'='*55}")
        print(f"Top {len(results)} retrieved sections:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['law_name']} — {r['section_id']}")
            print(f"    Similarity: {r['similarity']}")
            print(f"    Text: {r['text'][:200]}…")
    else:
        log.info("Index ready. Use --query 'your problem' to test retrieval.")
        log.info("To run the full RAG pipeline, import generate_rag_answer() in your app.")


if __name__ == "__main__":
    main()
