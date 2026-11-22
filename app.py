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
        "status":           "ok" if embedder is not None and collection is not None else "degraded",
        "sections_indexed": count,
        "model_mode":       "Neural RAG + Hybrid Legal Engine",
        "ready":            embedder is not None and collection is not None
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


# Prepare directories and initialize the backend when imported by Gunicorn/Render.
for d in [CHROMA_DIR, SECTION_DIR, DATASET_DIR, FRONTEND_DIR / "static" / "css", FRONTEND_DIR / "static" / "js"]:
    d.mkdir(parents=True, exist_ok=True)

try:
    load_resources()
    log.info("Nyaya AI backend resources initialized.")
except Exception:
    log.exception("Backend resource initialization failed. The app will serve status/error responses where possible.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    log.info("Starting Nyaya AI on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
