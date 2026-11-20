# Shared constants used across multiple parts of the pipeline
INSTRUCTION = (
    "You are an expert Indian legal assistant. "
    "Given the user's legal problem described in everyday language, "
    "identify the applicable Indian law, relevant section, practical solution, "
    "legal basis, risk category, applicable conditions, and provide a comprehensive answer. "
    "Output your answer as a valid JSON object with these exact keys: "
    "law, section, solution, solution_based_on_law, risk_category, "
    "applicable_conditions, expected_answer."
)

OUTPUT_COLUMNS = [
    "law",
    "section",
    "solution",
    "solution_based_on_law",
    "risk_category",
    "applicable_conditions",
    "expected_answer"
]

ALL_COLUMNS = ["problem"] + OUTPUT_COLUMNS
