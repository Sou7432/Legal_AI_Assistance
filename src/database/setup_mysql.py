"""
MySQL database setup for the AI Legal Assistant application.
Run this once to create all required tables.

Usage:
    python src/database/setup_mysql.py

Requires MySQL running locally and credentials in .env:
    DB_HOST=localhost
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=yourpassword
    DB_NAME=legal_assistant
"""

import os
import logging
from dotenv import load_dotenv
import mysql.connector

load_dotenv()
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "legal_assistant")
}

CREATE_DB_SQL = f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

CREATE_TABLES_SQL = [
    # Store user queries and AI responses
    """
    CREATE TABLE IF NOT EXISTS legal_queries (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        session_id      VARCHAR(64),
        problem         TEXT NOT NULL,
        law             VARCHAR(500),
        section         VARCHAR(500),
        solution        TEXT,
        solution_based_on_law TEXT,
        risk_category   VARCHAR(100),
        applicable_conditions TEXT,
        expected_answer TEXT,
        rag_used        TINYINT(1) DEFAULT 1,
        model_version   VARCHAR(100),
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        response_time_ms INT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    # Store the extracted legal sections index
    """
    CREATE TABLE IF NOT EXISTS legal_sections (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        law_name        VARCHAR(500) NOT NULL,
        section_id      VARCHAR(200),
        section_title   VARCHAR(1000),
        section_text    LONGTEXT,
        source_pdf      VARCHAR(255),
        char_count      INT,
        indexed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_law (law_name(255)),
        INDEX idx_section (section_id(100))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    # Store model evaluation results
    """
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        model_path      VARCHAR(500),
        rag_enabled     TINYINT(1) DEFAULT 0,
        test_set_size   INT,
        law_exact_match FLOAT,
        section_f1      FLOAT,
        bert_score_avg  FLOAT,
        hallucination_pct FLOAT,
        rag_recall_at_5 FLOAT,
        full_report_json LONGTEXT,
        run_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    # Store user feedback on AI answers
    """
    CREATE TABLE IF NOT EXISTS user_feedback (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        query_id        INT,
        helpful         TINYINT(1),
        correct_law     TINYINT(1),
        comment         TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (query_id) REFERENCES legal_queries(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
]


def setup():
    # First connect without database to create it
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()
    cursor.execute(CREATE_DB_SQL)
    conn.database = DB_CONFIG["database"]

    for sql in CREATE_TABLES_SQL:
        cursor.execute(sql)
        conn.commit()

    cursor.close()
    conn.close()
    print("✓ MySQL database and tables created successfully.")


if __name__ == "__main__":
    setup()
