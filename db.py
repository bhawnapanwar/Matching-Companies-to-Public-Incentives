"""
db.py — Database setup and connection management.
Creates PostgreSQL tables and loads CSV data.
"""

import os
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Return a new PostgreSQL connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "homodeus"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def setup_database(
    companies_csv: str,
    incentives_csv: str,
    use_cases_csv: str,
    force_reload: bool = False,
):
    """
    Create tables and load all CSVs into PostgreSQL.
    If force_reload=True, drops and recreates all tables.
    """
    conn = get_connection()
    cur = conn.cursor()

    if force_reload:
        print("Dropping existing tables...")
        cur.execute("DROP TABLE IF EXISTS matches CASCADE;")
        cur.execute("DROP TABLE IF EXISTS companies CASCADE;")
        cur.execute("DROP TABLE IF EXISTS incentives CASCADE;")
        cur.execute("DROP TABLE IF EXISTS use_cases CASCADE;")
        conn.commit()

    # --- Create tables ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            company_name TEXT NOT NULL,
            cae_primary_label TEXT,
            trade_description_native TEXT,
            website TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS incentives (
            incentive_id TEXT PRIMARY KEY,
            incentive_name TEXT NOT NULL,
            managing_entity TEXT,
            program TEXT,
            type TEXT,
            max_funding_eur TEXT,
            funding_rate_pct TEXT,
            eligible_company_types TEXT,
            eligible_sectors TEXT,
            eligible_activities TEXT,
            deadline TEXT,
            description TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS use_cases (
            use_case_id TEXT PRIMARY KEY,
            use_case_name TEXT NOT NULL,
            description TEXT,
            target_industries TEXT,
            typical_roi TEXT,
            typical_timeline TEXT,
            product_platform TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id SERIAL PRIMARY KEY,
            incentive_id TEXT REFERENCES incentives(incentive_id),
            company_id INTEGER REFERENCES companies(id),
            company_name TEXT,
            score NUMERIC(4,2),
            justification TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()

    # --- Load data only if tables are empty ---
    cur.execute("SELECT COUNT(*) FROM companies;")
    company_count = cur.fetchone()[0]

    if company_count == 0:
        print(f"Loading companies from {companies_csv}...")
        df = pd.read_csv(companies_csv, dtype=str).fillna("")
        rows = [
            (
                row["company_name"],
                row["cae_primary_label"],
                row["trade_description_native"],
                row.get("website", ""),
            )
            for _, row in df.iterrows()
        ]
        execute_values(
            cur,
            "INSERT INTO companies (company_name, cae_primary_label, trade_description_native, website) VALUES %s",
            rows,
            page_size=1000,
        )
        conn.commit()
        print(f"  Loaded {len(rows):,} companies.")
    else:
        print(f"  Companies already loaded ({company_count:,} rows). Skipping.")

    cur.execute("SELECT COUNT(*) FROM incentives;")
    if cur.fetchone()[0] == 0:
        print(f"Loading incentives from {incentives_csv}...")
        df = pd.read_csv(incentives_csv, dtype=str).fillna("")
        rows = [tuple(row) for _, row in df.iterrows()]
        execute_values(
            cur,
            """INSERT INTO incentives
               (incentive_id, incentive_name, managing_entity, program, type,
                max_funding_eur, funding_rate_pct, eligible_company_types,
                eligible_sectors, eligible_activities, deadline, description)
               VALUES %s""",
            rows,
        )
        conn.commit()
        print(f"  Loaded {len(rows)} incentives.")

    cur.execute("SELECT COUNT(*) FROM use_cases;")
    if cur.fetchone()[0] == 0:
        print(f"Loading use cases from {use_cases_csv}...")
        df = pd.read_csv(use_cases_csv, dtype=str).fillna("")
        rows = [tuple(row) for _, row in df.iterrows()]
        execute_values(
            cur,
            """INSERT INTO use_cases
               (use_case_id, use_case_name, description, target_industries,
                typical_roi, typical_timeline, product_platform)
               VALUES %s""",
            rows,
        )
        conn.commit()
        print(f"  Loaded {len(rows)} use cases.")

    cur.close()
    conn.close()
    print("Database setup complete.")