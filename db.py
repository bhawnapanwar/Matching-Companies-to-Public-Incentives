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

    
    # Companies: insert only new ones (by name)
    print(f"Syncing companies from {companies_csv}...")
    df_companies = pd.read_csv(companies_csv, dtype=str).fillna("")
    cur.execute("SELECT company_name FROM companies;")
    existing_companies = {row[0] for row in cur.fetchall()}
    new_companies = [
        (
            row["company_name"],
            row["cae_primary_label"],
            row["trade_description_native"],
            row.get("website", ""),
        )
        for _, row in df_companies.iterrows()
        if row["company_name"] not in existing_companies
    ]
    if new_companies:
        execute_values(
            cur,
            "INSERT INTO companies (company_name, cae_primary_label, trade_description_native, website) VALUES %s",
            new_companies,
            page_size=1000,
        )
        conn.commit()
        print(f"  Added {len(new_companies):,} new companies ({len(existing_companies):,} already existed).")
    else:
        print(f"  Companies up to date ({len(existing_companies):,} rows).")

    # Incentives: upsert by incentive_id
    print(f"Syncing incentives from {incentives_csv}...")
    df_incentives = pd.read_csv(incentives_csv, dtype=str).fillna("")
    cur.execute("SELECT incentive_id FROM incentives;")
    existing_incentives = {row[0] for row in cur.fetchall()}
    new_incentives = [
        tuple(row) for _, row in df_incentives.iterrows()
        if row["incentive_id"] not in existing_incentives
    ]
    if new_incentives:
        execute_values(
            cur,
            """INSERT INTO incentives
               (incentive_id, incentive_name, managing_entity, program, type,
                max_funding_eur, funding_rate_pct, eligible_company_types,
                eligible_sectors, eligible_activities, deadline, description)
               VALUES %s""",
            new_incentives,
        )
        conn.commit()
        print(f"  Added {len(new_incentives)} new incentive(s).")
    else:
        print(f"  Incentives up to date ({len(existing_incentives)} rows).")

    # Use cases: upsert by use_case_id
    print(f"Syncing use cases from {use_cases_csv}...")
    df_use_cases = pd.read_csv(use_cases_csv, dtype=str).fillna("")
    cur.execute("SELECT use_case_id FROM use_cases;")
    existing_use_cases = {row[0] for row in cur.fetchall()}
    new_use_cases = [
        tuple(row) for _, row in df_use_cases.iterrows()
        if row["use_case_id"] not in existing_use_cases
    ]
    if new_use_cases:
        execute_values(
            cur,
            """INSERT INTO use_cases
               (use_case_id, use_case_name, description, target_industries,
                typical_roi, typical_timeline, product_platform)
               VALUES %s""",
            new_use_cases,
        )
        conn.commit()
        print(f"  Added {len(new_use_cases)} new use case(s).")
    else:
        print(f"  Use cases up to date ({len(existing_use_cases)} rows.")

    cur.close()
    conn.close()
    print("Database setup complete.")