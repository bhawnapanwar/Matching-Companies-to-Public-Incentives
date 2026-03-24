"""
db.py — Database setup and connection management.
Creates PostgreSQL tables and loads CSV data.
"""

import os
import time
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_connection():
    """Return a new PostgreSQL connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "homodeus"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )

def embed_and_save_companies():
    """
    Fetches all companies, creates a strict, short string for each to minimize costs,
    embeds them in batches, and saves the vectors to disk.
    """
    print("Generating and saving company embeddings (this may take a few minutes)...")
    conn = get_connection()
    cur = conn.cursor()

    # Fetch all companies
    cur.execute("SELECT id, cae_primary_label, trade_description_native FROM companies ORDER BY id;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("No companies found to embed.")
        return

    company_ids = []
    texts_to_embed = []

    for row in rows:
        comp_id = row[0]
        cae = row[1] or ""
        desc = row[2] or ""
        
        # STRICT TRUNCATION: This keeps it under ~30 tokens per company.
        # 250k companies * 30 tokens = 7.5M tokens ($0.15 total cost)
        text = f"Sector: {cae} | Activity: {desc[:100]}"
        
        company_ids.append(comp_id)
        texts_to_embed.append(text)

    # Batch embed using OpenAI (Limit is 2048 per batch)
    all_embeddings = []
    BATCH_SIZE = 2000 
    MAX_RETRIES = 3

    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts_to_embed) // BATCH_SIZE) + 1
        print(f"  Embedding batch {batch_num} / {total_batches}...")
        
        # Retry logic for OpenAI Server Errors
        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break  # Success! Break out of the retry loop and move to the next batch
                
            except Exception as e:
                print(f"    [!] Error on batch {batch_num}: {e}")
                if attempt < MAX_RETRIES - 1:
                    sleep_time = (attempt + 1) * 2
                    print(f"    Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("    Max retries reached. Crashing gracefully.")
                    raise e

    # Convert to fast numpy arrays and save
    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    ids_array = np.array(company_ids, dtype=np.int32)

    np.save('company_embeddings.npy', embeddings_matrix)
    np.save('company_ids.npy', ids_array)

    print(f"Successfully saved {len(company_ids)} embeddings to company_embeddings.npy")

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

    # Add indexes for performance (safe to run multiple times)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_companies_cae ON companies(cae_primary_label);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_companies_trade ON companies USING gin(to_tsvector('simple', trade_description_native));")
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

    embeddings_exist = os.path.exists('company_embeddings.npy') and os.path.exists('company_ids.npy')

    # Run if forced, if there's new data, OR if the files are missing
    if force_reload or new_companies or not embeddings_exist: 
        embed_and_save_companies()
    else:
        print("Company embeddings already exist on disk. Skipping generation.")