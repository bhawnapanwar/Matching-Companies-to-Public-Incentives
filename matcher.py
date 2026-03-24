"""
matcher.py — Core matching logic with RAG-style semantic search.

Strategy (HyDE + Pure Vector Search):
  1. Load pre-computed company embeddings (fast numpy matrix).
  2. For each incentive, use an LLM to generate a hypothetical "Ideal Company Profile" (HyDE).
  3. Embed that ONE ideal profile using OpenAI text-embedding-3-small.
  4. Perform pure vector matrix multiplication against all 250k companies instantly.
  5. Fetch the top 80 companies from the DB.
  6. LLM re-ranks the shortlist and picks the best 5 with scores + justifications.
"""

import json
import os
import time
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CANDIDATE_LIMIT = 80
MAX_WORKERS = 3
EMBEDDING_MODEL = "text-embedding-3-small"

# Thread-safe token tracking
_token_lock = threading.Lock()
_total_tokens = {"prompt": 0, "completion": 0, "embedding": 0}

# ── 1. HyDE Alignment & Embedding ────────────────────────────────────────────

def _generate_ideal_profile(incentive: dict) -> str:
    """Translates an incentive into a hypothetical perfect-match company."""
    prompt = f"""
    Write a 2-sentence description of a hypothetical company that perfectly matches this incentive.
    Format exactly like: "Sector: [Sector]. Activity: We [Action]."
    
    Incentive: {incentive.get('incentive_name')}
    Sectors: {incentive.get('eligible_sectors')}
    Activities: {incentive.get('eligible_activities')}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    with _token_lock:
        _total_tokens["prompt"] += response.usage.prompt_tokens
        _total_tokens["completion"] += response.usage.completion_tokens
        
    return response.choices[0].message.content

def _embed(text: str) -> np.ndarray:
    """Embeds a single text string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    with _token_lock:
        _total_tokens["embedding"] += response.usage.total_tokens
    return np.array(response.data[0].embedding, dtype=np.float32)

# ── 2. Pure Vector Retrieval ─────────────────────────────────────────────────

def _fetch_companies_by_ids(company_ids: list[int]) -> list[dict]:
    """Fetches full company details from DB, preserving the ranking order."""
    if not company_ids:
        return []
        
    conn = get_connection()
    cur = conn.cursor()
    
    # Fetch all matching companies
    placeholders = ",".join(["%s"] * len(company_ids))
    cur.execute(
        f"""SELECT id, company_name, cae_primary_label, trade_description_native
            FROM companies WHERE id IN ({placeholders})""",
        tuple(company_ids)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Create a lookup dictionary
    company_dict = {
        r[0]: {"id": r[0], "company_name": r[1], "cae_primary_label": r[2], "trade_description_native": r[3]}
        for r in rows
    }
    
    # Return them in the exact order requested by the vector search
    return [company_dict[cid] for cid in company_ids if cid in company_dict]

def _get_candidates_rag(incentive: dict, embeddings_matrix: np.ndarray, ids_array: np.ndarray) -> list[dict]:
    """True semantic search across the entire dataset in milliseconds."""
    
    # 1. Align the vector space (HyDE)
    ideal_text = _generate_ideal_profile(incentive)
    
    # 2. Embed the ideal company
    ideal_vector = _embed(ideal_text)
    
    # 3. Vector search across ALL companies instantly
    # Normalize query (matrix is assumed to be normalized or we do dot product if OpenAI already normalizes)
    ideal_vector_norm = ideal_vector / np.linalg.norm(ideal_vector)
    matrix_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    similarities = matrix_norm @ ideal_vector_norm
    
    # 4. Get top indices
    top_indices = np.argsort(similarities)[::-1][:CANDIDATE_LIMIT]
    top_company_ids = [int(ids_array[i]) for i in top_indices]
    
    return _fetch_companies_by_ids(top_company_ids)

# ── 3. LLM Re-ranking ────────────────────────────────────────────────────────

def _score_with_llm(incentive: dict, candidates: list[dict]) -> list[dict]:
    """Ask the LLM to re-rank the top candidates from the semantic shortlist."""
    candidate_text = "\n".join(
        [
            f"{i+1}. {c['company_name']} | Industry: {c['cae_primary_label']} | Activity: {c['trade_description_native'][:200]}"
            for i, c in enumerate(candidates)
        ]
    )

    system_prompt = """You are an expert in Portuguese public funding and business development.
Evaluate how well companies match a specific public incentive program.
Respond ONLY with a valid JSON object containing a 'results' array."""

    user_prompt = f"""
## Incentive:
- Name: {incentive['incentive_name']}
- Type: {incentive['type']}
- Eligible sectors: {incentive['eligible_sectors']}
- Eligible activities: {incentive['eligible_activities']}
- Eligible company types: {incentive['eligible_company_types']}
- Description: {incentive['description']}

## Candidates (pre-selected by semantic similarity):
{candidate_text}

Select the 5 best-matching companies. Respond ONLY with:
{{"results": [{{"company_name": "...", "score": 8.5, "justification": "1-2 sentences"}}, ...]}}

IMPORTANT: Each company must have a DIFFERENT score. Best fit = highest score, 5th best = lowest.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    with _token_lock:
        _total_tokens["prompt"] += response.usage.prompt_tokens
        _total_tokens["completion"] += response.usage.completion_tokens

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  WARNING: LLM returned invalid JSON for '{incentive.get('incentive_name', '?')}': {e}")
        return []

    if isinstance(parsed, list):
        return parsed
    for key in parsed:
        if isinstance(parsed[key], list):
            return parsed[key]

    print(f"  WARNING: Unexpected JSON structure for '{incentive.get('incentive_name', '?')}': {list(parsed.keys())}")
    return []

# ── 4. Main Pipeline ─────────────────────────────────────────────────────────

def _process_incentive(incentive: dict, idx: int, total: int, embeddings_matrix: np.ndarray, ids_array: np.ndarray) -> list[dict]:
    """Process a single incentive end-to-end. Runs inside a thread."""
    print(f"  [{idx+1}/{total}] {incentive['incentive_name'][:55]}")

    candidates = _get_candidates_rag(incentive, embeddings_matrix, ids_array)

    try:
        top5 = _score_with_llm(incentive, candidates)
    except Exception as e:
        print(f"  [{idx+1}/{total}] LLM error: {e}")
        return []

    name_to_id = {c["company_name"]: c["id"] for c in candidates}
    results = []
    for match in top5[:5]:
        results.append({
            "incentive_id": incentive["incentive_id"],
            "incentive_name": incentive["incentive_name"],
            "company_name": match.get("company_name", ""),
            "company_id": name_to_id.get(match.get("company_name", "")),
            "score": float(match.get("score", 0)),
            "justification": match.get("justification", ""),
        })

    return results

def run_matching(output_csv: str = "matches_output.csv"):
    global _total_tokens
    _total_tokens = {"prompt": 0, "completion": 0, "embedding": 0}

    # 1. Load pre-computed embeddings
    try:
        print("Loading 250k company embeddings into memory...")
        embeddings_matrix = np.load('company_embeddings.npy')
        ids_array = np.load('company_ids.npy')
    except FileNotFoundError:
        print("ERROR: Embeddings not found. Please run 'python run.py setup' first.")
        return

    # 2. Load incentives
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM incentives;")
    cols = [desc[0] for desc in cur.description]
    incentives = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()

    print(f"\nMatching {len(incentives)} incentives with {MAX_WORKERS} parallel workers...")
    print(f"Strategy: HyDE Translation + Pure Vector Search + LLM Re-ranking\n")
    start_time = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_incentive, incentive, idx, len(incentives), embeddings_matrix, ids_array
            ): incentive
            for idx, incentive in enumerate(incentives)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    # 3. Save to DB — delete + all inserts in a single transaction
    print("\nSaving to database...")
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM matches;")
        for r in all_results:
            cur.execute(
                """INSERT INTO matches (incentive_id, company_id, company_name, score, justification)
                   VALUES (%s, %s, %s, %s, %s)""",
                (r["incentive_id"], r.get("company_id"), r["company_name"], r["score"], r["justification"]),
            )
        conn.commit()
        print(f"  Saved {len(all_results)} matches to database.")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: Database write failed, rolled back. Reason: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    elapsed = time.time() - start_time

    # Export CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "company_id"} for r in all_results])
    df = df.sort_values(["incentive_id", "score"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Cost report
    embed_cost = _total_tokens["embedding"] * 0.000_000_02
    prompt_cost = _total_tokens["prompt"] * 0.000_000_15
    completion_cost = _total_tokens["completion"] * 0.000_000_60
    total_cost = embed_cost + prompt_cost + completion_cost

    print(f"\nMatching complete. Results saved to {output_csv}")
    print(f"\n--- Performance Report ---")
    print(f"   Time elapsed:       {elapsed:.1f}s")
    print(f"   Embedding tokens:   {_total_tokens['embedding']:,}  (${embed_cost:.4f})")
    print(f"   Prompt tokens:      {_total_tokens['prompt']:,}  (${prompt_cost:.4f})")
    print(f"   Completion tokens:  {_total_tokens['completion']:,}  (${completion_cost:.4f})")
    print(f"   Total Phase 2 Cost: ${total_cost:.4f} USD")

    return df