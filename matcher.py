"""
matcher.py — Core matching logic.

Strategy:
  1. For each incentive, use SQL to pull 80 candidate companies by keyword
     matching on both eligible_sectors and eligible_activities.
  2. Process 3 incentives in parallel using ThreadPoolExecutor.
  3. LLM scores each shortlist and returns top 5 with scores + justifications.
  4. Save results to DB and export to CSV.


"""

import json
import os
import time
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CANDIDATE_LIMIT = 80
MAX_WORKERS = 3

# Thread-safe token tracking
_token_lock = threading.Lock()
_total_tokens = {"prompt": 0, "completion": 0}


def _get_candidates(incentive: dict) -> list[dict]:
    """
    Pull candidate companies from PostgreSQL using keyword matching.
    Each call opens its own connection (required for thread safety).
    Uses both eligible_sectors and eligible_activities for richer matching.
    Falls back to a random sample if no keyword matches are found.
    """
    conn = get_connection()
    cur = conn.cursor()

    sectors_text = (incentive.get("eligible_sectors") or "").lower()
    activities_text = (incentive.get("eligible_activities") or "").lower()
    combined = f"{sectors_text} {activities_text}"
    keywords = list(set([w.strip() for w in combined.replace(",", " ").split() if len(w) > 4]))[:10]

    companies = []

    if keywords:
        conditions = " OR ".join(
            [f"LOWER(cae_primary_label) LIKE %s OR LOWER(trade_description_native) LIKE %s"
             for _ in keywords]
        )
        params = []
        for kw in keywords:
            params.extend([f"%{kw}%", f"%{kw}%"])

        cur.execute(
            f"""
            SELECT id, company_name, cae_primary_label, trade_description_native
            FROM companies
            WHERE {conditions}
            LIMIT %s
            """,
            params + [CANDIDATE_LIMIT],
        )
        companies = cur.fetchall()

    if not companies:
        cur.execute(
            """
            SELECT id, company_name, cae_primary_label, trade_description_native
            FROM companies ORDER BY RANDOM() LIMIT %s
            """,
            (CANDIDATE_LIMIT,),
        )
        companies = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "id": row[0],
            "company_name": row[1],
            "cae_primary_label": row[2],
            "trade_description_native": row[3],
        }
        for row in companies
    ]


def _score_with_llm(incentive: dict, candidates: list[dict]) -> list[dict]:
    """
    Ask the LLM to pick the top 5 companies from the candidate shortlist.
    Returns a list of dicts with company_name, score, justification.
    """
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

## Candidates:
{candidate_text}

Select the 5 best-matching companies. Respond ONLY with:
{{"results": [{{"company_name": "...", "score": 8.5, "justification": "1-2 sentences"}}, ...]}}

IMPORTANT: Each company must have a DIFFERENT score. Differentiate carefully — the best fit gets the highest score, 5th best gets the lowest. Never assign the same score to two companies.
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

    parsed = json.loads(response.choices[0].message.content)
    if isinstance(parsed, list):
        return parsed
    for key in parsed:
        if isinstance(parsed[key], list):
            return parsed[key]
    return []


def _process_incentive(incentive: dict, idx: int, total: int) -> list[dict]:
    """Process a single incentive end-to-end. Runs inside a thread."""
    print(f"  [{idx+1}/{total}] {incentive['incentive_name'][:55]}")

    candidates = _get_candidates(incentive)

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
    """
    Main matching pipeline.
    Processes all incentives in parallel (3 at a time) for speed.
    """
    global _total_tokens
    _total_tokens = {"prompt": 0, "completion": 0}

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM incentives;")
    cols = [desc[0] for desc in cur.description]
    incentives = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.execute("DELETE FROM matches;")
    conn.commit()
    cur.close()
    conn.close()

    print(f"Matching {len(incentives)} incentives with {MAX_WORKERS} parallel workers...\n")
    start_time = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_incentive, incentive, idx, len(incentives)): incentive
            for idx, incentive in enumerate(incentives)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Save to DB
    conn = get_connection()
    cur = conn.cursor()
    for r in all_results:
        cur.execute(
            """INSERT INTO matches (incentive_id, company_id, company_name, score, justification)
               VALUES (%s, %s, %s, %s, %s)""",
            (r["incentive_id"], r.get("company_id"), r["company_name"], r["score"], r["justification"]),
        )
    conn.commit()
    cur.close()
    conn.close()

    elapsed = time.time() - start_time

    # Export CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "company_id"} for r in all_results])
    df = df.sort_values(["incentive_id", "score"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Report
    prompt_cost = _total_tokens["prompt"] * 0.000_000_15
    completion_cost = _total_tokens["completion"] * 0.000_000_60
    total_cost = prompt_cost + completion_cost

    print(f"\nMatching complete. Results saved to {output_csv}")
    print(f"\n--- Performance Report ---")
    print(f"   Time elapsed:      {elapsed:.1f}s")
    print(f"   Prompt tokens:     {_total_tokens['prompt']:,}")
    print(f"   Completion tokens: {_total_tokens['completion']:,}")
    print(f"   Estimated cost:    ${total_cost:.4f} USD")

    return df