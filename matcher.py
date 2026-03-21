"""
matcher.py — Core matching logic.

Strategy:
  1. For each incentive, use SQL to pull a candidate shortlist (~100 companies)
     by doing keyword matching on eligible_sectors vs cae_primary_label.
  2. Send the shortlist to the LLM in a single batched prompt.
  3. LLM returns a ranked JSON list of top 5 with scores + justifications.
  4. Save results to the matches table and export to CSV.

This keeps LLM calls to one per incentive (20 total), staying well within budget.
"""

import json
import os
import time
import pandas as pd
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# How many candidates to shortlist per incentive before LLM scoring
CANDIDATE_LIMIT = 80

# Track API cost
_total_tokens = {"prompt": 0, "completion": 0}


def _get_candidates(cur, incentive: dict) -> list[dict]:
    """
    Pull candidate companies from PostgreSQL using keyword matching.
    Falls back to a random sample if no keyword matches are found.
    """
    sectors_text = (incentive.get("eligible_sectors") or "").lower()
    keywords = [w.strip() for w in sectors_text.replace(",", " ").split() if len(w) > 3]

    companies = []

    if keywords:
        # Build a WHERE clause with ILIKE for each keyword
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

    # Fallback: random sample if keywords yield nothing
    if not companies:
        cur.execute(
            """
            SELECT id, company_name, cae_primary_label, trade_description_native
            FROM companies
            ORDER BY RANDOM()
            LIMIT %s
            """,
            (CANDIDATE_LIMIT,),
        )
        companies = cur.fetchall()

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
    Ask the LLM to rank the top 5 companies from the candidate list.
    Returns a list of dicts with company_name, score, justification.
    """
    candidate_text = "\n".join(
        [
            f"{i+1}. {c['company_name']} | Industry: {c['cae_primary_label']} | Activity: {c['trade_description_native'][:200]}"
            for i, c in enumerate(candidates)
        ]
    )

    system_prompt = """You are an expert in Portuguese public funding and business development.
Your job is to evaluate how well companies match a specific public incentive program.
You must respond ONLY with a valid JSON array. No explanation outside the JSON."""

    user_prompt = f"""
## Incentive to match:
- Name: {incentive['incentive_name']}
- Type: {incentive['type']}
- Eligible sectors: {incentive['eligible_sectors']}
- Eligible activities: {incentive['eligible_activities']}
- Eligible company types: {incentive['eligible_company_types']}
- Description: {incentive['description']}

## Candidate companies (choose the best 5):
{candidate_text}

## Task:
Select the 5 best-matching companies. For each, provide:
- company_name (exact match from list above)
- score: float from 0.0 to 10.0 (10 = perfect fit)
- justification: 1-2 sentences explaining why this company fits

Respond ONLY with a JSON array like:
[
  {{"company_name": "...", "score": 8.5, "justification": "..."}},
  ...
]
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

    # Track token usage for cost reporting
    usage = response.usage
    _total_tokens["prompt"] += usage.prompt_tokens
    _total_tokens["completion"] += usage.completion_tokens

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Handle both {"results": [...]} and plain [...]
    if isinstance(parsed, list):
        return parsed
    for key in parsed:
        if isinstance(parsed[key], list):
            return parsed[key]
    return []


def run_matching(output_csv: str = "matches_output.csv"):
    """
    Main matching pipeline:
    - Loops over all incentives
    - Gets candidates via SQL
    - Scores via LLM
    - Saves to DB and CSV
    """
    conn = get_connection()
    cur = conn.cursor()

    # Load all incentives
    cur.execute("SELECT * FROM incentives;")
    cols = [desc[0] for desc in cur.description]
    incentives = [dict(zip(cols, row)) for row in cur.fetchall()]

    # Clear previous matches
    cur.execute("DELETE FROM matches;")
    conn.commit()

    all_results = []

    for idx, incentive in enumerate(incentives):
        print(f"\n[{idx+1}/{len(incentives)}] Matching: {incentive['incentive_name']}")

        candidates = _get_candidates(cur, incentive)
        print(f"  → {len(candidates)} candidates found via SQL")

        if not candidates:
            print("  → No candidates, skipping.")
            continue

        try:
            top5 = _score_with_llm(incentive, candidates)
        except Exception as e:
            print(f"  → LLM error: {e}")
            continue

        # Map company names back to IDs
        name_to_id = {c["company_name"]: c["id"] for c in candidates}

        for match in top5[:5]:
            company_name = match.get("company_name", "")
            score = float(match.get("score", 0))
            justification = match.get("justification", "")
            company_id = name_to_id.get(company_name)

            cur.execute(
                """INSERT INTO matches (incentive_id, company_id, company_name, score, justification)
                   VALUES (%s, %s, %s, %s, %s)""",
                (incentive["incentive_id"], company_id, company_name, score, justification),
            )

            all_results.append(
                {
                    "incentive_id": incentive["incentive_id"],
                    "incentive_name": incentive["incentive_name"],
                    "company_name": company_name,
                    "score": score,
                    "justification": justification,
                }
            )

        conn.commit()
        print(f"  → Top 5 saved. Sleeping 1s to respect rate limits...")
        time.sleep(1)

    cur.close()
    conn.close()

    # Export to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Matching complete. Results saved to {output_csv}")

    # Cost report
    prompt_cost = _total_tokens["prompt"] * 0.000_000_15
    completion_cost = _total_tokens["completion"] * 0.000_000_60
    total_cost = prompt_cost + completion_cost
    print(f"\n💰 API Cost Report:")
    print(f"   Prompt tokens:     {_total_tokens['prompt']:,}")
    print(f"   Completion tokens: {_total_tokens['completion']:,}")
    print(f"   Estimated cost:    ${total_cost:.4f} USD")

    return df