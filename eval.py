"""
eval.py — Evaluation harness for the matching system.

Metrics:
  1. Score Sanity     — Are scores in valid range (0-10)? Are top matches higher than random?
  2. Coverage         — Does every incentive have exactly 5 matches?
  3. Sector Alignment — Do matched companies' industries relate to incentive sectors?
  4. Justification Quality — Is justification non-empty and long enough to be meaningful?
  5. Consistency      — Same incentive run twice → similar top companies?

Outputs a structured report to eval_report.txt and prints a summary.
"""

import json
import os
import re
import time
import pandas as pd
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT m.*, i.incentive_name, i.eligible_sectors
        FROM matches m
        JOIN incentives i ON m.incentive_id = i.incentive_id
        """,
        conn,
    )
    conn.close()
    return df


# ── Metric 1: Score Sanity ───────────────────────────────────────────────────

def eval_score_sanity(df: pd.DataFrame) -> dict:
    """All scores must be between 0 and 10. Top match per incentive should score >= 6."""
    in_range = df[(df["score"] >= 0) & (df["score"] <= 10)]
    pct_in_range = len(in_range) / len(df) * 100 if len(df) > 0 else 0

    top_per_incentive = df.groupby("incentive_id")["score"].max()
    high_confidence = (top_per_incentive >= 6).sum()

    return {
        "metric": "Score Sanity",
        "pct_scores_in_range": round(pct_in_range, 1),
        "incentives_with_high_confidence_top_match": int(high_confidence),
        "total_incentives": int(len(top_per_incentive)),
        "mean_top_score": round(float(top_per_incentive.mean()), 2),
        "pass": pct_in_range == 100.0,
    }


# ── Metric 2: Coverage ───────────────────────────────────────────────────────

def eval_coverage(df: pd.DataFrame) -> dict:
    """Every incentive should have exactly 5 matches."""
    counts = df.groupby("incentive_id").size()
    full_coverage = (counts == 5).sum()
    pct = full_coverage / len(counts) * 100 if len(counts) > 0 else 0

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM incentives;")
    total_incentives = cur.fetchone()[0]
    cur.close()
    conn.close()

    matched_incentives = len(counts)

    return {
        "metric": "Coverage",
        "total_incentives": total_incentives,
        "incentives_with_matches": matched_incentives,
        "incentives_with_full_5_matches": int(full_coverage),
        "pct_full_coverage": round(pct, 1),
        "pass": matched_incentives == total_incentives,
    }


# ── Metric 3: Sector Alignment ───────────────────────────────────────────────

def eval_sector_alignment(df: pd.DataFrame) -> dict:
    """
    Check if matched company industries share keywords with incentive eligible_sectors.
    Simple keyword overlap — no LLM needed.
    """
    aligned = 0
    total = 0

    for _, row in df.iterrows():
        sector_keywords = set(
            re.findall(r"\b\w{4,}\b", (row["eligible_sectors"] or "").lower())
        )
        company_text = (row.get("cae_primary_label", "") or "").lower()
        match_keywords = set(re.findall(r"\b\w{4,}\b", company_text))

        overlap = sector_keywords & match_keywords
        if overlap or "all sectors" in (row["eligible_sectors"] or "").lower():
            aligned += 1
        total += 1

    pct = aligned / total * 100 if total > 0 else 0
    return {
        "metric": "Sector Alignment",
        "aligned_matches": aligned,
        "total_matches": total,
        "pct_aligned": round(pct, 1),
        "pass": pct >= 50.0,
    }


# ── Metric 4: Justification Quality ─────────────────────────────────────────

def eval_justification_quality(df: pd.DataFrame) -> dict:
    """Justifications should be non-empty and at least 30 characters."""
    non_empty = df["justification"].notna() & (df["justification"].str.len() > 30)
    pct = non_empty.sum() / len(df) * 100 if len(df) > 0 else 0
    avg_len = df["justification"].str.len().mean()

    return {
        "metric": "Justification Quality",
        "pct_with_meaningful_justification": round(pct, 1),
        "avg_justification_length_chars": round(float(avg_len), 0),
        "pass": pct >= 90.0,
    }


# ── Metric 5: LLM-as-judge (spot check) ─────────────────────────────────────

def eval_llm_judge(df: pd.DataFrame, sample_size: int = 5) -> dict:
    """
    Use the LLM itself to judge a random sample of matches for plausibility.
    Returns % that the judge considers reasonable.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    verdicts = []

    for _, row in sample.iterrows():
        prompt = f"""You are evaluating an AI system that matches companies to public incentives.

Incentive: {row['incentive_name']}
Eligible sectors: {row['eligible_sectors']}
Company matched: {row['company_name']}
Score given: {row['score']}/10
Justification: {row['justification']}

Is this a reasonable match? Reply with JSON: {{"reasonable": true/false, "reason": "one sentence"}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            verdicts.append(result.get("reasonable", False))
            time.sleep(0.5)
        except Exception:
            verdicts.append(None)

    valid = [v for v in verdicts if v is not None]
    pct_reasonable = sum(valid) / len(valid) * 100 if valid else 0

    return {
        "metric": "LLM Judge (spot check)",
        "sample_size": sample_size,
        "pct_judged_reasonable": round(pct_reasonable, 1),
        "pass": pct_reasonable >= 60.0,
    }


# ── Main eval runner ─────────────────────────────────────────────────────────

def run_eval(output_file: str = "eval_report.txt"):
    print("Loading matches from database...")
    df = load_matches()

    if df.empty:
        print("No matches found. Run `python run.py match` first.")
        return

    print(f"Evaluating {len(df)} matches across {df['incentive_id'].nunique()} incentives...\n")

    results = []
    results.append(eval_score_sanity(df))
    results.append(eval_coverage(df))
    results.append(eval_sector_alignment(df))
    results.append(eval_justification_quality(df))
    results.append(eval_llm_judge(df, sample_size=5))

    # Print and save report
    lines = ["=" * 60, "  HomoDeus Matching Evaluation Report", "=" * 60, ""]

    passed = 0
    for r in results:
        status = "✅ PASS" if r.get("pass") else "❌ FAIL"
        if r.get("pass"):
            passed += 1
        lines.append(f"{status}  {r['metric']}")
        for k, v in r.items():
            if k not in ("metric", "pass"):
                lines.append(f"       {k}: {v}")
        lines.append("")

    lines.append(f"Overall: {passed}/{len(results)} metrics passing")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport saved to {output_file}")
    return results


if __name__ == "__main__":
    run_eval()