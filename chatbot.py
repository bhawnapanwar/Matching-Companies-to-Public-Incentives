"""
chatbot.py — Conversational Q&A interface over the matches database.

Architecture: tool-calling agent loop with streaming output.
Phase 1: resolve tool calls (fast DB lookups, non-streamed)
Phase 2: stream the final answer token by token

Tools:
  - search_company_matches       : check if a specific company got any incentives
  - get_top_matches_for_incentive: top 5 companies for a given incentive
  - get_top_scoring_companies    : companies with highest scores across all incentives
  - search_incentives_by_sector  : find incentives relevant to a sector/industry keyword
  - get_unmatched_companies      : companies with no matches
  - list_all_incentives          : list all incentives
  - get_incentive_details        : full details of one incentive
"""

import json
import os
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_company_matches",
            "description": (
                "Check whether a specific named company has been matched to any incentives. "
                "Use when the user asks about a specific company by name, e.g. "
                "'Will DNR OBRAS get any incentive?' or 'What does DANIJO qualify for?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name or partial name to search for",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_matches_for_incentive",
            "description": "Get the top 5 matched companies for a specific incentive by name or ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incentive_query": {
                        "type": "string",
                        "description": "Incentive name (partial match OK) or incentive_id like IN01",
                    }
                },
                "required": ["incentive_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_scoring_companies",
            "description": (
                "Get the companies with the highest match scores across all incentives. "
                "Use when the user asks 'which companies scored highest?' or "
                "'which companies are the best matches overall?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many top companies to return (default 10)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_incentives_by_sector",
            "description": (
                "Find incentives relevant to a sector, industry, or type of company. "
                "Use when the user asks 'what incentives exist for X sector?' or "
                "'is there anything for construction/textile/tech companies?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector_keyword": {
                        "type": "string",
                        "description": "Sector or industry keyword e.g. 'construction', 'textile', 'tech', 'agriculture'",
                    }
                },
                "required": ["sector_keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unmatched_companies",
            "description": (
                "Find companies that were NOT matched to any incentive. "
                "Use when user asks 'which companies won't get incentives?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many examples to return (default 10)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_incentives",
            "description": "List all available public incentives with basic info.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_incentives_by_field",
            "description": (
                "Search incentives by any field: managing entity, program name, type, deadline, or any keyword. "
                "Use when user asks 'which incentives are managed by X?' or 'list Portugal 2030 programs' "
                "or 'what incentives have rolling deadlines?' or any question filtering incentives by a property."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search across all incentive fields",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_incentive_details",
            "description": "Get full details about a specific incentive program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incentive_query": {
                        "type": "string",
                        "description": "Incentive name or ID",
                    }
                },
                "required": ["incentive_query"],
            },
        },
    },
]


# ── Tool implementations ─────────────────────────────────────────────────────

def search_company_matches(company_name: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.company_name, m.score, i.incentive_name, m.justification
        FROM matches m
        JOIN incentives i ON m.incentive_id = i.incentive_id
        WHERE LOWER(m.company_name) LIKE %s
        ORDER BY m.score DESC
        """,
        (f"%{company_name.lower()}%",),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return f"NO_MATCH: '{company_name}' was not matched to any incentive. It was not in the top 5 for any program."

    result = f"MATCHED: '{rows[0][0]}' qualified for {len(rows)} incentive(s):\n"
    for row in rows:
        result += f"\n- {row[2]} (Score: {row[1]}/10): {row[3]}\n"
    return result


def get_top_matches_for_incentive(incentive_query: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.company_name, m.score, m.justification, i.incentive_name
        FROM matches m
        JOIN incentives i ON m.incentive_id = i.incentive_id
        WHERE LOWER(i.incentive_name) LIKE %s
           OR LOWER(i.incentive_id) = LOWER(%s)
        ORDER BY m.score DESC
        LIMIT 5
        """,
        (f"%{incentive_query.lower()}%", incentive_query),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return f"No matches found for incentive: '{incentive_query}'."

    result = f"Top 5 companies for '{rows[0][3]}':\n"
    for i, row in enumerate(rows, 1):
        result += f"\n{i}. {row[0]} (Score: {row[1]}/10)\n   {row[2]}\n"
    return result


def get_top_scoring_companies(limit: int = 10) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT company_name, MAX(score) as top_score, COUNT(*) as num_incentives,
               STRING_AGG(i.incentive_name, ', ' ORDER BY score DESC) as incentives
        FROM matches m
        JOIN incentives i ON m.incentive_id = i.incentive_id
        GROUP BY company_name
        ORDER BY top_score DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return "No match data found."

    result = f"Top {len(rows)} highest-scoring companies across all incentives:\n"
    for i, row in enumerate(rows, 1):
        result += f"\n{i}. {row[0]}\n   Top score: {row[1]}/10 | Matched {row[2]} incentive(s)\n   Programs: {row[3][:120]}\n"
    return result


def search_incentives_by_sector(sector_keyword: str) -> str:
    conn = get_connection()
    cur = conn.cursor()

    # Direct matches: incentive specifically mentions this sector
    cur.execute(
        """
        SELECT incentive_id, incentive_name, eligible_sectors,
               max_funding_eur, type
        FROM incentives
        WHERE LOWER(eligible_sectors) LIKE %s
           OR LOWER(eligible_activities) LIKE %s
           OR LOWER(eligible_company_types) LIKE %s
        ORDER BY incentive_id
        """,
        (f"%{sector_keyword.lower()}%",) * 3,
    )
    direct_rows = cur.fetchall()
    direct_ids = {r[0] for r in direct_rows}

    # General matches: open to all sectors (exclude already found ones)
    exclusion = tuple(direct_ids) if direct_ids else ("__none__",)
    cur.execute(
        f"""
        SELECT incentive_id, incentive_name, eligible_sectors,
               max_funding_eur, type
        FROM incentives
        WHERE LOWER(eligible_sectors) LIKE '%%all sectors%%'
          AND incentive_id NOT IN %s
        ORDER BY incentive_id
        """,
        (exclusion,),
    )
    general_rows = cur.fetchall()
    cur.close()
    conn.close()

    if not direct_rows and not general_rows:
        return f"No incentives found for '{sector_keyword}' sector."

    result = ""
    if direct_rows:
        result += f"Incentives specifically targeting '{sector_keyword}':\n"
        for row in direct_rows:
            result += f"\n- {row[0]}: {row[1]} | {row[4]} | Max: {row[3]}\n  Sectors: {row[2]}\n"

    if general_rows:
        result += f"\nIncentives open to ALL sectors (including {sector_keyword}):\n"
        for row in general_rows:
            result += f"\n- {row[0]}: {row[1]} | {row[4]} | Max: {row[3]}\n"

    return result


def get_unmatched_companies(limit: int = 10) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.company_name, c.cae_primary_label
        FROM companies c
        LEFT JOIN matches m ON LOWER(c.company_name) = LOWER(m.company_name)
        WHERE m.company_name IS NULL
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.execute(
        """
        SELECT COUNT(*) FROM companies c
        LEFT JOIN matches m ON LOWER(c.company_name) = LOWER(m.company_name)
        WHERE m.company_name IS NULL
        """
    )
    total = cur.fetchone()[0]
    cur.close()
    conn.close()

    if not rows:
        return "All companies in the database have at least one incentive match."

    result = f"Companies with no incentive match (showing {len(rows)} of {total:,} total):\n"
    for row in rows:
        result += f"\n- {row[0]} ({row[1]})\n"
    return result


def list_all_incentives() -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT incentive_id, incentive_name, type, max_funding_eur FROM incentives ORDER BY incentive_id;"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = f"All {len(rows)} available incentives:\n"
    for row in rows:
        result += f"\n{row[0]}: {row[1]} | {row[2]} | Max: {row[3]}\n"
    return result


def get_incentive_details(incentive_query: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT incentive_id, incentive_name, type, max_funding_eur,
               funding_rate_pct, eligible_sectors, eligible_activities,
               eligible_company_types, deadline, description
        FROM incentives
        WHERE LOWER(incentive_name) LIKE %s OR LOWER(incentive_id) = LOWER(%s)
        LIMIT 1
        """,
        (f"%{incentive_query.lower()}%", incentive_query),
    )
    row = cur.fetchone()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    if not row:
        return f"Incentive '{incentive_query}' not found."

    data = dict(zip(cols, row))
    return "\n".join([f"{k}: {v}" for k, v in data.items()])


def search_incentives_by_field(keyword: str) -> str:
    """Search incentives by any field — manager, program, type, deadline etc."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT incentive_id, incentive_name, managing_entity, program,
               type, max_funding_eur, deadline
        FROM incentives
        WHERE LOWER(incentive_name) LIKE %s
           OR LOWER(managing_entity) LIKE %s
           OR LOWER(program) LIKE %s
           OR LOWER(type) LIKE %s
           OR LOWER(deadline) LIKE %s
           OR LOWER(description) LIKE %s
        ORDER BY incentive_id
        """,
        (f"%{keyword.lower()}%",) * 6,
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return f"No incentives found matching '{keyword}'."

    result = "Incentives matching '" + keyword + "' (" + str(len(rows)) + " found):\n"
    for row in rows:
        result += f"\n- {row[0]}: {row[1]}\n  Manager: {row[2]} | Program: {row[3]} | Type: {row[4]} | Max: {row[5]} | Deadline: {row[6]}\n"
    return result


# ── Tool dispatcher ──────────────────────────────────────────────────────────

TOOL_MAP = {
    "search_company_matches": search_company_matches,
    "get_top_matches_for_incentive": get_top_matches_for_incentive,
    "get_top_scoring_companies": get_top_scoring_companies,
    "search_incentives_by_sector": search_incentives_by_sector,
    "get_unmatched_companies": get_unmatched_companies,
    "list_all_incentives": list_all_incentives,
    "get_incentive_details": get_incentive_details,
    "search_incentives_by_field": search_incentives_by_field,
}


def _call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for HomoDeus, a Portuguese AI company.
You help users explore public incentives and which companies are best matched to them.

You have tools to query a live database. Always call a tool — never guess from memory.

TOOL ROUTING:
- User asks about a specific company by name → search_company_matches
- User asks which companies scored highest / best overall → get_top_scoring_companies
- User asks what incentives exist for a sector (textile, construction, tech...) → search_incentives_by_sector
- User asks which companies qualified for a specific incentive → get_top_matches_for_incentive
- User asks which companies won't get incentives → get_unmatched_companies
- User asks to list all incentives → list_all_incentives
- User asks for details about a specific incentive → get_incentive_details
- User asks about incentives by manager, program, type, or any other property → search_incentives_by_field

RESPONSE RULES:
- Match response length to the question. Yes/no → one sentence first, then brief detail.
- Never dump raw data. Summarise in plain English.
- If nothing is found, say so clearly and suggest what the user can try instead.
- End with a short offer to dig deeper when relevant.
- When the tool returns two groups (specific sector matches AND all-sector matches), ALWAYS present them as two separate sections with clear headers like "Specifically for [sector]:" and "Open to all sectors (also applies):". Never merge them into one flat list.
- CRITICAL: NEVER answer questions about incentives or companies from your own memory. The database is the only source of truth. Always call a tool first, even if you think you know the answer.
"""


# ── Agent loop ───────────────────────────────────────────────────────────────

def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Single turn: resolve tool calls first, then stream the final answer.
    """
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Phase 1: resolve tool calls
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _call_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Phase 2: stream final answer
    print("\nAssistant: ", end="", flush=True)
    full_reply = ""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        print(text, end="", flush=True)
        full_reply += text

    print("\n")
    history.append({"role": "assistant", "content": full_reply})
    return full_reply, history


def run_chatbot():
    """Interactive CLI chatbot with streaming output."""
    print("\n" + "=" * 60)
    print("  HomoDeus — Public Incentives Q&A Agent")
    print("  Type 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60 + "\n")
    print("Try asking:")
    print("  - Which companies scored highest overall?")
    print("  - Is there any incentive for construction companies?")
    print("  - Will DANIJO get any incentive?")
    print("  - Which companies won't get any incentive?")
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history = []
            print("Conversation cleared.\n")
            continue

        chat(user_input, history)