"""
chatbot.py — Conversational Q&A interface over the matches database.

Architecture: tool-calling agent loop with streaming output.
Phase 1: resolve tool calls (fast DB lookups, non-streamed)
Phase 2: stream the final answer token by token

Tools:
  - search_company_matches: check if a company got any incentives
  - get_top_matches_for_incentive: top 5 companies for a given incentive
  - get_unmatched_companies: find companies with no incentive matches
  - list_all_incentives: list all incentives
  - get_incentive_details: full details of one incentive
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
                "Check whether a specific company has been matched to any incentives. "
                "Use this when the user asks about a specific company, e.g. "
                "'Will DNR OBRAS get any incentive?' or 'What incentives does X qualify for?'"
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
            "name": "get_unmatched_companies",
            "description": (
                "Find companies in the database that were NOT matched to any incentive. "
                "Use when user asks 'which companies won't get incentives?' or similar."
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
        return f"NO_MATCH: No incentive matches found for '{company_name}'. The company either was not in the top 5 for any incentive, or the name does not exist in the database."

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
        SELECT COUNT(*)
        FROM companies c
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


# ── Tool dispatcher ──────────────────────────────────────────────────────────

TOOL_MAP = {
    "search_company_matches": search_company_matches,
    "get_top_matches_for_incentive": get_top_matches_for_incentive,
    "get_unmatched_companies": get_unmatched_companies,
    "list_all_incentives": list_all_incentives,
    "get_incentive_details": get_incentive_details,
}


def _call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for HomoDeus, a Portuguese AI company.
You help users explore public incentives and which companies are best matched to them.

You have access to tools to query a database of matches between Portuguese companies
and public funding incentive programs.

STRICT RESPONSE RULES:
- Always call a tool before answering — never guess from memory.
- Match response length to the question:
  * Yes/no questions (e.g. "will X get an incentive?") -> answer YES or NO in the first sentence, then add one short follow-up line. Offer to give more detail if wanted.
  * "List" questions -> short bullet list only, no extra commentary.
  * "Tell me about X" -> brief structured summary, 3-5 lines max.
- Never dump raw scores, IDs, or long justifications unless the user explicitly asks for details.
- If a company has no matches, say so clearly in one sentence and explain why (not in top 5 for any incentive).
- If the user asks "which companies won't get incentives?" use the get_unmatched_companies tool.
- Always end with a short offer to dig deeper if relevant.
"""


# ── Agent loop ───────────────────────────────────────────────────────────────

def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Single turn of the agent loop with streaming final response.
    Phase 1: resolve all tool calls (fast DB lookups)
    Phase 2: stream the final answer token by token
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    # Phase 2: stream the final answer
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
    print("Example questions:")
    print("  - Will DNR OBRAS get any incentive?")
    print("  - Which companies qualified for SIFIDE II?")
    print("  - Which companies won't get any incentive?")
    print("  - Tell me about the Startup Voucher program")
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