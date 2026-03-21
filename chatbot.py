"""
chatbot.py — Conversational Q&A interface over the matches database.

Architecture: tool-calling agent loop.
The LLM decides which tool to call based on the user question.
Tools available:
  - search_matches: query the matches table
  - search_incentives: look up incentive details
  - search_companies: look up company info
  - get_top_matches_for_incentive: get ranked list for a specific incentive
"""

import json
import os
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_matches_for_incentive",
            "description": "Get the top matched companies for a specific incentive by name or ID.",
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
            "name": "search_matches",
            "description": "Search matches by company name or industry keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Company name or industry keyword to search for",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_incentives",
            "description": "List all available public incentives with their basic details.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_incentive_details",
            "description": "Get full details about a specific incentive.",
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

def get_top_matches_for_incentive(incentive_query: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.company_name, m.score, m.justification,
               i.incentive_name, i.type, i.max_funding_eur
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
        return f"No matches found for incentive: '{incentive_query}'. Run the matcher first or check the incentive name."

    result = f"Top matches for '{rows[0][3]}':\n"
    for i, row in enumerate(rows, 1):
        result += f"\n{i}. {row[0]} (Score: {row[1]}/10)\n   {row[2]}\n"
    return result


def search_matches(keyword: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.company_name, m.score, i.incentive_name, m.justification
        FROM matches m
        JOIN incentives i ON m.incentive_id = i.incentive_id
        WHERE LOWER(m.company_name) LIKE %s
        ORDER BY m.score DESC
        LIMIT 10
        """,
        (f"%{keyword.lower()}%",),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return f"No matches found for keyword: '{keyword}'"

    result = f"Matches containing '{keyword}':\n"
    for row in rows:
        result += f"\n• {row[0]} → {row[2]} (Score: {row[1]}/10)\n  {row[3]}\n"
    return result


def list_all_incentives() -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT incentive_id, incentive_name, type, max_funding_eur, deadline FROM incentives ORDER BY incentive_id;"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = "All available incentives:\n"
    for row in rows:
        result += f"\n{row[0]}: {row[1]}\n   Type: {row[2]} | Max funding: {row[3]} | Deadline: {row[4]}\n"
    return result


def get_incentive_details(incentive_query: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM incentives
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
    "get_top_matches_for_incentive": get_top_matches_for_incentive,
    "search_matches": search_matches,
    "list_all_incentives": list_all_incentives,
    "get_incentive_details": get_incentive_details,
}


def _call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)


# ── Agent loop ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for HomoDeus, a Portuguese AI company.
You help users explore public incentives and which companies are best matched to them.

You have access to tools to query the database of matches between Portuguese companies
and public funding incentives.

Guidelines:
- Always use a tool to answer questions about specific matches or incentives.
- If you don't know something or the data isn't available, say so clearly.
- Be concise and factual. Format results clearly.
- If asked about a company, search for it by name.
- If asked about an incentive, get its details and top matches.
"""


def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Single turn of the agent loop.
    Tool calls are resolved first (non-streamed, fast DB lookups).
    Final answer is streamed token by token so first chunk arrives immediately.
    Returns (assistant_reply, updated_history).
    """
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # ── Phase 1: resolve tool calls (non-streamed) ──────────────────────────
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break  # no more tools needed, move to streaming phase

        # Execute each tool and feed results back
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

    # ── Phase 2: stream the final answer ────────────────────────────────────
    print("\nAssistant: ", end="", flush=True)
    full_reply = ""

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
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