"""
chatbot.py — Conversational Q&A interface over the matches database.

"""

import json
import os
import numpy as np
from openai import OpenAI
from db import get_connection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1. Load Embeddings for Semantic Search ───────────────────────────────────
try:
    print("Loading AI embeddings for semantic search...")
    EMBEDDINGS_MATRIX = np.load('company_embeddings.npy')
    COMPANY_IDS = np.load('company_ids.npy')
except FileNotFoundError:
    print("Warning: Embeddings not found. Run 'python run.py setup' first.")
    EMBEDDINGS_MATRIX, COMPANY_IDS = None, None

# ── 2. Tool definitions ──────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_companies_by_description",
            "description": (
                "Semantic search for companies based on what they do, their industry, or their products. "
                "Use this when the user asks for companies by concept (e.g., 'Find me shoe manufacturers', "
                "'Are there any tech startups?', 'Show me companies that build software')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The concept, product, or activity to search for.",
                    }
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_company_matches",
            "description": "Check whether a specific named company has been matched to any incentives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Exact or partial company name.",
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
            "name": "search_incentives_by_sector",
            "description": (
                "Fetches all available incentives and their eligible sectors. "
                "Use this to find incentives relevant to a specific industry or sector keyword."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_scoring_companies",
            "description": "Get the companies with the highest match scores across all incentives.",
            "parameters": {"type": "object", "properties": {}},
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

# ── 3. Tool implementations ──────────────────────────────────────────────────

def find_companies_by_description(description: str) -> str:
    """Semantic search using pre-computed embeddings."""
    if EMBEDDINGS_MATRIX is None:
        return "Error: Semantic search is offline (embeddings missing)."
        
    # 1. Embed the search query
    response = client.embeddings.create(model="text-embedding-3-small", input=[description])
    query_vector = np.array(response.data[0].embedding, dtype=np.float32)
    
    # 2. Vector math against all 250k companies
    query_norm = query_vector / np.linalg.norm(query_vector)
    matrix_norm = EMBEDDINGS_MATRIX / np.linalg.norm(EMBEDDINGS_MATRIX, axis=1, keepdims=True)
    similarities = matrix_norm @ query_norm
    
    # 3. Get top 3 semantic matches
    top_indices = np.argsort(similarities)[::-1][:3]
    top_ids = [int(COMPANY_IDS[i]) for i in top_indices]
    
    # 4. Fetch company details and their incentive matches from DB
    conn = get_connection()
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(top_ids))
    
    cur.execute(f"SELECT id, company_name, trade_description_native FROM companies WHERE id IN ({placeholders})", tuple(top_ids))
    companies = {r[0]: {"name": r[1], "desc": r[2], "matches": []} for r in cur.fetchall()}
    
    cur.execute(f"""
        SELECT m.company_id, i.incentive_name, m.score, m.justification 
        FROM matches m 
        JOIN incentives i ON m.incentive_id = i.incentive_id
        WHERE m.company_id IN ({placeholders})
    """, tuple(top_ids))
    
    for row in cur.fetchall():
        if row[0] in companies:
            companies[row[0]]["matches"].append(f"  - {row[1]} (Score {row[2]}/10): {row[3]}")
            
    cur.close()
    conn.close()

    result = f"Top 3 companies matching the concept '{description}':\n\n"
    for cid in top_ids:
        c = companies.get(cid)
        if c:
            result += f"Company: {c['name']}\n"
            result += f"Activity: {c['desc'][:200]}...\n"
            if c['matches']:
                result += "Matched Incentives:\n" + "\n".join(c['matches']) + "\n\n"
            else:
                result += "Matched Incentives: None.\n\n"
                
    return result

def search_incentives_by_sector() -> str:
    """Dumps all sector rules so the LLM can semantically filter them."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT incentive_id, incentive_name, eligible_sectors, max_funding_eur FROM incentives")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    result = "Here are the sector eligibility rules for all incentives. Read these carefully to determine which ones apply to the user's request:\n\n"
    for row in rows:
        result += f"- [{row[0]}] {row[1]} | Max Funding: {row[3]} | Eligible Sectors: {row[2]}\n"
    return result

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
        return f"NO_MATCH: '{company_name}' was not matched to any incentive."

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
        result += f"\n{i}. {row[0]} (Score: {row[1]}/10)\n   Why: {row[2]}\n"
    return result

def get_top_scoring_companies() -> str:
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
        LIMIT 10
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return "No match data found."

    result = f"Top {len(rows)} highest-scoring companies:\n"
    for i, row in enumerate(rows, 1):
        result += f"\n{i}. {row[0]} | Top score: {row[1]}/10 | Matched {row[2]} program(s)\n"
    return result

def list_all_incentives() -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT incentive_id, incentive_name, type, max_funding_eur FROM incentives ORDER BY incentive_id;")
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

# ── 4. Tool dispatcher ───────────────────────────────────────────────────────

TOOL_MAP = {
    "find_companies_by_description": find_companies_by_description,
    "search_company_matches": search_company_matches,
    "get_top_matches_for_incentive": get_top_matches_for_incentive,
    "search_incentives_by_sector": search_incentives_by_sector,
    "get_top_scoring_companies": get_top_scoring_companies,
    "list_all_incentives": list_all_incentives,
    "get_incentive_details": get_incentive_details,
}

def _call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)

# ── 5. System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are HomoDeus AI, an expert advisor on Portuguese public incentives.
Your job is to help users explore a massive database of companies and funding programs.

You have access to powerful tools. ALWAYS use a tool before answering. Never guess.

TOOL ROUTING STRATEGY:
- If the user asks for companies by a CONCEPT or INDUSTRY (e.g., "Find me shoe makers", "Are there tech startups?"): Use `find_companies_by_description`.
- If the user asks about a SPECIFIC company by NAME: Use `search_company_matches`.
- If the user asks "What incentives exist for [Sector]?": Use `search_incentives_by_sector`, read the raw data, and figure out which ones apply to their sector.
- If the user asks about an incentive's top matches: Use `get_top_matches_for_incentive`.

RESPONSE RULES:
- Be incredibly helpful and analytical. Don't just list data; explain it.
- If a tool returns "Justifications" or "Why" a company matched, YOU MUST summarize that reasoning for the user. Act like an expert consultant.
- Keep responses concise but highly informative. Use bullet points for readability.
- If you use `search_incentives_by_sector`, explicitly tell the user which programs target their sector specifically, and which are "Open to all sectors".
"""

# ── 6. Agent loop ────────────────────────────────────────────────────────────

def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    
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
    print("\n" + "=" * 60)
    print("  HomoDeus — Public Incentives Q&A Agent")
    print("  Type 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60 + "\n")
    print("Try asking:")
    print("  - Find me companies that manufacture clothing and tell me their matches.")
    print("  - What incentives exist for the agriculture sector?")
    print("  - Which companies scored highest overall?")
    print("  - Which companies qualified for SIFIDE II?")
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