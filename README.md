# Matching Companies to Public Incentives

Matches Portuguese companies to public funding incentives using LLM scoring, with a conversational Q&A interface.
 
---
 
## Setup 
 
### 1. Install dependencies
 
```bash
pip install psycopg2-binary pandas openai python-dotenv
```
 
### 2. Create your `.env` file
 
Copy `.env.example` to `.env` and fill in your values:
 
```
OPENAI_API_KEY=sk-...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=homodeus
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```
 
### 3. Create the PostgreSQL database
 
Open a terminal and run:
 
```bash
psql -U postgres -c "CREATE DATABASE homodeus;"
```
 
### 4. Place the data files
 
Put the CSVs in a `data/` folder:
 
```
data/
  companies.csv
  portugal-public-incentives.csv
  homodeus-use-cases.csv
```
 
---
 
## Running the system
 
```bash
# Step 1: Load data into PostgreSQL
python run.py setup
 
# Step 2: Run LLM matching (scores top 5 companies per incentive)
python run.py match
 
# Step 3: Start the chatbot
python run.py chat
 
# Or run everything in sequence
python run.py all
```
 
---
 
 
## Output
 
- `matches_output.csv` — All matches with scores and justifications

 
---
 
## Architecture Decisions
 
| Decision | Rationale |
|---|---|
| PostgreSQL | Required by spec; handles 250k companies efficiently |
| SQL pre-filter before LLM | 250k companies × 20 incentives = too expensive to score all. SQL narrows to ~80 candidates per incentive first |
| gpt-4o-mini | Cost-efficient for structured scoring; ~$0.01 total for full run |
| Batched LLM calls | One LLM call per incentive (20 total), not per company pair |
| Tool-calling chatbot | Agent loop allows multi-turn reasoning with DB lookups |
 
---
 
## Cost
 
Full matching run (20 incentives × 80 candidates): **< $0.05 USD**

