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
## Output
 
- `matches_output.csv` — All matches with scores and justifications
- `company_embeddings.npy`: A massive matrix containing the 1,536-dimensional vector coordinates for every company. This allows the system to perform real-time semantic search (Cosine Similarity) across a quarter-million rows entirely in local RAM.
- `company_ids.npy`: The index that perfectly aligns the vector matrix back to the exact PostgreSQL database IDs, ensuring the fast math can be seamlessly joined back to the relational data.
 

## Architecture
---
 Setup (one-time):
  CSV → PostgreSQL → embed all 250k companies → save matrix to disk

Match (fast, repeatable):
  load matrix from disk → HyDE query generation → cosine similarity
  → top 80 candidates → LLM re-rank → top 5 per incentive → CSV

Chat:
  user question → tool routing → DB query → stream answer
 
---


## Cost

Total run cost: **< $0.30 USD**


