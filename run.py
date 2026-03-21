"""
run.py — Main entry point for the HomoDeus matching system.

Usage:
  python run.py setup     # Load CSVs into PostgreSQL
  python run.py match     # Run matching (LLM scores companies per incentive)
  python run.py chat      # Start the conversational chatbot
  python run.py all       # Run setup + match + chat in sequence
"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ("setup", "all"):
        from db import setup_database

        companies_csv = os.getenv("COMPANIES_CSV", "data/companies.csv")
        incentives_csv = os.getenv("INCENTIVES_CSV", "data/portugal-public-incentives.csv")
        use_cases_csv = os.getenv("USE_CASES_CSV", "data/homodeus-use-cases.csv")

        print("Setting up database...")
        setup_database(companies_csv, incentives_csv, use_cases_csv)

    if command in ("match", "all"):
        from matcher import run_matching
        print("\nRunning matching pipeline...")
        run_matching(output_csv="matches_output.csv")

    if command in ("chat", "all"):
        from chatbot import run_chatbot
        run_chatbot()

    if command not in ("setup", "match", "chat", "all"):
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()