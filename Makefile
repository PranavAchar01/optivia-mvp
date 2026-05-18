.PHONY: install dev test lint typecheck serve db-migrate db-migrate-new

install:
	pip install -e ".[dev]"

dev:
	uvicorn backend.main:app --reload --port 8000

serve:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check backend/ cli/ tests/

typecheck:
	mypy backend/ cli/

db-migrate:
	psql "$$DATABASE_URL" -f backend/db/schema.sql

# Safe to re-run on existing databases (all statements use IF NOT EXISTS).
# Adds experience_records + bandit_state tables from the patent-spec update.
db-migrate-new: db-migrate

# Run the CLI
run:
	python -m cli.main run "$(PROMPT)"
