.PHONY: up down logs test lint db-reset migrate revision fe-install fe-dev

BACKEND_CONTAINER := topicdrift-backend

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f backend

test:
	docker exec -it $(BACKEND_CONTAINER) pytest -v

lint:
	docker exec -it $(BACKEND_CONTAINER) sh -c "ruff check . --fix && ruff format ."
	cd frontend && bun run lint

# Apply migrations to the dev database.
migrate:
	docker exec -it $(BACKEND_CONTAINER) alembic upgrade head

# Generate a new migration from current ORM models.
# Usage: make revision m="add papers table"
revision:
	docker exec -it $(BACKEND_CONTAINER) alembic revision --autogenerate -m "$(m)"

db-reset:
	docker compose down -v
	docker compose up -d db

fe-install:
	cd frontend && bun install

fe-dev:
	cd frontend && bun run dev
