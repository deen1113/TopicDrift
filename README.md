# TopicDrift

Analyzing how research themes evolve over time at long-running software engineering conferences (ICSE, ICSA/ECSA).

## What this is

CS4530 Project 1: ingest paper metadata (titles, abstracts, years) for a single conference series, apply keyword frequency / topic clustering, and visualize how topics rise and fall across decades.

## Architecture

- **backend/** — FastAPI + Pydantic v2, three-layer (routes → services → repositories) over SQLAlchemy 2.0 async + Postgres 17. Alembic for migrations. Python 3.13, managed by uv.
- **backend/app/pipeline/** — data ingestion + analysis (DBLP/Semantic Scholar fetchers, TF-IDF, clustering). Runs as CLI jobs, not API requests.
- **frontend/** — React 19 + TypeScript + Vite 6, TanStack Query v5 + Zod 4, Tailwind v4 (CSS-first config). Bun is the package manager.
- **scripts/** — DB init SQL (creates the test database on first container start).

Conventions for both halves live in [`CONVENTIONS.md`](./CONVENTIONS.md). **Read it before adding code.** Host-tool setup is in [`DEPENDENCIES.md`](./DEPENDENCIES.md).

## Getting started

```bash
make check-deps   # verify docker, bun, uv, python, make are installed
make bootstrap    # uv lock + bun install + docker compose build
make up           # start backend + postgres
make migrate      # apply alembic migrations
```

Day-to-day:

```bash
make logs         # tail backend logs
make test         # pytest inside the container
make lint         # ruff + tsc (read-only)
make format       # ruff check --fix && ruff format
make revision m="add papers table"  # autogenerate a migration
make db-reset     # drop volumes, restart just the db
make fe-dev       # vite dev server on :5173
make shell        # bash into the backend container
```

Default ports:
- Backend API: <http://localhost:8000> (docs at `/docs`)
- Postgres: `localhost:5432` (user/pw `postgres`, dev DB `topicdrift`, test DB `topicdrift_test`)
- Frontend: <http://localhost:5173>

## Status

Bootstrap scaffolding only. No example entity wired up yet — see §10 of `CONVENTIONS.md` for the checklist; step 6 (one entity end-to-end) is next.
