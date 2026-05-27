# TopicDrift

Analyzing how research themes evolve over time at long-running software engineering conferences (ICSE, ICSA/ECSA).

## What this is

CS4530 Project 1: ingest paper metadata (titles, abstracts, years) for a single conference series, apply keyword frequency / topic clustering, and visualize how topics rise and fall across decades.

## Architecture

- **backend/** — FastAPI + Pydantic, three-layer (routes → services → repositories) over SQLAlchemy 2.0 async + Postgres. Alembic for migrations.
- **frontend/** — React + TypeScript + Vite, TanStack Query + Zod, Tailwind. Bun is the package manager.
- **scripts/** — DB init SQL (creates the test database on first container start).

Conventions for both halves live in `CONVENTIONS.md`. **Read it before adding code.** Every entity follows the same backend layering and one-hook-file frontend pattern.

## Getting started

Prerequisites: Docker, Bun, Make.

```bash
make up           # start backend + postgres via docker compose
make down         # stop everything
make logs         # tail backend logs
make migrate      # alembic upgrade head
make revision m="add papers table"  # autogenerate a migration
make test         # run backend pytest suite inside the container
make lint         # ruff (backend) + bun lint (frontend)
make db-reset     # drop volumes, restart just the db
make fe-install   # bun install
make fe-dev       # vite dev server
```

Default ports:
- Backend API: <http://localhost:8000> (docs at `/docs`)
- Postgres: `localhost:5432` (user/pw `postgres`, dev DB `topicdrift`, test DB `topicdrift_test`)
- Frontend: <http://localhost:5173>

## Status

Bootstrap scaffolding only. No example entity wired up yet — see §10 of `CONVENTIONS.md` for the new-project checklist; step 5 (one entity end-to-end) is the next thing to do before opening feature tickets.
