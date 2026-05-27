# Coding Practices & Repository Conventions

This document is intended to be handed to an AI coding agent (e.g. Claude Code) at the start of a new project. It captures the conventions that have proven to work well on a production FastAPI + React + Supabase project (StrideTrack), so the new project can be set up the same way from day one.

Treat everything here as **defaults, not dogma** — but deviate only with a stated reason.

> **Project-specific deviation (TopicDrift):** this repo uses plain Postgres + SQLAlchemy 2.0 async + Alembic instead of Supabase. The original conventions reference a Supabase client; here the equivalent is an `AsyncSession`. See §6 for details and §2 for the model-layer addition.

---

## 1. Repository Layout

```
project-root/
├── backend/
│   ├── app/
│   │   ├── core/              # config, db engine/session, exceptions, shared infra
│   │   ├── models/            # SQLAlchemy ORM models (one file per entity)
│   │   ├── schemas/           # Pydantic schemas (one file per entity)
│   │   ├── repositories/      # DB access (one file per entity)
│   │   ├── services/          # Business logic (one file per entity)
│   │   ├── routes/            # HTTP endpoints (one file per entity)
│   │   └── main.py            # FastAPI app, router registration
│   ├── alembic/               # migration env + versioned scripts
│   ├── alembic.ini
│   ├── tests/
│   │   ├── conftest.py        # fixtures (httpx client, db session, cleanup)
│   │   └── tests/             # actual test files, suffixed `_tests.py`
│   ├── pyproject.toml         # ruff + pytest config lives here
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── lib/               # api client (axios), utilities
│   │   ├── hooks/             # one `useX.hooks.ts` per entity/concern
│   │   ├── components/
│   │   │   └── shared/        # QueryLoading, QueryError, etc.
│   │   ├── pages/             # route-level components
│   │   ├── types/             # cross-file TS types
│   │   ├── index.css          # CSS custom properties = theme tokens
│   │   └── App.tsx            # router
│   ├── package.json           # Bun as package manager
│   └── tsconfig.json
├── scripts/                   # one-off setup SQL (test DB init, etc.)
├── docker-compose.yml         # backend + postgres
├── Makefile                   # `make up`, `make down`, `make test`, `make lint`, `make migrate`
└── README.md
```

**Rule:** Every entity (Paper, Conference, Topic, …) has exactly five files on the backend — `models/`, `schemas/`, `repositories/`, `services/`, `routes/` — and one hook file on the frontend (`useX.hooks.ts`). No exceptions. If you can't find a clean home for something, the architecture isn't wrong, the design is unclear — stop and clarify before adding a new folder.

> The fifth file (`models/`) is the TopicDrift-specific addition for the SQLAlchemy ORM model. The original Supabase-based StrideTrack convention had only four files because Supabase has no ORM layer.

---

## 2. Backend Architecture — The Three-Layer Pattern

This is the non-negotiable spine of the codebase. Every request flows: **route → service → repository → DB**, and the response flows back the same way.

### 2.1 Schemas (Pydantic)

Pydantic schemas are used **at every layer boundary**, not just the API decorator. This gives type safety, validation, and clear contracts between layers.

Three schemas per entity, minimum:
- `EntityCreate` — fields needed to create the row (no `id`, no `created_at`)
- `EntityUpdate` — all fields optional (`field: T | None = None`)
- `EntityResponse` — what gets returned to the client

### 2.2 Repositories

- **Only layer that touches the database.**
- Takes an `AsyncSession` in `__init__` (injected via the `get_db` FastAPI dependency).
- Owns the SQLAlchemy queries and any ORM ↔ Pydantic conversion. Returns Pydantic objects, not raw rows or ORM instances. Conversion happens **here**, not in the service.
- Raises domain exceptions (`NotFoundException`) — never lets DB errors bubble up raw.
- Every public method has a docstring and logs `Repository: <action>`.

### 2.3 Services

- Business logic only. **No DB access** — depends on a repository injected in `__init__`.
- Takes Pydantic input (`AthleteCreate`) and returns Pydantic output (`AthleteResponse`).
- Logs `Service: <action>`.
- If logic is trivial (pure pass-through to repo), the service is still required — consistency beats brevity.

### 2.4 Routes

- Thin HTTP layer. Validates input via Pydantic, calls a service, returns the result.
- Uses FastAPI dependency injection to wire `repository → service → route`.
- Sets explicit `response_model` and `status_code` on every endpoint.
- Logs `Route: <METHOD /path>`.

### 2.5 Anti-patterns to refuse

- Repositories returning raw ORM instances or `dict`s instead of Pydantic objects.
- Services importing from `sqlalchemy` or holding an `AsyncSession` directly.
- Routes containing business logic (more than a couple of lines beyond `await service.foo()`).
- A "utils" or "helpers" folder used as a dumping ground.
- Skipping the service layer "because it's just a pass-through." It isn't optional.
- Defining ORM models inside `repositories/` files — models live in `models/`.

---

## 3. Backend Linting (ruff)

Configure ruff in `pyproject.toml` with these rules. The team uses ruff exclusively — no black, no isort, no flake8.

- Double quotes for strings.
- Modern union syntax (`str | None`) — Python 3.10+ only.
- Type annotations on **everything**, including `-> None` on `__init__`.
- Import order: stdlib → third-party → local, blank line between each group.
- Docstrings on every class and every public method (one-line is fine for trivial ones).
- Run `ruff check . --fix && ruff format .` before every commit.

Recommended starter ruleset: enable `E, F, I, ANN, D, UP, B, SIM`. Keep `ANN201` on FastAPI route handlers — they should declare return types matching `response_model`.

---

## 4. Backend Testing

- pytest, with `pytest-asyncio` in auto mode for async tests.
- Test files live in `tests/tests/` and are named `<entity>_tests.py` (not `test_<entity>.py`).
- `conftest.py` provides session-scoped `test_client` and `supabase_client` fixtures, plus an `autouse` cleanup fixture that wipes test tables **before and after each test**.
- **Cleanup must include every table touched by tests.** When adding tests for a new entity, the cleanup fixture is updated in the same PR.
- Use `Faker` for realistic test data, but seed it for determinism where it matters.
- One test class per repository/service/route trio: happy path, validation errors, not-found, foreign-key violations.
- Run tests inside the Docker container, not on the host.

---

## 5. Frontend Architecture

### 5.1 Stack

- React + TypeScript, Vite.
- **Bun** as the package manager (not npm/yarn/pnpm).
- TanStack Query (React Query) for all server state — never `useEffect` + `fetch`.
- Zod for response validation at the boundary.
- Tailwind CSS, with theme tokens defined as CSS custom properties in `index.css` and consumed via Tailwind classes. Never hardcode hex values in components.
- React Router for navigation.

### 5.2 The API client

One axios instance, default-exported from `@/lib/api`. No duplicates, no per-hook base paths. Hooks import it as `import api from "@/lib/api"` and call paths **without** an `/api` prefix (the base URL already has it).

### 5.3 Hook pattern

One file per entity at `src/hooks/useX.hooks.ts`. Every hook:
1. Defines a Zod schema for the response.
2. Uses `safeParse` and throws on failure.
3. Returns a destructured object with **prefixed keys** so multiple hooks can be used in one component without clobbering names (e.g. `athletes`, `athletesIsLoading`, `athletesError`, `athletesRefetch`).

Mutations follow the same destructured-return pattern. On success, invalidate the relevant query key.

### 5.4 Loading & error UI

Use shared `<QueryLoading />` and `<QueryError error={...} refetch={...} />` from `src/components/shared/`. Every query-driven component renders these states rather than inlining its own JSX.

### 5.5 Type safety details

- No `any`. If TS complains about implicit any in a `.map`, fix the source (usually the hook's generic).
- Use `z.infer<typeof schema>` as the canonical type for entities. Cross-file types live in `src/types/<entity>.types.ts` only if shared by more than one hook/component.

### 5.6 Frontend anti-patterns

- Defining axios instances in components or hooks.
- Hardcoded color hex values in JSX or Tailwind arbitrary values.
- Skipping Zod validation "because the backend types match" — they drift. Validate.
- Stringly-typed query keys scattered across files; lift reused keys to constants.

---

## 6. Database & Migrations

- Postgres 17 in Docker. SQLAlchemy 2.0 async with `asyncpg` as the driver.
- ORM models live under `app/models/`, one file per entity. They inherit from `app.models.base.Base`.
- Migrations are Alembic revisions in `backend/alembic/versions/`. Generate with `make revision m="..."`, apply with `make migrate`.
- New ORM models must be imported in `backend/alembic/env.py` so `--autogenerate` can detect them.
- Foreign keys are explicit; cascading deletes only when intentional and documented in the migration.
- Every migration has a working `upgrade()` *and* `downgrade()` — Alembic generates the latter, but verify it before merging.

When tests fail with foreign-key violations: the seed data is missing a parent row. Don't disable the FK — fix the seed.

---

## 7. Local Dev: Docker + Make

Everything runs in Docker via `docker-compose.yml`. The Makefile is the canonical interface — reach for `make` targets before raw `docker compose` or `pytest` calls.

Standard local ports:
- Backend API: `localhost:8000`, docs at `localhost:8000/docs`
- Postgres: `localhost:5432`
- Frontend dev server: `localhost:5173`

---

## 8. Git Workflow

- Branch per ticket: `<ticket-number>-<short-kebab-description>`.
- **Rebase, don't merge**, when integrating main. Use `--force-with-lease`, never plain `--force`.
- One PR per ticket. PR description lists what was done, what was tested, remaining TODOs.
- Code review: architecture and patterns, not formatting (that's ruff's job).

---

## 9. Working Style Expected from the Agent

1. **Read before writing.** Match existing file patterns exactly. If patterns conflict, ask which is canonical.
2. **Work in small, verifiable steps.** Implement → run → confirm → next.
3. **One recommendation, not three.** State trade-offs upfront if they matter.
4. **Say "I don't know."** Better than confidently wrong.
5. **Show only changed sections plus a few lines of context.** Preserve existing comments verbatim.
6. **Push back on bad ideas.** Don't agree to be agreeable.
7. **Prioritize current best practices (2024–2025).** Search the web to verify if uncertain.

---

## 10. New-Project Checklist

1. Scaffold the folder tree from §1.
2. Set up `pyproject.toml` with ruff config (§3).
3. Create `app/core/exceptions.py` with at least `NotFoundException`.
4. Create `app/core/db.py` with the async engine, session factory, and `get_db` dependency. Add `app/models/base.py` with the declarative `Base`.
5. Wire up Alembic (`alembic.ini`, `alembic/env.py` configured for async + `Base.metadata`).
6. Build **one** example entity end-to-end (model → schema → repo → service → route → tests → hook → component) and generate its first migration.
7. Set up `conftest.py` with the per-test session + truncate-tables cleanup fixture (§4).
8. Frontend: `lib/api.ts`, `components/shared/QueryLoading.tsx`, `components/shared/QueryError.tsx`, `index.css` with theme tokens.
9. Write the Makefile. `make up`, `make down`, `make test`, `make lint`, `make migrate` must all work.
10. Write a short `README.md` pointing at this document and the example entity.

Only after all nine are done does the team start opening tickets for actual features.
