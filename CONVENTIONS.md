# Coding Practices & Repository Conventions

This document captures the conventions for **TopicDrift**, a data-analysis project for CS4530. It's adapted from a production FastAPI + React + Postgres template, with updates for current best practices (early 2026) and this project's purpose (paper-metadata ingestion + topic clustering, not a typical CRUD app).

Treat everything here as **defaults, not dogma** — but deviate only with a stated reason.

> See `DEPENDENCIES.md` for host-tool install instructions and `README.md` for the day-to-day commands.

---

## 0. Stack Summary (2026)

| Layer | Choice | Why |
|---|---|---|
| Python | 3.13 | Latest stable; matches `target-version = "py313"`. |
| Python package manager | **uv** 0.5+ | Single tool: lockfile + venv + run. Replaces pip/pip-tools/virtualenv. |
| Backend framework | FastAPI | Async, Pydantic-native, typed routes. |
| ORM | SQLAlchemy 2.0 async + asyncpg | Typed `Mapped[T]` style; only modern option. |
| Migrations | Alembic (async template) | Standard with SQLAlchemy. |
| Python linter/formatter | Ruff (one tool) | Replaces black/isort/flake8/pyupgrade. `ruff format` is stable. |
| DB | Postgres 17 (Docker) | Plain Postgres — no Supabase. |
| JS package manager | **Bun** 1.2+ | Fast installs, text-based `bun.lock`. |
| Frontend framework | React 19 + TypeScript | `ref` as prop (no `forwardRef`); strict mode unchanged. |
| Build tool | Vite 6 | No-op upgrade from v5 for SPAs. |
| Styling | Tailwind CSS v4 | CSS-first `@theme {}` config, no `tailwind.config.js` or PostCSS. |
| Server state | TanStack Query v5 | Single-object hook signature; `gcTime` (was `cacheTime`). |
| Schema validation | Zod 4 | Faster, smaller; new top-level helpers (`z.email()`). |
| Routing | React Router v6 | v7 exists but v6 is stable enough for this project. |
| Orchestration | Docker Compose v2 + Make | One canonical interface (`make up`, `make test`, ...). |

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
│   │   ├── pipeline/          # Data ingestion + analysis (NOT entity-scoped, see §11)
│   │   └── main.py            # FastAPI app + lifespan + router registration
│   ├── alembic/               # Async migration env + versioned scripts
│   ├── alembic.ini
│   ├── tests/
│   │   ├── conftest.py        # async fixtures (httpx client, db session, truncate)
│   │   └── tests/             # actual test files, suffixed `_tests.py`
│   ├── pyproject.toml         # uv project + ruff + pytest config
│   ├── uv.lock                # COMMITTED
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── lib/               # api client (axios), utilities
│   │   ├── hooks/             # one `useX.hooks.ts` per entity/concern
│   │   ├── components/
│   │   │   └── shared/        # QueryLoading, QueryError, etc.
│   │   ├── pages/             # route-level components
│   │   ├── types/             # cross-file TS types
│   │   ├── index.css          # Tailwind v4 @import + @theme tokens
│   │   └── App.tsx            # router
│   ├── package.json           # Bun
│   ├── bun.lock               # COMMITTED (text format, v1.2+)
│   └── tsconfig.json
├── scripts/                   # one-off setup SQL (test DB init, etc.)
├── docker-compose.yml         # backend + postgres
├── Makefile                   # canonical interface — `make help` for targets
├── DEPENDENCIES.md
├── CONVENTIONS.md
└── README.md
```

**Entity rule:** Every CRUD-style entity (Paper, Conference, Topic, …) has exactly five files on the backend — `models/`, `schemas/`, `repositories/`, `services/`, `routes/` — and one hook file on the frontend (`useX.hooks.ts`). No exceptions for entities.

**Pipeline rule:** Data ingestion + analysis code is **not** entity-scoped and lives in `app/pipeline/` (see §11). This is the only sanctioned exception to "no new folders."

---

## 2. Backend Architecture — The Three-Layer Pattern

Every request flows: **route → service → repository → DB**, and the response flows back the same way.

### 2.1 Models (SQLAlchemy ORM)

- One file per entity in `app/models/`. Each file defines exactly one model class.
- Inherit from `app.models.base.Base`.
- Use the modern `Mapped[T]` + `mapped_column()` style. Don't use the legacy `Column()` declarative.
- Models are imported in `backend/alembic/env.py` so Alembic autogenerate can detect them.

```python
# app/models/paper_model.py
from datetime import datetime
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class Paper(Base):
    """A conference paper."""

    __tablename__ = "papers"

    paper_id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    abstract: Mapped[str | None]
    year: Mapped[int]
    venue: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
```

### 2.2 Schemas (Pydantic v2)

Pydantic schemas are used **at every layer boundary**.

- `EntityCreate` — fields needed to create the row (no `id`, no `created_at`).
- `EntityUpdate` — all fields optional (`field: T | None = None`).
- `EntityResponse` — what gets returned to the client. Use `model_config = ConfigDict(from_attributes=True)` so a SQLAlchemy ORM object can be passed directly to `Schema.model_validate(orm_obj)`.

### 2.3 Repositories

- **Only layer that touches the database.**
- Takes an `AsyncSession` in `__init__` (injected via the `get_db` FastAPI dependency).
- Owns the SQLAlchemy queries and ORM → Pydantic conversion. Returns Pydantic objects, never ORM instances or raw rows.
- Raises domain exceptions (`NotFoundException`) — never lets DB errors bubble up raw.
- Every public method has a docstring and logs `Repository: <action>`.

### 2.4 Services

- Business logic only. **No DB access** — depends on a repository injected in `__init__`.
- Takes Pydantic input and returns Pydantic output.
- Logs `Service: <action>`.
- If logic is trivial (pure pass-through to the repo), the service is still required — consistency beats brevity.

### 2.5 Routes

- Thin HTTP layer. Validates input via Pydantic, calls a service, returns the result.
- Wire dependencies with `Annotated[T, Depends(...)]` (matches Ruff's `FAST002` rule and FastAPI's modern style):

```python
from typing import Annotated
from fastapi import APIRouter, Depends, status

PaperServiceDep = Annotated[PaperService, Depends(get_paper_service)]

@router.post("", response_model=PaperResponse, status_code=status.HTTP_201_CREATED)
async def create_paper(data: PaperCreate, service: PaperServiceDep) -> PaperResponse:
    """Create a new paper."""
    return await service.create_paper(data)
```

- Set explicit `response_model` and `status_code` on every endpoint.
- Logs `Route: <METHOD /path>`.

### 2.6 Anti-patterns to refuse

- Repositories returning raw ORM instances or `dict`s.
- Services importing from `sqlalchemy` or holding an `AsyncSession` directly.
- Routes containing business logic beyond `await service.foo()`.
- Defining ORM models inside `repositories/` files — models live in `models/`.
- A "utils" or "helpers" folder used as a dumping ground.
- Skipping the service layer "because it's just a pass-through."
- The old `x: int = Depends(...)` style — use `Annotated`.

---

## 3. Backend Linting & Formatting (Ruff)

Ruff is the only Python linter/formatter. No black/isort/flake8/pyupgrade.

Rule set lives in `backend/pyproject.toml`:

```
E, W, F, I, UP, B, SIM, C4, PTH, RET, TC, FAST, ASYNC, ANN, D
```

`FAST` covers FastAPI-specific lints (Annotated deps). `ASYNC` flags blocking calls in async code (e.g., `time.sleep` inside an `async def`). `PTH` nudges off `os.path` toward `pathlib`.

Run before every commit:

```bash
make format   # ruff check --fix && ruff format
make lint     # ruff check (read-only) + bun lint
```

---

## 4. Backend Testing

- pytest, with `pytest-asyncio` in `auto` mode.
- Test files live in `tests/tests/` and are named `<entity>_tests.py` (not `test_<entity>.py`).
- `conftest.py` provides an async `httpx.AsyncClient` wired via `ASGITransport`, a per-test `AsyncSession` fixture, and an `autouse` `cleanup_tables` fixture that `TRUNCATE`s test tables before and after each test.
- **Cleanup must include every table touched by tests.** Append to `_CLEANUP_TABLES` in `conftest.py` in the same PR that adds the entity.
- Use `Faker` for realistic test data; seed it for determinism where it matters.
- One test class per repository/service/route trio: happy path, validation errors, not-found, foreign-key violations.
- Run inside the Docker container:

```bash
make test
```

---

## 5. Frontend Architecture

### 5.1 Tailwind v4 — CSS-first config

There is **no** `tailwind.config.js` and **no** `postcss.config.js`. Tailwind v4 is wired in via the Vite plugin and configured directly in CSS:

```css
/* src/index.css */
@import "tailwindcss";

@theme {
  --color-background: hsl(0 0% 100%);
  --color-foreground: hsl(222 47% 11%);
  /* ... */
}
```

Theme tokens defined in `@theme` become utility classes automatically (`bg-background`, `text-foreground`). **Never hardcode hex values** — always go through tokens. Adding a new color = adding a `--color-*` line in `@theme`, nothing else.

Do **not** use the v3 `@tailwind base/components/utilities` triple — that's removed.

### 5.2 The API client

One axios instance, default-exported from `@/lib/api`. No duplicates, no per-hook base paths.

Hooks call paths **without** an `/api` prefix — the base URL already has it (`VITE_API_URL=http://localhost:8000/api`).

### 5.3 Hook pattern

One file per entity at `src/hooks/useX.hooks.ts`. Every query hook:

1. Defines a Zod schema for the response.
2. Uses `safeParse` and throws on failure (TanStack Query surfaces the error).
3. Returns a destructured object with **prefixed keys** so multiple hooks can be used in one component without name collisions.

```ts
import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import api from "@/lib/api";

const paperSchema = z.object({
  paper_id: z.number(),
  title: z.string(),
  abstract: z.string().nullable(),
  year: z.number(),
  venue: z.string(),
});

export type Paper = z.infer<typeof paperSchema>;

export function useGetAllPapers() {
  const query = useQuery({
    queryKey: ["papers"],
    queryFn: async () => {
      const response = await api.get("/papers");
      const parsed = z.array(paperSchema).safeParse(response.data);
      if (!parsed.success) throw new Error("Invalid response format");
      return parsed.data;
    },
  });

  return {
    papers: query.data ?? [],
    papersIsLoading: query.isLoading,
    papersError: query.error,
    papersRefetch: query.refetch,
  };
}
```

**Mutations** follow the same destructured-return pattern with prefixed names (`createPaper`, `createPaperIsLoading`, `createPaperError`). On success, invalidate the relevant `queryKey`.

**Zod 4 reminders:** prefer top-level helpers (`z.email()`, `z.uuid()`, `z.url()`) over `z.string().email()`. Use `z.flattenError(err)` / `z.treeifyError(err)` instead of the old `err.flatten()` / `err.format()`.

### 5.4 Loading & error UI

Use the shared `<QueryLoading />` and `<QueryError error={...} refetch={...} />` from `src/components/shared/`. Every query-driven component renders these states rather than inlining its own JSX.

### 5.5 React 19 conventions

- **No `forwardRef`.** `ref` is just a regular prop on function components now. Type it as `ref?: React.Ref<HTMLDivElement>` (or whichever element).
- **No `defaultProps` on function components** (removed). Use default parameter values.
- `ref` callbacks may return a cleanup function — useful for measuring DOM nodes without a follow-up `useEffect`.

### 5.6 Type safety details

- No `any`. If TS complains about implicit any in a `.map`, fix the source (usually the hook's generic).
- Use `z.infer<typeof schema>` as the canonical type for entities. Cross-file types live in `src/types/<entity>.types.ts` only if shared by more than one hook/component.

### 5.7 Frontend anti-patterns

- Defining axios instances in components or hooks.
- Hardcoded color values, or Tailwind arbitrary values like `bg-[#E8724A]`.
- Skipping Zod validation "because the backend types match" — they drift.
- Stringly-typed query keys scattered across files; lift reused keys to constants.
- Using `forwardRef` or `defaultProps` (both phased out in React 19).
- Adding `tailwind.config.js` or `postcss.config.js` — v4 doesn't need them.

---

## 6. Database & Migrations

- Postgres 17 in Docker. SQLAlchemy 2.0 async with `asyncpg`.
- ORM models live under `app/models/`, one file per entity, inheriting from `app.models.base.Base`.
- Migrations are Alembic revisions in `backend/alembic/versions/`. Generate with `make revision m="..."`, apply with `make migrate`.
- **New ORM models must be imported in `backend/alembic/env.py`** so `--autogenerate` can detect them.
- The async env.py committed here was derived from `alembic init -t async`. If regenerating, use that template — don't hand-write.
- Foreign keys are explicit; cascading deletes only when intentional and documented in the migration.
- Every migration has a working `upgrade()` *and* `downgrade()` — Alembic generates the latter, but verify it before merging.
- `session_factory` uses `expire_on_commit=False` (essentially mandatory for async to avoid implicit IO after commit).

When tests fail with foreign-key violations: the seed data is missing a parent row. Don't disable the FK — fix the seed.

---

## 7. Local Dev: Docker, uv & Make

Everything runs in Docker via `docker-compose.yml`. The Makefile is the canonical interface — reach for `make` targets before raw `docker compose` / `uv` / `bun` calls.

### Dockerfile pattern (backend)

Follows Astral's official uv + Docker guide. Key points:

- Multi-stage copy: pull `uv` from `ghcr.io/astral-sh/uv:0.5` (pinned minor) into a `python:3.13-slim` base.
- `uv sync --no-install-project` first, before copying source — gives Docker a stable layer cache for deps.
- `.venv` lives at `/app/.venv` and is added to `PATH`.
- **Anonymous volume** for `/app/.venv` in `docker-compose.yml` so the bind-mount of `./backend:/app` doesn't mask the installed deps.

### Standard local ports

- Backend API: `localhost:8000`, docs at `localhost:8000/docs`
- Postgres: `localhost:5432`
- Frontend dev server: `localhost:5173`

### uv quick reference

```bash
cd backend
uv lock                 # regenerate uv.lock
uv add httpx            # add a runtime dep
uv add --group dev pytest-mock   # add a dev dep
uv run pytest           # run a command in the project env
```

`uv.lock` is committed. Don't commit `.venv/`.

### Bun quick reference

```bash
cd frontend
bun install             # install from bun.lock
bun add @tanstack/react-router        # add a runtime dep
bun add -d @types/node                # add a dev dep
bun run dev             # vite dev server
```

`bun.lock` is committed (text format, Bun 1.2+). Don't commit `node_modules/`.

---

## 8. Git Workflow

- Branch per ticket: `<ticket-number>-<short-kebab-description>`.
- **Rebase, don't merge**, when integrating main. Use `--force-with-lease`, never plain `--force`.
- One PR per ticket. Description lists what was done, what was tested, remaining TODOs.
- Code review: architecture and patterns, not formatting (that's Ruff's job).

---

## 9. Working Style Expected from the Agent

1. **Read before writing.** Match existing file patterns exactly. If patterns conflict, ask which is canonical.
2. **Work in small, verifiable steps.** Implement → run → confirm → next.
3. **One recommendation, not three.** State trade-offs upfront if they matter.
4. **Say "I don't know."** Better than confidently wrong.
5. **Show only changed sections plus a few lines of context.** Preserve existing comments verbatim.
6. **Push back on bad ideas.** Don't agree to be agreeable.
7. **Prioritize current best practices.** Search the web to verify if uncertain — this stack moves fast.

---

## 10. New-Project Checklist

1. Scaffold the folder tree from §1. ✓
2. `pyproject.toml` with uv project config + ruff + pytest. ✓
3. `app/core/exceptions.py` with at least `NotFoundException`. ✓
4. `app/core/db.py` with async engine, session factory, `get_db` dependency. `app/models/base.py` with `Base`. ✓
5. Alembic async env.py + `alembic.ini` (use `alembic init -t async` template). ✓
6. Build **one** example entity end-to-end (model → schema → repo → service → route → tests → hook → component) and generate its first migration. ← **next**
7. `conftest.py` with per-test session + truncate-tables cleanup. ✓
8. Frontend: `lib/api.ts`, `components/shared/{QueryLoading,QueryError}.tsx`, `index.css` with Tailwind v4 `@theme`. ✓
9. Makefile with `check-deps`, `bootstrap`, `up`, `down`, `test`, `lint`, `format`, `migrate`. ✓
10. `README.md` + `DEPENDENCIES.md` + `CONVENTIONS.md`. ✓

Only after step 6 does the team start opening tickets for actual features.

---

## 11. Data Pipeline (TopicDrift-Specific)

TopicDrift is a data-analysis project. The bulk of the work is:
1. **Ingest** paper metadata from DBLP, Semantic Scholar, or the ACM Digital Library.
2. **Analyze** titles + abstracts (TF-IDF, topic clustering).
3. **Visualize** topic prevalence over time.

This work is **not entity-scoped**, so it doesn't fit the model/schema/repo/service/route pattern. It lives in `app/pipeline/`:

```
app/pipeline/
├── ingest/         # source-specific clients: dblp.py, semantic_scholar.py
├── analyze/        # tfidf.py, clustering.py
└── jobs/           # CLI entrypoints invoked via `uv run python -m app.pipeline.jobs.ingest_icse`
```

**Rules for pipeline code:**

- Pipeline code **may** read/write the DB, but **must go through repositories**. No raw SQLAlchemy in `pipeline/`.
- Long-running jobs run as CLI entrypoints (`python -m app.pipeline.jobs.foo`), not as API requests. Add a `make ingest CONF=icse` target rather than a route.
- Heavy ML libraries (scikit-learn, spaCy, NLTK) are **runtime** deps in `pyproject.toml` only if they're used by request-time code. If only the pipeline uses them, put them in a separate `[dependency-groups]` entry (e.g., `analysis`) so the API container doesn't carry them.
- Analysis **results** (topic timelines, cluster summaries) become entities — those go through the normal five-file pattern.

When in doubt: the API serves *queries against pre-computed results*. The pipeline computes those results offline.

---

*This document is a snapshot of conventions; revise it when the team decides to change them, not when an individual disagrees mid-ticket.*
