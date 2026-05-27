"""Shared pytest fixtures for backend tests."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.db import get_db
from app.main import app

test_engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_factory() as session:
        yield session


app.dependency_overrides[get_db] = _override_get_db


# Tables to truncate before and after each test. Append entries here whenever
# a new entity is added to the test suite.
_CLEANUP_TABLES: list[str] = [
    # "papers",
]


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Per-test SQLAlchemy async session."""
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """HTTPX async client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture(autouse=True)
async def cleanup_tables() -> AsyncGenerator[None, None]:
    """Truncate test tables before and after each test."""
    await _truncate()
    yield
    await _truncate()


async def _truncate() -> None:
    if not _CLEANUP_TABLES:
        return
    async with test_engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(_CLEANUP_TABLES)} RESTART IDENTITY CASCADE"),
        )
