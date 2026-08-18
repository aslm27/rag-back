"""PostgreSQL connection management for Supabase."""
from contextlib import contextmanager

from psycopg2.pool import SimpleConnectionPool

from config import SUPABASE_DB_URL


_connection_pool: SimpleConnectionPool | None = None


def init_db_pool() -> None:
    """Initialize DB pool once at startup."""
    global _connection_pool
    if _connection_pool is not None:
        return
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not configured.")
    _connection_pool = SimpleConnectionPool(minconn=1, maxconn=8, dsn=SUPABASE_DB_URL)


def close_db_pool() -> None:
    """Close DB pool on shutdown."""
    global _connection_pool
    if _connection_pool is None:
        return
    _connection_pool.closeall()
    _connection_pool = None


@contextmanager
def get_db_connection():
    """Yield a pooled DB connection and always return it."""
    if _connection_pool is None:
        raise RuntimeError("Database pool is not initialized.")

    conn = _connection_pool.getconn()
    try:
        yield conn
    finally:
        _connection_pool.putconn(conn)
