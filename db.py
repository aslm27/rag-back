"""Small persistence adapter for the RAG API.

When SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are configured, writes use the
Supabase PostgREST API. Without them, a JSON store keeps local development and
smoke tests functional. The SQL schema in supabase/schema.sql is the source of
truth for the Supabase tables.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from psycopg2.pool import SimpleConnectionPool
except ImportError:  # Optional for local RAG-only development.
    SimpleConnectionPool = None

from config import DATA_DIR, SUPABASE_DB_URL

_db_pool = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


class Persistence:
    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.remote = bool(self.supabase_url and self.supabase_key)
        self.state_path = DATA_DIR / "state.json"
        self.lock = threading.Lock()
        self._state: dict[str, list[dict[str, Any]]] | None = None

    @property
    def mode(self) -> str:
        return "supabase" if self.remote else "local-json"

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def _request(self, method: str, table: str, *, params: dict[str, str] | None = None, payload: Any = None) -> list[dict[str, Any]]:
        response = requests.request(
            method,
            f"{self.supabase_url}/rest/v1/{table}",
            headers=self._headers(),
            params=params,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        if not response.content:
            return []
        data = response.json()
        return data if isinstance(data, list) else [data]

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if self._state is None:
            if self.state_path.exists():
                try:
                    self._state = json.loads(self.state_path.read_text())
                except (json.JSONDecodeError, OSError):
                    self._state = {}
            else:
                self._state = {}
        for table in ("projects", "documents", "ingestion_jobs", "conversations", "messages", "retrieval_runs", "retrieval_chunks", "evaluations", "pipeline_logs"):
            self._state.setdefault(table, [])
        return self._state

    def _save(self) -> None:
        assert self._state is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._state, indent=2, default=str))
        temp.replace(self.state_path)

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        row = {**row}
        row.setdefault("id", new_id())
        row.setdefault("created_at", utcnow())
        row.setdefault("updated_at", utcnow())
        if self.remote:
            rows = self._request("POST", table, payload=row)
            return rows[0] if rows else row
        with self.lock:
            state = self._load()
            state[table].append(row)
            self._save()
        return row

    def update(self, table: str, row_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        changes = {**changes, "updated_at": utcnow()}
        if self.remote:
            rows = self._request("PATCH", table, params={"id": f"eq.{row_id}"}, payload=changes)
            return rows[0] if rows else None
        with self.lock:
            state = self._load()
            for row in state[table]:
                if str(row.get("id")) == str(row_id):
                    row.update(changes)
                    self._save()
                    return row
        return None

    def get(self, table: str, row_id: str) -> dict[str, Any] | None:
        if self.remote:
            rows = self._request("GET", table, params={"id": f"eq.{row_id}", "limit": "1"})
            return rows[0] if rows else None
        with self.lock:
            return next((row for row in self._load()[table] if str(row.get("id")) == str(row_id)), None)

    def list(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        if self.remote:
            params = {key: f"eq.{value}" for key, value in filters.items()}
            params["order"] = "created_at.asc"
            return self._request("GET", table, params=params)
        with self.lock:
            rows = list(self._load()[table])
        return [row for row in rows if all(str(row.get(k)) == str(v) for k, v in filters.items())]

    def log_pipeline(self, *, request_id: str, stage: str, status: str = "ok", project_id: str | None = None, conversation_id: str | None = None, message_id: str | None = None, details: dict[str, Any] | None = None, latency_ms: int | None = None) -> dict[str, Any]:
        return self.insert("pipeline_logs", {
            "request_id": request_id,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "stage": stage,
            "status": status,
            "latency_ms": latency_ms,
            "details": details or {},
        })


store = Persistence()


def init_db_pool() -> None:
    """Initialize the optional direct PostgreSQL pool used by auth routes."""
    global _db_pool
    if _db_pool is not None:
        return
    if not SUPABASE_DB_URL:
        return
    if SimpleConnectionPool is None:
        raise RuntimeError("psycopg2-binary is required when SUPABASE_DB_URL is configured.")
    _db_pool = SimpleConnectionPool(minconn=1, maxconn=8, dsn=SUPABASE_DB_URL)


def close_db_pool() -> None:
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None


@contextmanager
def get_db_connection():
    """Yield a pooled direct PostgreSQL connection for authentication queries."""
    if _db_pool is None:
        raise RuntimeError("SUPABASE_DB_URL is not configured.")
    connection = _db_pool.getconn()
    try:
        yield connection
    finally:
        _db_pool.putconn(connection)
