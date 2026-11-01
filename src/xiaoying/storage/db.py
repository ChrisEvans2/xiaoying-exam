import json
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from xiaoying.config import require_separate_database
from xiaoying.domain import AppError, LeaseLost, digest


def uid():
    return str(uuid4())


def jsonable(value):
    return json.loads(
        json.dumps(value, default=lambda v: v.isoformat() if isinstance(v, (date, datetime)) else str(v))
    )


class Database:
    def __init__(self, url, max_size=12):
        self.url = url
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            open=True,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "options": "-c lock_timeout=2000 -c statement_timeout=10000 -c idle_in_transaction_session_timeout=10000",
            },
        )
        self.pool.wait(timeout=10)

    @contextmanager
    def tx(self):
        with self.pool.connection() as conn, conn.transaction():
            yield conn

    def one(self, sql, args=()):
        with self.pool.connection() as conn:
            row = conn.execute(sql, args).fetchone()
            return jsonable(row) if row else None

    def all(self, sql, args=()):
        with self.pool.connection() as conn:
            return jsonable(conn.execute(sql, args).fetchall())

    def close(self):
        self.pool.close()


def migrate(url):
    from langgraph.checkpoint.postgres import PostgresSaver

    root = Path(__file__).resolve().parents[3]
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations(version text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(831026)")
            for path in sorted((root / "migrations").glob("*.sql")):
                if not conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=%s", (path.name,)
                ).fetchone():
                    conn.execute(path.read_text())
                    conn.execute("INSERT INTO schema_migrations(version) VALUES(%s)", (path.name,))
        PostgresSaver(conn).setup()


def create_isolated_database(url, business_url):
    """Provision only a separate mock/test database; retain every existing database."""
    require_separate_database(url, business_url)
    name = conninfo_to_dict(url)["dbname"]
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True, connect_timeout=5) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone():
            try:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            except psycopg.errors.DuplicateDatabase:
                pass


def require(row, message="资源不存在"):
    if row is None:
        raise AppError("NOT_FOUND", message, 404)
    return row


def valid_id(value):
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        raise AppError("INVALID_ID", "无效的资源 ID", 422) from None


def fence(conn, job_id, token):
    row = conn.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job_id,)).fetchone()
    now = conn.execute("SELECT clock_timestamp() AS t").fetchone()["t"]
    if not row or row["status"] != "running" or row["lease_token"] != token or row["lease_until"] <= now:
        raise LeaseLost()
    return row


def event(conn, run_id, event_id, kind, payload):
    # Run-row locking assigns sequence numbers and deduplicates retries in one transaction.
    conn.execute("SELECT id FROM runs WHERE id=%s FOR UPDATE", (run_id,))
    if conn.execute(
        "SELECT 1 FROM trace_events WHERE run_id=%s AND event_id=%s", (run_id, event_id)
    ).fetchone():
        return
    seq = conn.execute(
        "UPDATE runs SET next_event_seq=next_event_seq+1 WHERE id=%s RETURNING next_event_seq", (run_id,)
    ).fetchone()["next_event_seq"]
    conn.execute(
        "INSERT INTO trace_events(run_id,seq,event_id,type,payload) VALUES(%s,%s,%s,%s,%s)",
        (run_id, seq, event_id, kind, Jsonb(jsonable(payload))),
    )


def idempotent(conn, scope, key, body, operation):
    if not key or len(key) > 160:
        raise AppError("IDEMPOTENCY_REQUIRED", "写操作需要长度 1—160 的 Idempotency-Key", 422)
    request_hash = digest(body)
    conn.execute(
        "INSERT INTO idempotency_requests(scope,key,request_hash) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
        (scope, key, request_hash),
    )
    row = conn.execute(
        "SELECT * FROM idempotency_requests WHERE scope=%s AND key=%s FOR UPDATE", (scope, key)
    ).fetchone()
    if row["request_hash"] != request_hash:
        raise AppError("IDEMPOTENCY_CONFLICT", "同一个幂等键不能用于不同内容", 409)
    if row["response"] is not None:
        return row["response"]
    response = jsonable(operation())
    conn.execute(
        "UPDATE idempotency_requests SET response=%s WHERE scope=%s AND key=%s", (Jsonb(response), scope, key)
    )
    return response
