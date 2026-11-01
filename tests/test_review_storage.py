"""Real PostgreSQL checks. Only per-test UUID resources are created/deleted."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest

from xiaoying.domain import AppError, LeaseLost
from xiaoying.config import Settings, get_test_database_url
from xiaoying.storage.db import Database, create_isolated_database, event, fence, idempotent, migrate

pytestmark = pytest.mark.integration


def new_id():
    return str(uuid4())


@pytest.fixture(scope="module")
def db():
    url = get_test_database_url()
    try:
        create_isolated_database(url, Settings().database_url)
        with psycopg.connect(url, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL integration database unavailable: {exc.__class__.__name__}")
    migrate(url)
    database = Database(url)
    yield database
    database.close()


@pytest.fixture
def resources(db):
    ids = {name: new_id() for name in ("kb", "version", "conversation", "run", "job")}
    with db.tx() as conn:
        conn.execute("INSERT INTO knowledge_bases(id,name) VALUES(%s,'review isolated')", (ids["kb"],))
        conn.execute(
            "INSERT INTO kb_versions(id,kb_id,status,config,manifest_hash) VALUES(%s,%s,'ready','{}','review')",
            (ids["version"], ids["kb"]),
        )
        conn.execute("INSERT INTO conversations(id,kb_id) VALUES(%s,%s)", (ids["conversation"], ids["kb"]))
        conn.execute(
            "INSERT INTO runs(id,conversation_id,kb_id,kb_version_id,question,config) VALUES(%s,%s,%s,%s,'review','{}')",
            (ids["run"], ids["conversation"], ids["kb"], ids["version"]),
        )
        conn.execute(
            "INSERT INTO jobs(id,type,resource_id,status,lease_token,lease_until) "
            "VALUES(%s,'run',%s,'running',7,clock_timestamp()+interval '1 minute')",
            (ids["job"], ids["run"]),
        )
    yield ids
    with db.tx() as conn:
        conn.execute("DELETE FROM jobs WHERE id=%s", (ids["job"],))
        for table in ("trace_events", "evidence_reads", "run_actions", "clarifications", "messages"):
            conn.execute(f"DELETE FROM {table} WHERE run_id=%s", (ids["run"],))
        conn.execute("DELETE FROM runs WHERE id=%s", (ids["run"],))
        conn.execute("DELETE FROM conversations WHERE id=%s", (ids["conversation"],))
        conn.execute("UPDATE knowledge_bases SET current_version_id=NULL WHERE id=%s", (ids["kb"],))
        conn.execute("DELETE FROM kb_versions WHERE id=%s", (ids["version"],))
        conn.execute("DELETE FROM knowledge_bases WHERE id=%s", (ids["kb"],))


def test_idempotency_serializes_concurrent_requests(db):
    scope = f"review:{new_id()}"
    executions = []

    def request(_):
        with db.tx() as conn:
            def operation():
                executions.append(1)
                return {"resource_id": new_id()}
            return idempotent(conn, scope, "same", {"question": "上海住宿"}, operation)

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(request, range(6)))
        assert len(executions) == 1
        assert all(result == results[0] for result in results)
        with db.tx() as conn, pytest.raises(AppError) as error:
            idempotent(conn, scope, "same", {"question": "不同问题"}, lambda: {"wrong": True})
        assert error.value.status == 409
        assert error.value.code == "IDEMPOTENCY_CONFLICT"
    finally:
        with db.tx() as conn:
            conn.execute("DELETE FROM idempotency_requests WHERE scope=%s", (scope,))


def test_stale_and_expired_workers_cannot_pass_fence(db, resources):
    with db.tx() as conn:
        assert fence(conn, resources["job"], 7)["lease_token"] == 7
        with pytest.raises(LeaseLost):
            fence(conn, resources["job"], 6)
        conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (resources["job"],))
        with pytest.raises(LeaseLost):
            fence(conn, resources["job"], 7)


def test_completed_job_cannot_write_with_old_valid_token(db, resources):
    with db.tx() as conn:
        conn.execute("UPDATE jobs SET status='done' WHERE id=%s", (resources["job"],))
        with pytest.raises(LeaseLost):
            fence(conn, resources["job"], 7)


def test_same_conversation_rejects_second_active_run(db, resources):
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO runs(id,conversation_id,kb_id,kb_version_id,question,config) VALUES(%s,%s,%s,%s,'second','{}')",
                (new_id(), resources["conversation"], resources["kb"], resources["version"]),
            )


def test_version_from_another_knowledge_base_is_rejected(db, resources):
    # Transaction rolls back the second KB alongside the rejected cross-KB pointer.
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.tx() as conn:
            other_kb = new_id()
            conn.execute("INSERT INTO knowledge_bases(id,name) VALUES(%s,'review cross-kb')", (other_kb,))
            conn.execute("UPDATE knowledge_bases SET current_version_id=%s WHERE id=%s", (resources["version"], other_kb))


def test_events_concurrently_deduplicate_without_sequence_gaps(db, resources):
    def append(number):
        with db.tx() as conn:
            event(conn, resources["run"], f"review:{number}", "agent.decision", {"number": number})

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(append, [0, 1, 2, 0, 3, 1, 4, 5, 2]))
    events = db.all("SELECT seq,event_id,payload FROM trace_events WHERE run_id=%s ORDER BY seq", (resources["run"],))
    assert [item["seq"] for item in events] == [1, 2, 3, 4, 5, 6]
    assert {item["payload"]["number"] for item in events} == set(range(6))
    assert db.one("SELECT next_event_seq FROM runs WHERE id=%s", (resources["run"],))["next_event_seq"] == 6


def test_clarification_count_cannot_exceed_one(db, resources):
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.tx() as conn:
            conn.execute("UPDATE runs SET clarification_count=2 WHERE id=%s", (resources["run"],))
