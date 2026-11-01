"""Cross-worker queue limits and the job-specific CLI path use the same scheduling rules."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from xiaoying.cli import claim_specific
from xiaoying.providers.embeddings import MockReranker
from xiaoying.services.runs import RunService
from xiaoying.worker import Worker

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_storage import db as database_fixture

knowledge = knowledge_fixture
db = database_fixture
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind,limit", [("run", 2), ("build", 1)])
def test_concurrent_workers_share_global_slots(knowledge, kind, limit):
    h = knowledge
    if h.db.one("SELECT count(*) n FROM jobs WHERE type=%s AND status='running' AND lease_until>clock_timestamp()", (kind,))["n"]:
        pytest.skip("Another process currently owns global slots; do not alter its jobs")
    kb = h.create()
    version = h.initial(kb)
    if kind == "build":
        requests = [h.start(kb, version, auto_publish=False) for _ in range(4)]
    else:
        runs = RunService(h.db, h.settings, h.embedder)
        conversations = [runs.create_conversation(kb, h.key())["id"] for _ in range(4)]
        requests = [runs.create_run(cid, "review scheduler", h.key()) for cid in conversations]
    ids = [request["job_id"] for request in requests]
    with h.db.tx() as conn:
        # Select only our jobs before any unrelated pending queue entries.
        conn.execute("UPDATE jobs SET created_at='1900-01-01' WHERE id=ANY(%s::uuid[])", (ids,))
    workers = [Worker(h.db, h.settings, h.embedder, MockReranker()) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        claimed = list(executor.map(lambda worker: worker.claim(kind), workers))
    claimed = [job for job in claimed if job is not None]
    assert len(claimed) == limit
    assert len({job["id"] for job in claimed}) == limit
    assert {job["id"] for job in claimed} <= set(ids)
    remaining = next(identifier for identifier in ids if identifier not in {job["id"] for job in claimed})
    assert claim_specific(h.db, h.settings, remaining) is None


@pytest.mark.parametrize("setting", ["attempts", "backoff"])
def test_specific_job_entrypoint_cannot_bypass_retry_schedule(knowledge, setting):
    h = knowledge
    kb = h.create()
    version = h.initial(kb)
    request = h.start(kb, version, auto_publish=False)
    with h.db.tx() as conn:
        if setting == "attempts":
            conn.execute("UPDATE jobs SET attempts=max_attempts WHERE id=%s", (request["job_id"],))
        else:
            conn.execute("UPDATE jobs SET next_attempt_at=clock_timestamp()+interval '1 hour' WHERE id=%s", (request["job_id"],))
    assert claim_specific(h.db, h.settings, request["job_id"]) is None
