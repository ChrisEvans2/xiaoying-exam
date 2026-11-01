"""Real PG/pgvector lifecycle with explicitly deterministic embedding/rerank adapters."""

from uuid import uuid4

import pytest

from xiaoying.config import Settings
from xiaoying.domain import AppError, BuildRequest, LeaseLost
from xiaoying.providers.embeddings import MockEmbedder, MockReranker
from xiaoying.rag.retrieval import Retriever, reciprocal_rank_fusion
from xiaoying.services.knowledge import KnowledgeService

from test_review_storage import db as database_fixture

db = database_fixture

pytestmark = pytest.mark.integration


class Harness:
    def __init__(self, database, path):
        self.db = database
        self.prefix = f"review-{uuid4()}-"
        self.counter = 0
        self.kbs = []
        self.settings = Settings(
            _env_file=None, provider_mode="mock", pdf_backend="test_text", data_dir=path,
            database_url=database.url,
        )
        self.embedder = MockEmbedder()
        self.service = KnowledgeService(database, self.settings, self.embedder)

    def key(self):
        self.counter += 1
        return self.prefix + str(self.counter)

    def create(self):
        result = self.service.create("review isolated knowledge", self.key())
        self.kbs.append(result["id"])
        return result["id"]

    def upload(self, kb, doc="DREV", amount=500):
        text = (
            f"住宿标准\n文档 ID：{doc}\n版本日期：2026-10-31\n适用范围：中国大陆境内差旅\n\n"
            f"1. 上海住宿费用每人每晚最高{amount}元。\n2. 费用应当提供真实发票。\n"
        )
        return self.service.upload(kb, f"{doc}.txt", text.encode(), {}, self.key())["upload_id"]

    def start(self, kb, base=None, changes=None, auto_publish=True):
        return self.service.create_build(
            kb, BuildRequest(base_version_id=base, changes=changes or [], auto_publish=auto_publish), self.key()
        )

    def claim(self, result):
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,lease_token=lease_token+1,"
                "lease_until=clock_timestamp()+interval '5 minutes' WHERE id=%s",
                (result["job_id"],),
            )
        return self.db.one("SELECT * FROM jobs WHERE id=%s", (result["job_id"],))

    def build(self, result):
        self.service.build(self.claim(result))
        return result["version_id"]

    def initial(self, kb, amount=500):
        upload = self.upload(kb, amount=amount)
        return self.build(self.start(kb, changes=[{"action": "add", "document_id": "DREV", "upload_id": upload}]))

    def cleanup(self):
        with self.db.tx() as conn:
            for kb in self.kbs:
                versions = "SELECT id FROM kb_versions WHERE kb_id=%s"
                run_ids = conn.execute("SELECT id FROM runs WHERE kb_id=%s", (kb,)).fetchall()
                for run in run_ids:
                    rid = str(run["id"])
                    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                        conn.execute(f"DELETE FROM {table} WHERE thread_id=%s", (rid,))
                    for table in ("trace_events", "evidence_reads", "run_actions", "clarifications", "messages"):
                        conn.execute(f"DELETE FROM {table} WHERE run_id=%s", (rid,))
                    conn.execute("DELETE FROM jobs WHERE type='run' AND resource_id=%s", (rid,))
                conn.execute("DELETE FROM runs WHERE kb_id=%s", (kb,))
                conn.execute("DELETE FROM conversations WHERE kb_id=%s", (kb,))
                conn.execute(f"DELETE FROM jobs WHERE resource_id IN ({versions})", (kb,))
                conn.execute(f"DELETE FROM chunks WHERE kb_version_id IN ({versions})", (kb,))
                conn.execute(f"DELETE FROM source_blocks WHERE kb_version_id IN ({versions})", (kb,))
                conn.execute("DELETE FROM kb_version_documents WHERE kb_id=%s", (kb,))
                conn.execute("UPDATE knowledge_bases SET current_version_id=NULL WHERE id=%s", (kb,))
                conn.execute("DELETE FROM kb_versions WHERE kb_id=%s", (kb,))
                conn.execute("DELETE FROM document_versions WHERE kb_id=%s", (kb,))
                conn.execute("DELETE FROM documents WHERE kb_id=%s", (kb,))
                conn.execute("DELETE FROM uploads WHERE kb_id=%s", (kb,))
                conn.execute("DELETE FROM knowledge_bases WHERE id=%s", (kb,))
            conn.execute("DELETE FROM idempotency_requests WHERE key LIKE %s", (self.prefix + "%",))


@pytest.fixture
def knowledge(db, tmp_path):
    harness = Harness(db, tmp_path)
    yield harness
    harness.cleanup()


def test_replacing_file_retains_old_snapshot_and_unmodified_revisions(knowledge):
    h = knowledge
    kb = h.create()
    first = h.upload(kb, "DREV", 500)
    unchanged = h.upload(kb, "UNCHANGED", 350)
    a = h.build(h.start(kb, changes=[
        {"action": "add", "document_id": "DREV", "upload_id": first},
        {"action": "add", "document_id": "UNCHANGED", "upload_id": unchanged},
    ]))
    original = {row["document_id"]: row for row in h.service.documents(kb, a)}
    original_chunks = h.db.all("SELECT id,text FROM chunks WHERE kb_version_id=%s ORDER BY id", (a,))
    replacement = h.upload(kb, "DREV", 600)
    pending = h.start(kb, a, [{"action": "replace", "document_id": "DREV", "upload_id": replacement}])
    assert h.db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (kb,))["current_version_id"] == a
    b = h.build(pending)
    current = {row["document_id"]: row for row in h.service.documents(kb, b)}
    assert original["DREV"]["document_version_id"] != current["DREV"]["document_version_id"]
    assert original["UNCHANGED"]["document_version_id"] == current["UNCHANGED"]["document_version_id"]
    assert h.db.all("SELECT id,text FROM chunks WHERE kb_version_id=%s ORDER BY id", (a,)) == original_chunks
    new_chunks = h.db.all("SELECT id,text FROM chunks WHERE kb_version_id=%s", (b,))
    assert not ({item["id"] for item in original_chunks} & {item["id"] for item in new_chunks})
    assert any("500元" in block["text"] for block in h.service.source(kb, a, original["DREV"]["document_version_id"])["blocks"])
    assert any("600元" in block["text"] for block in h.service.source(kb, b, current["DREV"]["document_version_id"])["blocks"])
    assert h.db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (kb,))["current_version_id"] == b


def test_competing_publications_cannot_overwrite_a_newer_manifest(knowledge):
    h = knowledge
    kb = h.create()
    a = h.initial(kb)
    one = h.start(kb, a, auto_publish=False)
    b = h.build(one)
    two = h.start(kb, a, auto_publish=False)
    c = h.build(two)
    h.service.publish(kb, b, a, h.key())
    with pytest.raises(AppError) as stale:
        h.service.publish(kb, c, a, h.key())
    assert stale.value.code == "VERSION_CONFLICT"
    with pytest.raises(AppError) as forged_base:
        h.service.publish(kb, c, b, h.key())
    assert forged_base.value.code == "VERSION_CONFLICT"
    assert h.db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (kb,))["current_version_id"] == b


def test_invalid_embedding_never_replaces_published_snapshot(knowledge):
    h = knowledge
    kb = h.create()
    a = h.initial(kb)
    replacement = h.upload(kb, amount=600)
    pending = h.start(kb, a, [{"action": "replace", "document_id": "DREV", "upload_id": replacement}])

    class BadShape(MockEmbedder):
        def encode(self, texts):
            return [[1.0] * 1023 for _ in texts]

    broken = KnowledgeService(h.db, h.settings, BadShape())
    with pytest.raises(AppError) as failure:
        broken.build(h.claim(pending))
    assert failure.value.code == "EMBEDDING_INVALID"
    with pytest.raises(AppError) as publish:
        h.service.publish(kb, pending["version_id"], a, h.key())
    assert publish.value.code == "VERSION_NOT_READY"
    assert h.db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (kb,))["current_version_id"] == a


def test_cross_knowledge_base_upload_and_sources_are_rejected(knowledge):
    h = knowledge
    kb, other = h.create(), h.create()
    upload = h.upload(kb)
    with pytest.raises(AppError) as mismatched_upload:
        h.start(other, changes=[{"action": "add", "document_id": "DREV", "upload_id": upload}])
    assert mismatched_upload.value.status == 404
    version = h.build(h.start(kb, changes=[{"action": "add", "document_id": "DREV", "upload_id": upload}]))
    revision = h.service.documents(kb, version)[0]["document_version_id"]
    for method in (
        lambda: h.service.documents(other, version),
        lambda: h.service.source(other, version, revision),
        lambda: Retriever(h.db, h.embedder, MockReranker()).search(other, version, ["住宿"]),
    ):
        with pytest.raises(AppError) as error:
            method()
        assert error.value.status == 404


def test_search_returns_only_candidates_not_read_evidence_and_handles_sql_text(knowledge):
    h = knowledge
    kb = h.create()
    version = h.initial(kb)
    retriever = Retriever(h.db, h.embedder, MockReranker())
    response = retriever.search(kb, version, ["上海住宿", "'); DROP TABLE chunks; --"])
    assert response["candidates"]
    assert not response["degraded"]
    for item in response["candidates"]:
        assert not {"text", "quote", "evidence_id", "source_ranges"} & item.keys()
    assert h.db.one("SELECT count(*) n FROM chunks WHERE kb_version_id=%s", (version,))["n"] > 0


def test_retrieval_degradation_is_visible_and_both_failed_routes_raise(knowledge):
    h = knowledge
    kb = h.create()
    version = h.initial(kb)

    class FailingEmbedding(MockEmbedder):
        def encode(self, texts):
            raise AppError("TEST_VECTOR_FAILURE", "injected", 503)

    class FailingReranker:
        def rank(self, query, texts):
            raise AppError("TEST_RERANK_FAILURE", "injected", 503)

    result = Retriever(h.db, FailingEmbedding(), FailingReranker()).search(kb, version, ["住宿"])
    assert result["candidates"]
    assert result["degraded"]
    assert {failure["stage"] for failure in result["failures"]} == {"vector", "rerank"}

    class FailedQueryDatabase:
        def one(self, *args):
            return h.db.one(*args)

        def all(self, *args):
            raise AppError("TEST_DATABASE_FAILURE", "injected", 503)

    with pytest.raises(AppError) as error:
        Retriever(FailedQueryDatabase(), h.embedder, MockReranker()).search(kb, version, ["住宿"])
    assert error.value.code == "RETRIEVAL_FAILED"


def test_fused_ranks_do_not_count_duplicate_chunk_twice_in_one_route():
    scores = reciprocal_rank_fusion([["a", "a", "b"], ["b", "a"]])
    assert scores["a"] == pytest.approx(scores["b"])


def test_finished_build_job_cannot_rewrite_ready_chunks(knowledge):
    h = knowledge
    kb = h.create()
    upload = h.upload(kb)
    request = h.start(kb, changes=[{"action": "add", "document_id": "DREV", "upload_id": upload}])
    job = h.claim(request)
    h.service.build(job)
    with pytest.raises(LeaseLost):
        h.service.build(job)


def test_retried_build_uses_original_total_deadline(knowledge):
    h = knowledge
    kb = h.create()
    published = h.initial(kb)
    request = h.start(kb, published)
    h.claim(request)
    h.settings.build_timeout_seconds = 1
    with h.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET payload=jsonb_set(payload,'{first_started_at}',to_jsonb(clock_timestamp()-interval '1 hour')) "
            "WHERE id=%s", (request["job_id"],)
        )
    job = h.db.one("SELECT * FROM jobs WHERE id=%s", (request["job_id"],))
    with pytest.raises(AppError) as timeout:
        h.service.build(job)
    assert timeout.value.code == "BUILD_TIMEOUT"
    assert h.db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (kb,))["current_version_id"] == published
    assert h.db.one("SELECT count(*) n FROM chunks WHERE kb_version_id=%s", (request["version_id"],))["n"] == 0
