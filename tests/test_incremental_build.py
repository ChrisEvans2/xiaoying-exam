"""Incremental indexing and durable build discovery against isolated PostgreSQL resources."""

from pathlib import Path
from uuid import uuid4

import pytest

from xiaoying.domain import AppError, BuildRequest
from xiaoying.providers.embeddings import MockEmbedder, MockReranker
from xiaoying.rag.retrieval import Retriever
from xiaoying.services import knowledge as knowledge_module

from test_review_api import client_for
from test_review_knowledge import knowledge as knowledge_fixture
from test_review_storage import db as database_fixture

db = database_fixture
knowledge = knowledge_fixture
pytestmark = pytest.mark.integration


def no_processing(monkeypatch, embedder):
    def forbidden(*args, **kwargs):
        pytest.fail("Unchanged documents must not be parsed, chunked or embedded again")
    monkeypatch.setattr(knowledge_module, "parse_document", forbidden)
    monkeypatch.setattr(knowledge_module, "chunk_document", forbidden)
    monkeypatch.setattr(embedder, "encode", forbidden)


@pytest.mark.parametrize("action,document", [("add", "NEW"), ("replace", "DREV")])
def test_only_added_or_replaced_document_is_processed(knowledge, monkeypatch, action, document):
    h = knowledge
    kb = h.create()
    base = h.initial(kb)
    upload = h.upload(kb, document, 600)
    parsed, chunked, encoded = [], [], []
    original_parse, original_chunk, original_encode = (
        knowledge_module.parse_document, knowledge_module.chunk_document, h.embedder.encode
    )

    def parse(path, metadata, settings, artifacts):
        assert metadata["document_id"] == document
        parsed.append(metadata["document_id"])
        return original_parse(path, metadata, settings, artifacts)

    def chunk(ir, embedder, config):
        chunked.append(ir.metadata["document_id"])
        return original_chunk(ir, embedder, config)

    def encode(texts):
        encoded.extend(texts)
        return original_encode(texts)

    monkeypatch.setattr(knowledge_module, "parse_document", parse)
    monkeypatch.setattr(knowledge_module, "chunk_document", chunk)
    monkeypatch.setattr(h.embedder, "encode", encode)
    request = h.start(kb, base, [{"action": action, "document_id": document, "upload_id": upload}])
    version = h.build(request)
    assert parsed == chunked == [document] and encoded
    progress = h.db.one("SELECT progress FROM jobs WHERE id=%s", (request["job_id"],))["progress"]
    assert progress["rebuilt"] == 1
    assert progress["reused"] == (1 if action == "add" else 0)
    assert {d["document_id"] for d in h.service.documents(kb, version)} == {"DREV", document}
    assert any("600元" in c["text"] for c in h.db.all("SELECT text FROM chunks WHERE kb_version_id=%s", (version,)))
    assert any("500元" in c["text"] for c in h.db.all("SELECT text FROM chunks WHERE kb_version_id=%s", (base,)))


def test_reuse_copies_complete_index_and_source_with_new_citation_ids(knowledge, monkeypatch):
    h = knowledge
    kb = h.create()
    base = h.initial(kb)
    old_chunks = h.db.all("SELECT * FROM chunks WHERE kb_version_id=%s ORDER BY ordinal", (base,))
    old_blocks = h.db.all("SELECT * FROM source_blocks WHERE kb_version_id=%s ORDER BY ordinal", (base,))
    no_processing(monkeypatch, h.embedder)
    request = h.start(kb, base)
    version = h.build(request)
    new_chunks = h.db.all("SELECT * FROM chunks WHERE kb_version_id=%s ORDER BY ordinal", (version,))
    new_blocks = h.db.all("SELECT * FROM source_blocks WHERE kb_version_id=%s ORDER BY ordinal", (version,))
    strip = lambda rows, keys: [{k: v for k, v in row.items() if k not in keys} for row in rows]
    assert strip(old_chunks, {"id", "kb_version_id"}) == strip(new_chunks, {"id", "kb_version_id"})
    assert strip(old_blocks, {"kb_version_id"}) == strip(new_blocks, {"kb_version_id"})
    assert {c["id"] for c in old_chunks}.isdisjoint(c["id"] for c in new_chunks)
    assert h.db.all("SELECT * FROM chunks WHERE kb_version_id=%s ORDER BY ordinal", (base,)) == old_chunks
    assert h.db.all("SELECT * FROM source_blocks WHERE kb_version_id=%s ORDER BY ordinal", (base,)) == old_blocks
    revision = h.service.documents(kb, version)[0]["document_version_id"]
    assert h.service.source(kb, version, revision)["blocks"]
    # Retrieval must work from the new version without relying on the old version's chunk IDs.
    retriever = Retriever(h.db, MockEmbedder(), MockReranker())
    candidates = retriever.search(kb, version, ["住宿"])["candidates"]
    assert candidates and {c["chunk_id"] for c in candidates} <= {c["id"] for c in new_chunks}
    progress = h.db.one("SELECT progress FROM jobs WHERE id=%s", (request["job_id"],))["progress"]
    assert progress["reused"] == 1 and progress["rebuilt"] == 0 and progress["published"]


def test_identical_reupload_reuses_revision_without_processing(knowledge, monkeypatch):
    h = knowledge
    kb = h.create()
    base = h.initial(kb)
    revision = h.service.documents(kb, base)[0]["document_version_id"]
    upload = h.upload(kb)
    no_processing(monkeypatch, h.embedder)
    version = h.build(h.start(kb, base, [{"action": "replace", "document_id": "DREV", "upload_id": upload}]))
    assert h.service.documents(kb, version)[0]["document_version_id"] == revision


@pytest.mark.parametrize("changed", ["chunking", "embedding", "metadata", "format"])
def test_changed_build_inputs_are_reprocessed(knowledge, monkeypatch, changed):
    h = knowledge
    kb = h.create()
    base = h.initial(kb)
    parsed = []
    original = knowledge_module.parse_document

    def parse(path, metadata, settings, artifacts):
        parsed.append(metadata["document_id"])
        return original(path, metadata, settings, artifacts)

    monkeypatch.setattr(knowledge_module, "parse_document", parse)
    body = BuildRequest(base_version_id=base)
    if changed == "chunking":
        body.config.target_tokens = 300
    elif changed == "embedding":
        h.embedder.identity = "test-next-embedding-model"
    else:
        source = h.db.one("SELECT * FROM uploads WHERE kb_id=%s", (kb,))
        upload = h.service.upload(
            kb, "DREV.md" if changed == "format" else source["name"],
            Path(source["storage_path"]).read_bytes(),
            {"company": "新增的公司元数据"} if changed == "metadata" else {}, h.key(),
        )["upload_id"]
        body = BuildRequest(base_version_id=base, changes=[
            {"action": "replace", "document_id": "DREV", "upload_id": upload}
        ])
    request = h.service.create_build(kb, body, h.key())
    h.build(request)
    assert parsed == ["DREV"]
    progress = h.db.one("SELECT progress FROM jobs WHERE id=%s", (request["job_id"],))["progress"]
    assert progress["reused"] == 0 and progress["rebuilt"] == 1


def test_removal_only_reuses_remaining_documents(knowledge, monkeypatch):
    h = knowledge
    kb = h.create()
    uploads = [(doc, h.upload(kb, doc)) for doc in ("DREV", "REMOVE")]
    base = h.build(h.start(kb, changes=[
        {"action": "add", "document_id": doc, "upload_id": upload} for doc, upload in uploads
    ]))
    no_processing(monkeypatch, h.embedder)
    version = h.build(h.start(kb, base, [{"action": "remove", "document_id": "REMOVE"}]))
    assert [d["document_id"] for d in h.service.documents(kb, version)] == ["DREV"]


def test_build_discovery_survives_new_http_client_and_is_scoped(knowledge):
    h = knowledge
    kb, other = h.create(), h.create()
    base = h.initial(kb)
    body, key = BuildRequest(base_version_id=base), h.key()
    request = h.service.create_build(kb, body, key)
    assert h.service.create_build(kb, body, key) == request
    other_request = h.start(other, changes=[
        {"action": "add", "document_id": "DREV", "upload_id": h.upload(other)}
    ])
    with client_for(h) as client:
        jobs = client.get(f"/api/v1/knowledge-bases/{kb}/builds").json()
        assert jobs[0]["id"] == request["job_id"] and jobs[0]["status"] == "pending"
        assert all("lease_token" not in job and "worker_id" not in job for job in jobs)
        assert other_request["job_id"] not in str(jobs)
        assert client.get(f"/api/v1/knowledge-bases/{other}/builds").json()[0]["id"] == other_request["job_id"]
        assert client.get(f"/api/v1/knowledge-bases/{uuid4()}/builds").status_code == 404
    h.build(request)
    with client_for(h) as reopened:
        jobs = reopened.get(f"/api/v1/knowledge-bases/{kb}/builds").json()
        assert jobs[0]["id"] == request["job_id"] and jobs[0]["status"] == "done"
        assert jobs[0]["progress"]["published"]


def test_duplicate_build_does_not_create_version_or_job(knowledge):
    h = knowledge
    kb = h.create()
    base = h.initial(kb)
    request = h.start(kb, base)
    count = h.db.one("SELECT count(*) n FROM kb_versions WHERE kb_id=%s", (kb,))["n"]
    with pytest.raises(AppError) as failure:
        h.start(kb, base)
    assert failure.value.code == "BUILD_IN_PROGRESS"
    assert h.db.one("SELECT count(*) n FROM kb_versions WHERE kb_id=%s", (kb,))["n"] == count
    assert h.service.builds(kb)[0]["id"] == request["job_id"]
