"""HTTP/SSE boundary tests against the real database and persisted event log."""

import json
import io
import zipfile
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from xiaoying.api import create_app
from xiaoying.providers.embeddings import BGEEmbedder
from xiaoying.services.runs import final_result

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_runtime import setup
from test_review_storage import db as database_fixture

knowledge = knowledge_fixture
db = database_fixture
pytestmark = pytest.mark.integration


def client_for(harness):
    return TestClient(create_app(settings=harness.settings, db=harness.db, embedder=harness.embedder))


def parse_sse(text):
    records = []
    for block in text.strip().split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        records.append((int(fields["id"]), json.loads(fields["data"])))
    return records


def test_health_reports_mode_without_exposing_api_secret(knowledge):
    knowledge.settings.deepseek_api_key = "review-secret-never-expose"
    with client_for(knowledge) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["provider_mode"] == "mock"
    assert response.json()["model_configured"] is True
    assert knowledge.settings.deepseek_api_key not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_http_write_idempotency_and_uniform_request_errors(knowledge):
    h = knowledge
    with client_for(h) as client:
        missing = client.post("/api/v1/knowledge-bases", json={"name": "review"})
        assert missing.status_code == 422
        error = missing.json()["error"]
        assert {"code", "message", "retryable", "request_id"} <= error.keys()
        assert error["request_id"] == missing.headers["x-request-id"]
        headers = {"Idempotency-Key": h.key()}
        first = client.post("/api/v1/knowledge-bases", json={"name": "review"}, headers=headers)
        assert first.status_code == 201
        h.kbs.append(first.json()["id"])
        second = client.post("/api/v1/knowledge-bases", json={"name": "review"}, headers=headers)
        assert first.json() == second.json()
        conflict = client.post("/api/v1/knowledge-bases", json={"name": "different"}, headers=headers)
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        malformed = client.get("/api/v1/runs/not-a-uuid")
        assert malformed.status_code == 422


def test_incompatible_index_is_rejected_before_run_job_or_user_message_is_created(knowledge):
    h = knowledge
    kb = h.create()
    h.initial(kb)
    # The live adapter stays lazy: rejection must never load/call BGE or DeepSeek.
    embedder = BGEEmbedder("BAAI/bge-m3", h.settings.embedding_revision)
    with TestClient(create_app(settings=h.settings, db=h.db, embedder=embedder)) as client:
        conversation = client.post(
            "/api/v1/conversations", json={"kb_id": kb}, headers={"Idempotency-Key": h.key()}
        ).json()["id"]
        jobs_before = h.db.one("SELECT count(*) n FROM jobs")["n"]
        response = client.post(
            f"/api/v1/conversations/{conversation}/runs",
            json={"question": "上海住宿上限多少？"}, headers={"Idempotency-Key": h.key()},
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INDEX_MODEL_MISMATCH"
    assert "构建并发布新版本" in response.json()["error"]["message"]
    assert response.json()["error"]["retryable"] is False
    assert h.db.one("SELECT count(*) n FROM runs WHERE kb_id=%s", (kb,))["n"] == 0
    assert h.db.one("SELECT count(*) n FROM messages WHERE conversation_id=%s", (conversation,))["n"] == 0
    assert h.db.one("SELECT count(*) n FROM jobs")["n"] == jobs_before
    assert embedder._model is None


def test_upload_rejects_unsupported_types_bad_metadata_and_oversize(knowledge):
    h = knowledge
    kb = h.create()
    with client_for(h) as client:
        unsupported = client.post(f"/api/v1/knowledge-bases/{kb}/uploads", headers={"Idempotency-Key": h.key()}, files={"file": ("old.doc", b"old word")})
        assert unsupported.status_code == 415
        malformed = client.post(f"/api/v1/knowledge-bases/{kb}/uploads", headers={"Idempotency-Key": h.key()}, files={"file": ("policy.txt", b"text")}, data={"metadata": "[]"})
        assert malformed.status_code == 422
        h.settings.max_upload_bytes = 32
        large = client.post(f"/api/v1/knowledge-bases/{kb}/uploads", headers={"Idempotency-Key": h.key()}, files={"file": ("policy.txt", b"x" * 33)})
        assert large.status_code == 413


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_corrupt_supported_file_returns_structured_client_error(knowledge, kind):
    h = knowledge
    kb = h.create()
    if kind == "pdf":
        payload = b"%PDF-1.7\nnot a valid PDF body\n%%EOF"
    else:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", "<invalid-document/>")
        payload = stream.getvalue()
    with TestClient(create_app(settings=h.settings, db=h.db, embedder=h.embedder), raise_server_exceptions=False) as client:
        response = client.post(f"/api/v1/knowledge-bases/{kb}/uploads", headers={"Idempotency-Key": h.key()}, files={"file": (f"broken.{kind}", payload)})
    assert response.status_code in {415, 422}, response.text
    assert {"code", "message", "retryable", "request_id"} <= response.json()["error"].keys()


def test_sse_replay_is_contiguous_and_last_event_id_has_priority(knowledge):
    h = knowledge
    runs, request, job, runtime = setup(h)
    runtime.run(job)
    rid = request["run_id"]
    with client_for(h) as client:
        complete = client.get(f"/api/v1/runs/{rid}/events")
        assert complete.status_code == 200
        assert complete.headers["content-type"].startswith("text/event-stream")
        records = parse_sse(complete.text)
        assert [seq for seq, _ in records] == list(range(1, len(records) + 1))
        assert records[-1][1]["type"] == "run.finished"
        cut = records[len(records) // 2][0]
        replay = client.get(f"/api/v1/runs/{rid}/events?after_seq=0", headers={"Last-Event-ID": str(cut)})
        assert parse_sse(replay.text) == [(seq, value) for seq, value in records if seq > cut]
        empty = client.get(f"/api/v1/runs/{rid}/events", headers={"Last-Event-ID": str(records[-1][0])})
        assert empty.status_code == 200 and empty.text == ""
        invalid = client.get(f"/api/v1/runs/{rid}/events", headers={"Last-Event-ID": str(records[-1][0] + 1)})
        assert invalid.status_code == 422
        assert runs.get(rid)["task_status"] == "answered"


def test_source_and_download_keep_revision_boundary_and_hide_local_path(knowledge):
    h = knowledge
    kb, other = h.create(), h.create()
    version = h.initial(kb)
    revision = h.service.documents(kb, version)[0]["document_version_id"]
    path = f"/api/v1/knowledge-bases/{kb}/versions/{version}/documents/{revision}"
    with client_for(h) as client:
        source = client.get(path + "/source")
        assert source.status_code == 200
        assert "storage_path" not in source.json()
        assert source.json()["blocks"]
        download = client.get(source.json()["file_url"])
        assert download.status_code == 200 and "500元" in download.content.decode()
        assert download.headers["content-disposition"].startswith("attachment")
        wrong_kb = client.get(path.replace(kb, other) + "/file")
        assert wrong_kb.status_code == 404
        wrong_revision = client.get(path.replace(revision, str(uuid4())) + "/file")
        assert wrong_revision.status_code == 404


def test_job_status_does_not_publish_fencing_token(knowledge):
    h = knowledge
    _, request, _, _ = setup(h)
    with client_for(h) as client:
        response = client.get(f"/api/v1/jobs/{request['job_id']}")
    assert response.status_code == 200
    assert "lease_token" not in response.json()
    assert "worker_id" not in response.json()


def test_history_lists_saved_conversations_by_kb_and_can_reopen_old_turns(knowledge):
    h = knowledge
    runs, first, job, runtime = setup(h)
    runtime.run(job)
    original = runs.get(first["run_id"])
    kb, first_cid = original["kb_id"], original["conversation_id"]
    second_cid = runs.create_conversation(kb, h.key())["id"]
    second = runs.create_run(second_cid, "第二次独立咨询", h.key())
    second_job = h.claim(second)
    runs.start(second_job)
    runs.finish(second_job, final_result("incomplete", "TEST_FAILURE", "脚本故障记录，保留供历史查看。"))
    followup = runs.create_run(first_cid, "第一段对话的追问", h.key())
    runtime.run(h.claim(followup))
    empty_cid = runs.create_conversation(kb, h.key())["id"]
    other_kb = h.create()
    h.initial(other_kb)
    other_cid = runs.create_conversation(other_kb, h.key())["id"]
    runs.create_run(other_cid, "其他知识库的对话", h.key())
    before = h.db.one("SELECT count(*) n,sum(model_calls) model_calls FROM runs")
    with client_for(h) as client:
        response = client.get(f"/api/v1/knowledge-bases/{kb}/conversations")
        assert response.status_code == 200
        history = response.json()
        assert [item["id"] for item in history] == [first_cid, second_cid]
        assert history[0]["title"] == "上海住宿上限多少？", "A follow-up must not replace the conversation title"
        assert set(history[0]) == {"id", "kb_id", "title", "created_at", "last_activity_at", "input_count"}
        assert history[0]["input_count"] == 2 and history[1]["input_count"] == 1
        assert other_cid not in response.text and empty_cid not in response.text
        failed = client.get(f"/api/v1/conversations/{second_cid}").json()["runs"][0]
        assert failed["task_status"] == "incomplete" and "脚本故障" in failed["result"]["answer"]
        reopened = client.get(f"/api/v1/conversations/{first_cid}")
        assert reopened.status_code == 200
        assert [run["id"] for run in reopened.json()["runs"]] == [first["run_id"], followup["run_id"]]
        assert all(run["result"]["citations"] for run in reopened.json()["runs"])
        traces = client.get(f"/api/v1/runs/{first['run_id']}/events")
        assert parse_sse(traces.text)[-1][1]["type"] == "run.finished"
    assert h.db.one("SELECT count(*) n,sum(model_calls) model_calls FROM runs") == before


def test_history_handles_empty_missing_and_malformed_knowledge_base(knowledge):
    kb = knowledge.create()
    with client_for(knowledge) as client:
        empty = client.get(f"/api/v1/knowledge-bases/{kb}/conversations")
        assert empty.status_code == 200 and empty.json() == []
        missing = client.get(f"/api/v1/knowledge-bases/{uuid4()}/conversations")
        assert missing.status_code == 404
        malformed = client.get("/api/v1/knowledge-bases/not-a-uuid/conversations")
        assert malformed.status_code == 422
