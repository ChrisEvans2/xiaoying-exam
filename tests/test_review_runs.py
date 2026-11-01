"""Persistent run boundaries without claiming a scripted provider is model evaluation."""

from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

import pytest

from xiaoying.domain import AppError
from xiaoying.providers.bounded import bounded
from xiaoying.providers.embeddings import BGEEmbedder
from xiaoying.services.runs import RunService

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_storage import db as database_fixture

knowledge = knowledge_fixture
db = database_fixture

pytestmark = pytest.mark.integration


def start(harness):
    kb = harness.create()
    version = harness.initial(kb)
    runs = RunService(harness.db, harness.settings, harness.embedder)
    conversation = runs.create_conversation(kb, harness.key())["id"]
    request = runs.create_run(conversation, "上海住宿上限多少？", harness.key())
    job = harness.claim(request)
    runs.start(job)
    return kb, version, runs, request, job


def wait_for_user(runs, job, action_id="clarify:1"):
    payload = runs.prepare_clarification(job, action_id, {
        "answer": "尚缺少适用身份，请补充。",
        "information_gaps": ["员工身份尚未明确"],
        "questions": ["是否已经转正？"],
        "citations": [],
    })
    runs.waiting(job, payload)
    return payload


def read_action(runs, job, action_id, args):
    cached, output = runs.reserve_action(job, action_id, "tool", {"name": "read_document", "args": args})
    if not cached:
        output = runs.read(job, action_id, args)
        runs.complete_action(job, action_id, output)
    return output


def test_worker_rejects_queued_index_after_model_configuration_changes(knowledge):
    h = knowledge
    kb = h.create()
    h.initial(kb)
    runs = RunService(h.db, h.settings, h.embedder)
    cid = runs.create_conversation(kb, h.key())["id"]
    request = runs.create_run(cid, "上海住宿上限？", h.key())
    embedder = BGEEmbedder("BAAI/bge-m3", h.settings.embedding_revision)
    changed = RunService(h.db, h.settings, embedder)
    with pytest.raises(AppError) as error:
        changed.start(h.claim(request))
    assert error.value.code == "INDEX_MODEL_MISMATCH"
    assert runs.get(request["run_id"])["execution_status"] == "queued"
    assert runs.get(request["run_id"])["model_calls"] == 0
    assert embedder._model is None


@pytest.mark.parametrize("explicit_version", [False, True])
def test_incompatible_resume_keeps_clarification_pending(knowledge, explicit_version):
    h = knowledge
    _, version, runs, request, job = start(h)
    wait_for_user(runs, job)
    changed = RunService(h.db, h.settings, BGEEmbedder("BAAI/bge-m3", h.settings.embedding_revision))
    body = {"action": "submit", "content": "我已转正"}
    if explicit_version:
        body["version_id"] = version
    with pytest.raises(AppError) as error:
        changed.clarify(request["run_id"], body, h.key())
    assert error.value.code == "INDEX_MODEL_MISMATCH"
    assert runs.get(request["run_id"])["execution_status"] == "waiting"
    assert h.db.one("SELECT status FROM clarifications WHERE run_id=%s", (request["run_id"],))["status"] == "pending"


def test_unread_modified_and_user_supplied_quotes_are_rejected(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    base = {"task_status": "answered", "answer": "上海上限500元。", "information_gaps": []}
    with pytest.raises(AppError) as unread:
        runs.validate_result(rid, {**base, "citations": [{"evidence_id": str(uuid4()), "quote": "用户说可以报销999元"}]})
    assert unread.value.code == "UNREAD_CITATION"
    evidence = read_action(runs, job, "read:1", {"document_id": "DREV"})["evidence"]
    relevant = next(item for item in evidence if "500元" in item["quote"])
    with pytest.raises(AppError) as modified:
        runs.validate_result(rid, {**base, "citations": [{"evidence_id": relevant["evidence_id"], "quote": relevant["quote"].replace("500", "999")}]})
    assert modified.value.code == "UNREAD_CITATION"
    checked = runs.validate_result(rid, {**base, "citations": [{"evidence_id": relevant["evidence_id"], "quote": relevant["quote"]}]})
    assert checked["citations"][0]["document_id"] == "DREV"
    assert checked["citations"][0]["kb_version_id"] == relevant["kb_version_id"]


def test_first_insufficient_cannot_bypass_single_clarification_window(knowledge):
    _, _, runs, request, _ = start(knowledge)
    with pytest.raises(AppError) as error:
        runs.validate_result(request["run_id"], {
            "task_status": "insufficient", "answer": "缺少资料。", "citations": [], "information_gaps": ["制度不足"]
        })
    assert error.value.code == "CLARIFICATION_REQUIRED"


def test_clarification_resume_is_idempotent_and_keeps_used_budgets(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    runs.reserve_action(job, "model:1", "model", {"question": "identity"})
    runs.complete_action(job, "model:1", {"ok": True})
    runs.reserve_action(job, "read:1", "tool", {"document_id": "DREV"})
    runs.complete_action(job, "read:1", {"ok": True})
    payload = wait_for_user(runs, job)
    assert payload["deadline_at"]
    assert runs.get(rid)["clarification_count"] == 1
    key, body = h.key(), {"action": "submit", "content": "我已转正"}
    first = runs.clarify(rid, body, key)
    assert runs.clarify(rid, body, key) == first
    resumed_job = h.claim(first)
    runs.start(resumed_job)
    saved = runs.get(rid)
    assert saved["model_calls"] == 1 and saved["tool_calls"] == 1
    with pytest.raises(AppError) as second_window:
        wait_for_user(runs, resumed_job, "clarify:2")
    assert second_window.value.code == "CLARIFICATION_EXHAUSTED"
    result = runs.validate_result(rid, {
        "task_status": "insufficient", "answer": "补充后仍缺少制度依据。", "citations": [], "information_gaps": ["缺少制度"]
    })
    runs.finish(resumed_job, result)
    assert runs.get(rid)["task_status"] == "insufficient"
    assert h.db.one("SELECT count(*) n FROM clarifications WHERE run_id=%s", (rid,))["n"] == 1


def test_abandon_has_terminal_state_and_cannot_be_resumed(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    wait_for_user(runs, job)
    response = runs.clarify(rid, {"action": "abandon"}, h.key())
    assert response["result"]["reason_code"] == "USER_ABANDONED"
    assert runs.get(rid)["execution_status"] == "finished"
    with pytest.raises(AppError) as error:
        runs.clarify(rid, {"action": "submit", "content": "又补充了"}, h.key())
    assert error.value.code == "NOT_WAITING"


def test_expired_submission_commits_timeout_and_never_calls_it_abandonment(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    wait_for_user(runs, job)
    with h.db.tx() as conn:
        conn.execute("UPDATE clarifications SET deadline_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (rid,))
        conn.execute("UPDATE runs SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=%s", (rid,))
    with pytest.raises(AppError) as error:
        runs.clarify(rid, {"action": "submit", "content": "我已转正"}, h.key())
    assert error.value.code == "CLARIFICATION_EXPIRED"
    result = runs.get(rid)["result"]
    assert result["task_status"] == "insufficient"
    assert result["reason_code"] == "CLARIFICATION_TIMEOUT"
    assert "放弃" not in result["answer"]
    assert h.db.one("SELECT count(*) n FROM jobs WHERE resource_id=%s AND status IN ('pending','running')", (rid,))["n"] == 0


def test_timeout_scanner_and_late_submission_have_one_terminal_event(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    wait_for_user(runs, job)
    with h.db.tx() as conn:
        conn.execute("UPDATE clarifications SET deadline_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (rid,))
        conn.execute("UPDATE runs SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=%s", (rid,))

    def submit():
        try:
            runs.clarify(rid, {"action": "submit", "content": "迟到补充"}, h.key())
        except AppError as error:
            assert error.code in {"NOT_WAITING", "CLARIFICATION_EXPIRED"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(runs.sweep), executor.submit(submit)]
        for future in futures:
            future.result()
    assert runs.get(rid)["result"]["reason_code"] == "CLARIFICATION_TIMEOUT"
    assert h.db.one("SELECT count(*) n FROM trace_events WHERE run_id=%s AND type='run.finished'", (rid,))["n"] == 1


def test_explicit_version_change_invalidates_old_evidence_but_preserves_audit(knowledge):
    h = knowledge
    kb, a, runs, request, job = start(h)
    rid = request["run_id"]
    old = next(item for item in read_action(runs, job, "read:1", {"document_id": "DREV"})["evidence"] if "500元" in item["quote"])
    wait_for_user(runs, job)
    upload = h.upload(kb, amount=600)
    b = h.build(h.start(kb, a, [{"action": "replace", "document_id": "DREV", "upload_id": upload}]))
    assert runs.get(rid)["kb_version_id"] == a
    response = runs.clarify(rid, {"action": "submit", "version_id": b, "content": "已更新制度"}, h.key())
    new_job = h.claim(response)
    runs.start(new_job)
    with pytest.raises(AppError) as stale:
        runs.validate_result(rid, {"task_status": "answered", "answer": "上限500元", "citations": [old]})
    assert stale.value.code == "UNREAD_CITATION"
    assert h.db.one("SELECT count(*) n FROM evidence_reads WHERE id=%s", (old["evidence_id"],))["n"] == 1
    new = next(item for item in read_action(runs, new_job, "read:2", {"document_id": "DREV"})["evidence"] if "600元" in item["quote"])
    result = runs.validate_result(rid, {"task_status": "answered", "answer": "上限600元", "citations": [new]})
    assert result["citations"][0]["kb_version_id"] == b
    assert runs.get(rid)["evidence_epoch"] == 1


def test_unknown_call_replay_consumes_budget_and_completed_replay_does_not(knowledge):
    h = knowledge
    h.settings.max_model_calls = 2
    _, _, runs, request, job = start(h)
    args = {"query": "上海"}
    assert runs.reserve_action(job, "model:1", "model", args) == (False, None)
    assert runs.reserve_action(job, "model:1", "model", args) == (False, None)
    with pytest.raises(AppError) as error:
        runs.reserve_action(job, "model:1", "model", args)
    assert error.value.code == "EXECUTION_LIMIT"
    runs.complete_action(job, "model:1", {"answer": "observed"})
    assert runs.reserve_action(job, "model:1", "model", args) == (True, {"answer": "observed"})
    assert runs.get(request["run_id"])["model_calls"] == 2


def test_ordinary_read_cannot_fetch_chunk_from_new_snapshot(knowledge):
    h = knowledge
    kb, a, runs, _, job = start(h)
    upload = h.upload(kb, amount=600)
    b = h.build(h.start(kb, a, [{"action": "replace", "document_id": "DREV", "upload_id": upload}]))
    foreign = h.db.one("SELECT id FROM chunks WHERE kb_version_id=%s LIMIT 1", (b,))["id"]
    with pytest.raises(AppError) as error:
        runs.read(job, "read:foreign", {"chunk_ids": [foreign]})
    assert error.value.code == "SOURCE_NOT_FOUND"


def test_timed_out_read_arriving_late_does_not_become_citable(knowledge):
    h = knowledge
    _, _, runs, request, job = start(h)
    rid = request["run_id"]
    action_id = "read:late"
    args = {"document_id": "DREV"}
    runs.reserve_action(job, action_id, "tool", {"name": "read_document", "args": args})
    release, returned = threading.Event(), threading.Event()
    late_result = {}

    def delayed_read():
        release.wait(2)
        try:
            late_result.update(runs.read(job, action_id, args))
        finally:
            returned.set()

    try:
        with pytest.raises(AppError) as timeout:
            bounded(delayed_read, 0.02)
        assert timeout.value.code == "TIMEOUT"
        runs.complete_action(job, action_id, {"error": {"code": "TIMEOUT", "message": "injected deadline", "retryable": True}})
    finally:
        release.set()
        assert returned.wait(2), "Delayed read must exit before the fixture is removed"
    assert runs.get(rid)["execution_status"] == "running"
    assert late_result["evidence"], "The thread did fetch original source after its caller timed out"
    assert h.db.one("SELECT count(*) n FROM evidence_reads WHERE run_id=%s", (rid,))["n"] == 0
    evidence = late_result["evidence"][0]
    with pytest.raises(AppError) as invalid:
        runs.validate_result(rid, {"task_status": "answered", "answer": "Cannot use unobserved evidence", "citations": [evidence]})
    assert invalid.value.code == "UNREAD_CITATION"
    runs.complete_action(job, action_id, late_result)
    assert h.db.one("SELECT output FROM run_actions WHERE run_id=%s AND action_id=%s", (rid, action_id))["output"]["error"]["code"] == "TIMEOUT"
    assert h.db.one("SELECT count(*) n FROM evidence_reads WHERE run_id=%s", (rid,))["n"] == 0
