"""Session -> input round -> audited model/tool calls, using isolated PG and stubs."""

import json

from fastapi.testclient import TestClient
import pytest

from xiaoying.agent.runtime import AgentRuntime
from xiaoying.api import create_app
from xiaoying.domain import AppError, LeaseLost
from xiaoying.providers.audit import public_model_payload
from xiaoying.providers.chat import DeepSeekChat

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_runtime import ScriptedChat, setup
from test_review_storage import db as database_fixture
from test_thinking_review import (
    PRIVATE, SequenceChat, assistant, completion, finish, setup_live, tool_call,
    sdk as sdk_fixture,
)

db = database_fixture
knowledge = knowledge_fixture
sdk = sdk_fixture
pytestmark = pytest.mark.integration


def decisions(suffix):
    return [
        assistant([tool_call("search_knowledge", {"queries": ["上海住宿"]}, f"search-{suffix}")]),
        assistant([tool_call("read_document", {"document_id": "DREV"}, f"read-{suffix}")]),
        finish(f"finish-{suffix}"),
    ]


def test_session_contains_two_rounds_with_separate_chains_and_shared_context(knowledge):
    h = knowledge
    runs, first, job, runtime = setup_live(h, SequenceChat(decisions("one")))
    runtime.run(job)
    cid = runs.get(first["run_id"])["conversation_id"]
    second = runs.create_run(cid, "那试用期员工呢？", h.key())
    AgentRuntime(h.db, h.settings, runs, runtime.retriever, chat=SequenceChat(decisions("two"))).run(h.claim(second))
    other_cid = runs.create_conversation(runs.get(first["run_id"])["kb_id"], h.key())["id"]
    other = runs.create_run(other_cid, "这是另一个会话", h.key())
    AgentRuntime(h.db, h.settings, runs, runtime.retriever, chat=SequenceChat(decisions("other"))).run(h.claim(other))
    before = h.db.one("SELECT count(*) n,sum(model_calls) calls FROM runs")
    with TestClient(create_app(settings=h.settings, db=h.db, embedder=h.embedder)) as client:
        response = client.get(f"/api/v1/conversations/{cid}/trace")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == cid
        assert [r["run"]["id"] for r in data["rounds"]] == [first["run_id"], second["run_id"]]
        assert other["run_id"] not in response.text
        assert PRIVATE not in response.text and "reasoning_content" not in response.text
        for round_ in data["rounds"]:
            record = round_["run"]
            assert record["conversation_id"] == cid
            assert record["model_calls"] == 4 and record["tool_calls"] == 2
            assert all(e["run_id"] == record["id"] for e in round_["events"])
            assert [e["payload"]["call_number"] for e in round_["events"] if e["type"] == "model.started"] == [1,2,3,4]
            assert [e["payload"]["tool"] for e in round_["events"] if e["type"] == "tool.started"] == ["search_knowledge", "read_document", "finish_answer"]
        detail = client.get(f"/api/v1/runs/{second['run_id']}/model-calls/2")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["phase"] == "decide" and payload["status"] == "completed"
        text = json.dumps(payload["request"], ensure_ascii=False)
        assert "上海住宿上限多少？" in text and "那试用期员工呢？" in text
        assert "search-one" in text, "A compatible session keeps earlier closed tool context"
        assert PRIVATE not in detail.text and "reasoning_content" not in detail.text
        first_detail = client.get(f"/api/v1/runs/{first['run_id']}/model-calls/2")
        assert "那试用期员工呢？" not in json.dumps(first_detail.json(), ensure_ascii=False)
    assert h.db.one("SELECT count(*) n,sum(model_calls) calls FROM runs") == before


def test_audit_uses_the_exact_sdk_request_builder_and_hides_private_fields(knowledge, sdk):
    sdk.responses.extend([completion(), completion(assistant([
        tool_call("finish_answer", {
            "task_status":"incomplete", "answer":"仅验证协议，不生成业务结论。", "citations":[],
            "information_gaps":["未读取业务原文"], "reason_code":"PROTOCOL_TEST",
        })
    ]))])
    chat = DeepSeekChat(sdk.settings)
    runs, request, job, runtime = setup_live(knowledge, chat)
    runtime.run(job)
    rid = request["run_id"]
    assert len(sdk.requests) == 2
    for number, sent in enumerate(sdk.requests, 1):
        record = runs.model_call(rid, number)
        assert record["request"] == public_model_payload(sent)
        assert record["status"] == "completed" and record["elapsed_ms"] >= 0
        serialized = json.dumps(record)
        assert PRIVATE not in serialized and sdk.settings.deepseek_api_key not in serialized
        assert "reasoning_content" not in serialized
    assert runs.model_call(rid, 2)["request"]["messages"][0]["role"] == "system"
    assert runs.model_call(rid, 2)["request"]["tools"]
    assert public_model_payload({"api_key":"secret", "headers":{"Authorization":"secret"}, "messages":[{"role":"assistant", "reasoning_content":"secret", "content":"visible"}]}) == {"messages":[{"role":"assistant", "content":"visible"}]}


def test_failed_model_call_has_request_error_and_session_trace(knowledge):
    class TimeoutChat(ScriptedChat):
        def decide(self, messages):
            raise AppError("TIMEOUT", "测试注入：模型请求超时", 504)
    runs, request, job, runtime = setup(knowledge, TimeoutChat())
    runtime.run(job)
    record = runs.model_call(request["run_id"], 2)
    assert record["status"] == "failed" and record["error"]["code"] == "TIMEOUT"
    assert "上海住宿上限多少？" in json.dumps(record["request"], ensure_ascii=False)
    cid = runs.get(request["run_id"])["conversation_id"]
    events = runs.conversation_trace(cid)["rounds"][0]["events"]
    assert any(event["type"] == "model.failed" for event in events)
    assert events[-1]["type"] == "run.finished"


def test_stale_worker_cannot_change_audited_model_call(knowledge):
    runs, request, job, _ = setup(knowledge)
    runs.start(job)
    runs.reserve_action(job, "model:fence", "model", {"messages": []})
    runs.model_started(job, "model:fence", "decide", "test", "test", {"messages": []})
    knowledge.claim(request)
    with pytest.raises(LeaseLost):
        runs.model_finished(job, "model:fence", response={"content":"stale"})
    assert runs.model_call(request["run_id"], 1)["status"] == "running"


def test_empty_session_and_legacy_call_are_reported_without_fabrication(knowledge):
    runs, first, job, runtime = setup(knowledge)
    runtime.run(job)
    kb = runs.get(first["run_id"])["kb_id"]
    empty = runs.create_conversation(kb, knowledge.key())["id"]
    with TestClient(create_app(settings=knowledge.settings, db=knowledge.db, embedder=knowledge.embedder)) as client:
        assert client.get(f"/api/v1/conversations/{empty}/trace").json()["rounds"] == []
        assert client.get(f"/api/v1/runs/{first['run_id']}/model-calls/999").status_code == 404
        assert client.get("/api/v1/conversations/not-a-uuid/trace").status_code == 422
