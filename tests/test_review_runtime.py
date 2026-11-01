"""Real LangGraph/PostgreSQL recovery tests with a declared scripted decision provider."""

import json
import time

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg.rows import dict_row

from xiaoying.agent.runtime import AgentRuntime
from xiaoying.domain import AppError, LeaseLost
from xiaoying.providers.embeddings import MockReranker
from xiaoying.rag.retrieval import Retriever
from xiaoying.services.runs import RunService
from xiaoying.storage.checkpoints import FencedPostgresSaver

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_storage import db as database_fixture

knowledge = knowledge_fixture
db = database_fixture
pytestmark = pytest.mark.integration


class ScriptedChat:
    """Checks runtime behavior only; these decisions are not an evaluation of DeepSeek."""

    def __init__(self):
        self.decisions = 0
        self.expansions = 0

    def expand(self, question, history):
        self.expansions += 1
        return {"resolved_question": question, "expanded_queries": [], "unresolved_slots": []}

    def call(self, name, args):
        self.decisions += 1
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"scripted-{self.decisions}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }]}

    @staticmethod
    def observations(messages):
        return [json.loads(message["content"]) for message in messages if message.get("role") == "tool"]

    def decide(self, messages):
        observations = self.observations(messages)
        if not observations:
            return self.call("search_knowledge", {"queries": ["上海住宿"], "reason": "scripted test search"})
        latest = observations[-1]
        if latest.get("error", {}).get("code") == "TOOL_CALL_LIMIT":
            return self.call("finish_answer", {
                "task_status": "incomplete", "answer": "检索与读取次数已用完，原文读取故障未恢复，不能确认制度内容。",
                "reason_code": "TOOL_CALL_LIMIT", "citations": [], "information_gaps": ["原文读取故障尚未恢复"],
            })
        if latest.get("candidates") or latest.get("error", {}).get("code") in {"TIMEOUT", "TEST_SEARCH_FAILURE"}:
            return self.call("read_document", {"document_id": "DREV", "reason": "scripted test read"})
        if latest.get("evidence"):
            evidence = next(item for item in latest["evidence"] if "500元" in item["quote"])
            return self.call("finish_answer", {
                "task_status": "answered", "answer": "上海住宿上限500元。", "reason_code": "SCRIPTED_TEST",
                "citations": [{"evidence_id": evidence["evidence_id"], "quote": evidence["quote"]}], "information_gaps": [],
            })
        return self.call("finish_answer", {
            "task_status": "incomplete", "answer": "脚本测试观察到未恢复错误。", "reason_code": "SCRIPTED_FAILURE",
            "citations": [], "information_gaps": ["尚未完成查证"],
        })


class ClarifyingChat(ScriptedChat):
    def decide(self, messages):
        observations = self.observations(messages)
        if observations and observations[-1].get("evidence"):
            return self.call("request_clarification", {
                "answer": "请补充身份信息。", "citations": [], "information_gaps": ["身份未知"],
                "questions": ["是否已经转正？"],
            })
        if observations and observations[-1].get("task_status") == "awaiting_user":
            return self.call("finish_answer", {
                "task_status": "insufficient", "answer": "补充后仍无对应制度。", "reason_code": "SCRIPTED_INSUFFICIENT",
                "citations": [], "information_gaps": ["缺少对应制度"],
            })
        return super().decide(messages)


def setup(harness, chat=None):
    h = harness
    kb = h.create()
    h.initial(kb)
    runs = RunService(h.db, h.settings, h.embedder)
    conversation = runs.create_conversation(kb, h.key())["id"]
    request = runs.create_run(conversation, "上海住宿上限多少？", h.key())
    job = h.claim(request)
    retriever = Retriever(h.db, h.embedder, MockReranker())
    runtime = AgentRuntime(h.db, h.settings, runs, retriever, chat=chat or ScriptedChat())
    return runs, request, job, runtime


def test_real_graph_executes_search_read_and_validated_answer(knowledge):
    runs, request, job, runtime = setup(knowledge)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered"
    assert record["tool_calls"] == 2
    events = knowledge.db.all("SELECT type,payload FROM trace_events WHERE run_id=%s ORDER BY seq", (record["id"],))
    tools = [event["payload"]["tool"] for event in events if event["type"] == "tool.started"]
    assert tools == ["search_knowledge", "read_document", "finish_answer"]
    assert sum(event["type"] == "run.finished" for event in events) == 1
    assert "500元" in record["result"]["citations"][0]["quote"]


def test_read_failure_recovery_changes_observation_and_preserves_source_gate(knowledge):
    runs, request, job, runtime = setup(knowledge)
    runtime.faults = {"read": 1}
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered"
    assert record["tool_calls"] == 3
    errors = knowledge.db.all("SELECT payload FROM trace_events WHERE run_id=%s AND type='tool.failed'", (record["id"],))
    assert len(errors) == 1 and errors[0]["payload"]["error"]["code"] == "TIMEOUT"
    assert record["result"]["citations"]


def test_persistent_read_failure_ends_at_budget_without_fabricated_evidence(knowledge):
    knowledge.settings.max_tool_calls = 4
    runs, request, job, runtime = setup(knowledge)
    runtime.faults = {"read": "always"}
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "incomplete"
    assert record["result"]["reason_code"] == "TOOL_CALL_LIMIT"
    assert record["tool_calls"] == 4
    assert record["result"]["citations"] == []
    assert record["clarification_count"] == 0


def test_graph_interrupt_resume_uses_same_thread_and_cumulative_budget(knowledge):
    h = knowledge
    chat = ClarifyingChat()
    runs, request, job, runtime = setup(h, chat)
    rid = request["run_id"]
    runtime.run(job)
    waiting = runs.get(rid)
    assert waiting["execution_status"] == "waiting"
    assert waiting["clarification_count"] == 1
    assert h.db.one("SELECT count(*) n FROM checkpoints WHERE thread_id=%s", (rid,))["n"] > 0
    response = runs.clarify(rid, {"action": "submit", "content": "我已转正"}, h.key())
    runtime.run(h.claim(response))
    finished = runs.get(rid)
    assert finished["task_status"] == "insufficient"
    assert finished["model_calls"] > waiting["model_calls"]
    assert finished["tool_calls"] == waiting["tool_calls"]
    assert finished["clarification_count"] == 1


class SimulatedCrash(Exception):
    pass


def test_checkpoint_interrupt_before_business_projection_recovers_once(knowledge, monkeypatch):
    h = knowledge
    chat = ClarifyingChat()
    runs, request, job, runtime = setup(h, chat)
    real_waiting = runs.waiting

    def crash(*args):
        raise SimulatedCrash("after checkpoint, before waiting projection")

    monkeypatch.setattr(runs, "waiting", crash)
    with pytest.raises(SimulatedCrash):
        runtime.run(job)
    before = runs.get(request["run_id"])
    calls_before = (chat.expansions, chat.decisions)
    monkeypatch.setattr(runs, "waiting", real_waiting)
    runtime.run(h.claim(request))
    after = runs.get(request["run_id"])
    assert after["execution_status"] == "waiting"
    assert after["clarification_count"] == 1
    assert after["model_calls"] == before["model_calls"]
    assert (chat.expansions, chat.decisions) == calls_before
    assert h.db.one("SELECT count(*) n FROM trace_events WHERE run_id=%s AND type='clarification.requested'", (request["run_id"],))["n"] == 1


def test_checkpoint_finished_before_final_projection_does_not_repeat_model_calls(knowledge, monkeypatch):
    h = knowledge
    chat = ScriptedChat()
    runs, request, job, runtime = setup(h, chat)
    real_finish = runs.finish

    def crash(*args):
        raise SimulatedCrash("after checkpoint, before result projection")

    monkeypatch.setattr(runs, "finish", crash)
    with pytest.raises(SimulatedCrash):
        runtime.run(job)
    before = runs.get(request["run_id"])
    calls_before = (chat.expansions, chat.decisions)
    monkeypatch.setattr(runs, "finish", real_finish)
    runtime.run(h.claim(request))
    after = runs.get(request["run_id"])
    assert after["task_status"] == "answered"
    assert after["model_calls"] == before["model_calls"]
    assert (chat.expansions, chat.decisions) == calls_before
    assert h.db.one("SELECT count(*) n FROM trace_events WHERE run_id=%s AND type='run.finished'", (request["run_id"],))["n"] == 1


def test_checkpoint_old_lease_cannot_commit_after_reclaim(knowledge):
    h = knowledge
    _, request, job, _ = setup(h)
    rid = request["run_id"]
    config = {"configurable": {"thread_id": rid, "checkpoint_ns": ""}}
    with psycopg.connect(h.settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        saver = FencedPostgresSaver(conn, job["id"], job["lease_token"])
        saved = saver.put(config, empty_checkpoint(), {"source": "input", "step": -1, "parents": {}}, {})
        assert saver.get_tuple(saved) is not None
        h.claim(request)
        with pytest.raises(LeaseLost):
            saver.put(config, empty_checkpoint(), {"source": "loop", "step": 0, "parents": {}}, {})
    assert h.db.one("SELECT count(*) n FROM checkpoints WHERE thread_id=%s", (rid,))["n"] == 1


def test_search_can_be_recovered_by_successful_direct_document_read(knowledge):
    runs, request, job, runtime = setup(knowledge)

    class FailedSearch:
        def search(self, *args, **kwargs):
            raise AppError("TEST_SEARCH_FAILURE", "injected search failure", 503, True)

    runtime.retriever = FailedSearch()
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered", "Successful source read can recover a failed search"


def test_search_respects_remaining_total_active_time(knowledge):
    knowledge.settings.active_timeout_seconds = 0.15
    runs, request, job, runtime = setup(knowledge)

    class SlowSearch:
        def search(self, *args, **kwargs):
            time.sleep(0.5)
            return {"candidates": [], "degraded": False}

    runtime.retriever = SlowSearch()
    started = time.monotonic()
    runtime.run(job)
    elapsed = time.monotonic() - started
    assert elapsed < 0.4, f"Search exceeded active budget: {elapsed:.3f}s"
    assert runs.get(request["run_id"])["task_status"] == "incomplete"


@pytest.mark.parametrize(
    "field,value",
    [("reranker_model", "review/different-reranker"), ("reranker_revision", "0" * 40)],
)
def test_changed_reranker_configuration_cannot_resume_saved_run(knowledge, field, value):
    chat = ScriptedChat()
    runs, request, job, runtime = setup(knowledge, chat)
    original = runs.get(request["run_id"])["config"][field]
    assert original != value
    setattr(knowledge.settings, field, value)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "incomplete"
    assert record["result"]["reason_code"] == "CONFIG_MISMATCH"
    assert record["model_calls"] == 0 and record["tool_calls"] == 0
    assert chat.expansions == 0 and chat.decisions == 0
    assert record["config"][field] == original
