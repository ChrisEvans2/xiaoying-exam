"""Budget closing uses the real graph and isolated PostgreSQL; no live model calls."""

from copy import deepcopy
import json

import pytest

from xiaoying.providers.chat import MockChat, validate_tool_history

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_runtime import ScriptedChat, setup
from test_review_storage import db as database_fixture

db = database_fixture
knowledge = knowledge_fixture
pytestmark = pytest.mark.integration


class RecordingChat(ScriptedChat):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def decide(self, messages):
        validate_tool_history(messages, False)
        self.inputs.append(deepcopy(messages))
        return self.choose(messages)

    def choose(self, messages):
        return super().decide(messages)

    @staticmethod
    def budget(messages):
        content = messages[-1]["content"]
        return json.loads(content[content.index("{"):content.index("\n")])

    def close(self, messages):
        evidence = [
            item for observation in self.observations(messages)
            for item in observation.get("evidence", []) if "500元" in item["quote"]
        ]
        return self.call("finish_answer", {
            "task_status": "answered" if evidence else "incomplete",
            "answer": "上海住宿上限500元。" if evidence else "已检索住宿制度，但未完成原文读取，暂不能确认住宿金额。",
            "reason_code": "BUDGET_CLOSING",
            "citations": [{"evidence_id": evidence[0]["evidence_id"], "quote": evidence[0]["quote"]}] if evidence else [],
            "information_gaps": [] if evidence else ["住宿金额尚未取得可引用原文"],
        })


def test_last_tool_read_can_still_produce_validated_answer(knowledge):
    knowledge.settings.max_tool_calls = 2
    chat = RecordingChat()
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered"
    assert record["tool_calls"] == 2 and record["model_calls"] == 4
    assert record["result"]["citations"]
    assert [chat.budget(m)["tool_calls_remaining"] for m in chat.inputs] == [2, 1, 0]
    assert "必须调用 finish_answer" in chat.inputs[-1][-1]["content"]


@pytest.mark.parametrize("plan", [
    ["search", "read"],
    ["search", "read", "read", "search", "read", "read", "search", "read"],
])
def test_over_limit_tool_is_returned_as_observation_and_answer_can_recover(knowledge, plan):
    knowledge.settings.max_tool_calls = len(plan)

    class ExtraSearchChat(RecordingChat):
        def choose(self, messages):
            observations = self.observations(messages)
            latest = observations[-1] if observations else {}
            if latest.get("error", {}).get("code") == "TOOL_CALL_LIMIT":
                return self.close(messages)
            action = plan[len(observations)] if len(observations) < len(plan) else "search"
            if action == "read":
                return self.call("read_document", {"document_id": "DREV", "reason": "test source read"})
            return self.call("search_knowledge", {"queries": ["上海住宿"], "reason": "test closing recovery"})

    chat = ExtraSearchChat()
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered"
    assert record["tool_calls"] == len(plan) and record["model_calls"] == len(plan) + 3
    blocked = knowledge.db.one(
        "SELECT payload FROM trace_events WHERE run_id=%s AND type='tool.failed'", (record["id"],)
    )["payload"]
    assert blocked["error"]["code"] == "TOOL_CALL_LIMIT"
    assert blocked["tool_call_id"]
    assert len([o for o in chat.observations(chat.inputs[-1]) if o.get("candidates")]) == plan.count("search")
    assert len([o for o in chat.observations(chat.inputs[-1]) if o.get("evidence")]) == plan.count("read")
    assert record["result"]["citations"], "Budget closing must still validate real read evidence"


def test_last_model_decision_is_prompted_to_report_specific_unverified_gap(knowledge):
    knowledge.settings.max_model_calls = 3

    class ModelBudgetChat(RecordingChat):
        def choose(self, messages):
            if self.budget(messages)["model_calls_remaining"] == 1:
                return self.close(messages)
            return super().choose(messages)

    chat = ModelBudgetChat()
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["model_calls"] == 3 and record["tool_calls"] == 1
    assert record["task_status"] == "incomplete"
    assert "原文读取" in record["result"]["answer"]
    assert record["result"]["information_gaps"] and not record["result"]["citations"]


def test_model_ignoring_closing_cannot_exceed_caps_or_loop_forever(knowledge):
    knowledge.settings.max_tool_calls = 2
    knowledge.settings.max_model_calls = 5

    class StubbornChat(RecordingChat):
        def choose(self, messages):
            return self.call("search_knowledge", {"queries": ["住宿"], "reason": "test repeated search"})

    runs, request, job, runtime = setup(knowledge, StubbornChat())
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["model_calls"] == 5 and record["tool_calls"] == 2
    assert record["task_status"] == "incomplete"
    assert record["result"]["reason_code"] == "EXECUTION_LIMIT"
    assert "模型调用已达到 5 次上限" in record["result"]["answer"]
    assert "2 次检索、0 次原文读取" in record["result"]["answer"]
    assert record["result"]["information_gaps"] and not record["result"]["citations"]
    assert knowledge.db.one(
        "SELECT count(*) n FROM run_actions WHERE run_id=%s AND kind='tool'", (record["id"],)
    )["n"] == 2
    assert knowledge.db.one(
        "SELECT count(*) n FROM trace_events WHERE run_id=%s AND type='run.finished'", (record["id"],)
    )["n"] == 1


def test_closing_reserves_execute_step_before_graph_budget_runs_out(knowledge):
    knowledge.settings.max_graph_steps = 6

    class StepBudgetChat(RecordingChat):
        def choose(self, messages):
            if "必须调用 finish_answer" in messages[-1]["content"]:
                return self.close(messages)
            return super().choose(messages)

    runs, request, job, runtime = setup(knowledge, StepBudgetChat())
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["result"]["reason_code"] == "BUDGET_CLOSING"
    assert record["model_calls"] == 3 and record["tool_calls"] == 1
    assert record["graph_steps"] == 5


def test_no_clarification_wait_when_budget_cannot_process_supplement(knowledge):
    knowledge.settings.max_tool_calls = 2

    class LateClarificationChat(RecordingChat):
        def choose(self, messages):
            observations = self.observations(messages)
            latest = observations[-1] if observations else {}
            if latest.get("evidence"):
                return self.call("request_clarification", {
                    "answer": "请补充身份信息。", "citations": [],
                    "information_gaps": ["身份未知"], "questions": ["是否已经转正？"],
                })
            if latest.get("error", {}).get("code") == "FINALIZATION_REQUIRED":
                return self.call("finish_answer", {
                    "task_status": "incomplete", "answer": "已读取住宿标准，尚未核实适用身份；本轮查证次数已用完。",
                    "reason_code": "TOOL_CALL_LIMIT", "citations": [], "information_gaps": ["适用身份尚未核实"],
                })
            return super().choose(messages)

    runs, request, job, runtime = setup(knowledge, LateClarificationChat())
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["execution_status"] == "finished" and record["task_status"] == "incomplete"
    assert record["clarification_count"] == 0 and record["tool_calls"] == 2
    assert "身份" in record["result"]["information_gaps"][0]


def test_closing_does_not_accept_unread_citation(knowledge):
    knowledge.settings.max_tool_calls = 2

    class FabricatedClosingChat(RecordingChat):
        def choose(self, messages):
            observations = self.observations(messages)
            latest = observations[-1] if observations else {}
            if latest.get("evidence"):
                evidence = latest["evidence"][0]
                return self.call("finish_answer", {
                    "task_status": "answered", "answer": "伪造金额。", "reason_code": "TEST",
                    "citations": [{"evidence_id": evidence["evidence_id"], "quote": "不存在的制度原文"}],
                    "information_gaps": [],
                })
            if latest.get("error", {}).get("code") == "UNREAD_CITATION":
                return self.close(messages)
            return super().choose(messages)

    runs, request, job, runtime = setup(knowledge, FabricatedClosingChat())
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "answered" and record["tool_calls"] == 2
    assert "500元" in record["result"]["answer"]
    assert knowledge.db.one(
        "SELECT payload FROM trace_events WHERE run_id=%s AND type='tool.failed'", (record["id"],)
    )["payload"]["error"]["code"] == "UNREAD_CITATION"


def test_offline_query_uses_question_instead_of_new_budget_system_message(knowledge):
    runs, request, job, runtime = setup(knowledge, MockChat())
    runtime.run(job)
    started = knowledge.db.one(
        "SELECT payload FROM trace_events WHERE run_id=%s AND type='tool.started' ORDER BY seq LIMIT 1",
        (request["run_id"],),
    )["payload"]
    assert started["arguments"]["queries"] == ["上海住宿上限多少？"]
    assert runs.get(request["run_id"])["result"]["reason_code"] == "MOCK_ONLY"


def test_offline_persistent_failure_closes_at_tool_cap(knowledge):
    knowledge.settings.max_tool_calls = 3
    runs, request, job, runtime = setup(knowledge, MockChat())
    runtime.faults = {"read": "always"}
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "incomplete" and record["tool_calls"] == 3
    assert record["result"]["reason_code"] == "TOOL_CALL_LIMIT"
    assert record["model_calls"] < record["config"]["max_model_calls"]
    assert not record["result"]["citations"] and "离线演示" in record["result"]["answer"]
