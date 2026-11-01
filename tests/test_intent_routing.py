"""Intent routing regressions using isolated PostgreSQL and explicit model responses."""

import json

import pytest

from xiaoying.agent.runtime import AgentRuntime
from xiaoying.domain import AppError
from xiaoying.providers.chat import DeepSeekChat, MockChat, UNDERSTANDING_SYSTEM, validate_tool_history

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_runtime import ClarifyingChat, ScriptedChat, SimulatedCrash, setup
from test_review_storage import db as database_fixture
from test_thinking_review import (
    PRIVATE, SequenceChat, assistant, completion, finish, saved_messages, setup_live, tool_call,
    sdk as sdk_fixture,
)

db = database_fixture
knowledge = knowledge_fixture
sdk = sdk_fixture
pytestmark = pytest.mark.integration


def understanding(intent, question, **extra):
    return {"intent": intent, "resolved_question": question, "expanded_queries": [], "unresolved_slots": [], **extra}


def submit(h, runs, runtime, conversation, question, chat):
    request = runs.create_run(conversation, question, h.key())
    AgentRuntime(h.db, h.settings, runs, runtime.retriever, chat=chat).run(h.claim(request))
    return runs.get(request["run_id"])


@pytest.mark.parametrize("intent,question", [
    ("greeting", "你好"), ("thanks", "谢谢"), ("farewell", "再见"), ("capabilities", "你能做什么？"),
    ("unrelated", "讲个笑话"),
])
def test_social_turn_uses_one_sdk_request_no_tools_and_no_unchecked_model_answer(knowledge, sdk, intent, question):
    runs, seed, _, runtime = setup_live(knowledge, None)
    # A failed prior business question stays in public history, but must not resume.
    runs.finish(knowledge.claim(seed), {
        "task_status": "incomplete", "answer": "尚未查证年假制度。", "citations": [],
        "information_gaps": ["年假制度未读取"], "reason_code": "TEST_PRIOR_FAILURE",
    })
    cid = runs.get(seed["run_id"])["conversation_id"]
    payload = understanding(intent, "不可信扩展：住宿标准是9999元", expanded_queries=["重新查年假"],
                            direct_reply="不可信答案：住宿标准是9999元")
    sdk.responses.append(completion(assistant(content=json.dumps(payload, ensure_ascii=False))))
    result = submit(knowledge, runs, runtime, cid, question, DeepSeekChat(sdk.settings))
    assert result["task_status"] == "answered"
    assert result["tool_calls"] == 0 and result["model_calls"] == 1 and result["graph_steps"] == 1
    assert result["result"]["citations"] == [] and result["result"]["information_gaps"] == []
    assert result["result"]["reason_code"] == f"CONVERSATION_{intent.upper()}"
    assert "9999" not in result["result"]["answer"] and "年假" not in result["result"]["answer"]
    assert len(sdk.requests) == 1 and "tools" not in sdk.requests[0]
    sent = sdk.requests[0]["messages"]
    assert sent[0]["content"] == UNDERSTANDING_SYSTEM
    assert json.loads(sent[1]["content"])["question"] == question
    assert json.loads(sent[1]["content"])["history"], "The current greeting overrides prior business context"
    events = runs.conversation_trace(cid)["rounds"][-1]["events"]
    assert not any(e["type"].startswith("tool.") or e["type"] == "query.expanded" for e in events)
    assert next(e["payload"] for e in events if e["type"] == "intent.detected")["requires_knowledge"] is False
    assert runs.model_call(result["id"], 1)["response"]["intent"] == intent
    transcript = saved_messages(knowledge, result["id"])
    validate_tool_history(transcript, True)
    assert transcript[-1]["content"] == result["result"]["answer"]
    assert PRIVATE not in json.dumps(result)


@pytest.mark.parametrize("intent", ["knowledge", None, "direct_answer", ["greeting"]])
def test_business_and_unknown_intents_keep_retrieval_and_evidence_gate(knowledge, intent):
    class ClassifiedBusiness(ScriptedChat):
        def expand(self, question, history):
            return understanding(intent, question)

    runs, seed, _, runtime = setup(knowledge)
    cid = runs.get(seed["run_id"])["conversation_id"]
    runs.finish(knowledge.claim(seed), {"task_status": "incomplete", "answer": "测试种子", "citations": []})
    result = submit(knowledge, runs, runtime, cid, "你好，上海住宿上限多少？", ClassifiedBusiness())
    assert result["task_status"] == "answered" and result["tool_calls"] == 2
    assert result["model_calls"] == 4 and result["result"]["citations"]
    with pytest.raises(AppError, match="制度结论") as exc:
        runs.validate_result(result["id"], {
            "task_status": "answered", "answer": "可以报销9999元", "citations": [],
            "information_gaps": [], "reason_code": "CONVERSATION_GREETING", "intent": "greeting",
        })
    assert exc.value.code == "EVIDENCE_REQUIRED"


@pytest.mark.parametrize("intent,question", [("greeting", "你好"), ("unrelated", "讲个笑话")])
def test_business_nonbusiness_followup_share_session_and_keep_closed_context(knowledge, intent, question):
    class ClassifiedSequence(SequenceChat):
        def __init__(self, intent, responses):
            super().__init__(responses)
            self.intent = intent
            self.history = None

        def expand(self, question, history):
            self.history = history
            return understanding(self.intent, question)

    def chain(suffix):
        return [
            assistant([tool_call("search_knowledge", {"queries": ["上海住宿"]}, f"search-{suffix}")]),
            assistant([tool_call("read_document", {"document_id": "DREV"}, f"read-{suffix}")]),
            finish(f"finish-{suffix}"),
        ]

    runs, first, job, runtime = setup_live(knowledge, ClassifiedSequence("knowledge", chain("one")))
    runtime.run(job)
    cid = runs.get(first["run_id"])["conversation_id"]
    interlude = submit(knowledge, runs, runtime, cid, question, ClassifiedSequence(intent, []))
    followup_chat = ClassifiedSequence("knowledge", chain("three"))
    followup = submit(knowledge, runs, runtime, cid, "谢谢，那试用期呢？", followup_chat)
    assert interlude["tool_calls"] == 0 and followup["tool_calls"] == 2
    assert [r["run"]["id"] for r in runs.conversation_trace(cid)["rounds"]] == [first["run_id"], interlude["id"], followup["id"]]
    context = followup_chat.inputs[0]
    validate_tool_history(context, True)
    assert any(m.get("content") == interlude["result"]["answer"] for m in context)
    text = json.dumps(context, ensure_ascii=False)
    assert "上海住宿上限多少？" in text and "search-one" in text and "那试用期呢" in text
    assert any("上海住宿上限多少？" in json.dumps(m, ensure_ascii=False) for m in followup_chat.history)


@pytest.mark.parametrize("intent", ["greeting", "unrelated"])
def test_social_intent_cannot_discard_original_business_question_on_clarification(knowledge, intent):
    class ConfusedClarifyingChat(ClarifyingChat):
        def expand(self, question, history):
            return understanding(intent if "用户补充" in question else "knowledge", question)

    runs, request, job, runtime = setup(knowledge, ConfusedClarifyingChat())
    runtime.run(job)
    assert runs.get(request["run_id"])["execution_status"] == "waiting"
    response = runs.clarify(request["run_id"], {"action": "submit", "content": "你好"}, knowledge.key())
    runtime.run(knowledge.claim(response))
    result = runs.get(request["run_id"])
    assert result["task_status"] == "insufficient" and not result["result"]["reason_code"].startswith("CONVERSATION_")


def test_completed_intent_call_recovers_without_second_model_request(knowledge, sdk, monkeypatch):
    runs, request, job, runtime = setup_live(knowledge, DeepSeekChat(sdk.settings))
    # Give the first run a social input without editing any stored user message.
    runs.finish(job, {"task_status": "incomplete", "answer": "测试种子", "citations": []})
    cid = runs.get(request["run_id"])["conversation_id"]
    request = runs.create_run(cid, "你好", knowledge.key())
    sdk.responses.append(completion(assistant(content=json.dumps(understanding("greeting", "你好")))))
    complete_action = runs.complete_action

    def crash_after_save(job, action_id, output):
        complete_action(job, action_id, output)
        if action_id == "expand:0":
            raise SimulatedCrash("Intent output saved before graph checkpoint")

    monkeypatch.setattr(runs, "complete_action", crash_after_save)
    with pytest.raises(SimulatedCrash):
        runtime.run(knowledge.claim(request))
    monkeypatch.setattr(runs, "complete_action", complete_action)
    runtime.run(knowledge.claim(request))
    result = runs.get(request["run_id"])
    assert result["task_status"] == "answered" and result["tool_calls"] == 0
    assert result["model_calls"] == 1 and len(sdk.requests) == 1
    assert runs.model_call(request["run_id"], 1)["status"] == "completed"
    events = runs.conversation_trace(cid)["rounds"][-1]["events"]
    assert sum(e["type"] == "intent.detected" for e in events) == 1


@pytest.mark.parametrize("question", [
    "你好，上海住宿上限多少？", "谢谢，那试用期呢？", "再见之前，报销找谁？", "讲个笑话，然后告诉我上海住宿上限",
])
def test_mock_social_matching_never_swallows_mixed_business_input(question):
    assert MockChat().expand(question, [])["intent"] == "knowledge"


def test_mock_unrelated_request_skips_retrieval_in_the_real_graph(knowledge):
    runs, seed, _, runtime = setup(knowledge)
    cid = runs.get(seed["run_id"])["conversation_id"]
    runs.finish(knowledge.claim(seed), {"task_status": "incomplete", "answer": "测试种子", "citations": []})
    result = submit(knowledge, runs, runtime, cid, "讲个笑话", MockChat())
    assert result["task_status"] == "answered" and result["tool_calls"] == 0 and result["model_calls"] == 1
    assert result["result"]["reason_code"] == "CONVERSATION_UNRELATED"
    events = runs.conversation_trace(cid)["rounds"][-1]["events"]
    assert next(e["payload"] for e in events if e["type"] == "intent.detected")["intent"] == "unrelated"
