"""DeepSeek wire protocol and private transcript regressions; never call the real API.

PG cases create UUID-isolated fixture resources and delete only those resources.
Scripted reasoning markers verify transport/privacy, not model reasoning quality.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
from langgraph.checkpoint.postgres import PostgresSaver
from openai import APITimeoutError, BadRequestError
from openai.types.chat import ChatCompletion
import psycopg
from psycopg.rows import dict_row
import pytest

from xiaoying.agent.runtime import AgentRuntime
from xiaoying.api import create_app
from xiaoying.config import Settings
from xiaoying.domain import AppError
from xiaoying.providers import chat as chat_module
from xiaoying.providers.embeddings import MockReranker
from xiaoying.rag.retrieval import Retriever
from xiaoying.services.runs import RunService

from test_review_knowledge import knowledge as knowledge_fixture
from test_review_runtime import ClarifyingChat, ScriptedChat, setup
from test_review_storage import db as database_fixture

db = database_fixture
knowledge = knowledge_fixture
PRIVATE = "PRIVATE_THINKING_REVIEW_SENTINEL"


def completion(message=None, finish_reason="stop"):
    response = ChatCompletion.model_validate(
        {
            "id": "review-only-response",
            "object": "chat.completion",
            "created": 0,
            "model": "review-only-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": message
                    or {
                        "role": "assistant",
                        "content": '{"resolved_question":"住宿标准","expanded_queries":[],"unresolved_slots":[]}',
                        "reasoning_content": PRIVATE,
                    },
                }
            ],
        }
    )
    # DeepSeek has additional finish reasons; the default SDK transport decodes permissively.
    response.choices[0].finish_reason = finish_reason
    return response


@pytest.fixture
def sdk(monkeypatch):
    requests, responses = [], []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        assert responses, "The test must explicitly provide each SDK response"
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        chat_module,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    settings = Settings(
        _env_file=None,
        provider_mode="live",
        pdf_backend="mineru",
        deepseek_api_key="review-not-a-real-key",
        deepseek_thinking=True,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=16384,
    )
    return SimpleNamespace(requests=requests, responses=responses, settings=settings)


def tool_call(name, args, call_id="review-call"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def assistant(calls=None, marker=PRIVATE, content=""):
    result = {"role": "assistant", "content": content, "reasoning_content": marker}
    if calls is not None:
        result["tool_calls"] = calls
    return result


def finish(call_id="review-finish"):
    return assistant(
        [
            tool_call(
                "finish_answer",
                {
                    "task_status": "incomplete",
                    "reason_code": "REVIEW_SCRIPTED",
                    "answer": "这是协议测试的公开结果。",
                    "citations": [],
                    "information_gaps": [],
                },
                call_id,
            )
        ],
        marker=f"{PRIVATE}:{call_id}",
    )


def test_thinking_request_uses_auto_and_preserves_entire_assistant_message(sdk):
    long_reasoning = (PRIVATE + "\n保持空格  与换行\n") * 500
    message = assistant([tool_call("read_document", {"document_id": "DREV"})], long_reasoning, None)
    sdk.responses.append(completion(message, "tool_calls"))
    result = chat_module.DeepSeekChat(sdk.settings).decide([{"role": "user", "content": "住宿标准"}])
    request = sdk.requests[0]
    assert request["extra_body"]["thinking"]["type"] == "enabled"
    assert request.get("tool_choice", "auto") == "auto"
    assert request["max_tokens"] == 16384
    assert request["reasoning_effort"] == "high"
    assert result["reasoning_content"] == long_reasoning
    assert result["tool_calls"] == message["tool_calls"]
    assert result["content"] in (None, "")


def test_subsequent_sdk_request_keeps_reasoning_from_both_tool_and_plain_turns(sdk):
    previous = [
        {"role": "user", "content": "first question"},
        assistant([tool_call("search_knowledge", {"queries": ["住宿"]}, "first")], PRIVATE + ":tool"),
        {"role": "tool", "tool_call_id": "first", "content": "{}"},
        assistant(marker=PRIVATE + ":plain", content="model draft"),
        {"role": "user", "content": "next question"},
    ]
    sdk.responses.append(completion(assistant(content="another draft")))
    result = chat_module.DeepSeekChat(sdk.settings).decide(previous)
    assert sdk.requests[0]["messages"][1:] == previous
    assert result["reasoning_content"] == PRIVATE


def test_non_thinking_mode_remains_explicit_and_uses_required_tool(sdk):
    sdk.settings.deepseek_thinking = False
    sdk.responses.append(completion())
    chat_module.DeepSeekChat(sdk.settings).decide([{"role": "user", "content": "test"}])
    assert sdk.requests[0]["extra_body"]["thinking"]["type"] == "disabled"
    assert sdk.requests[0]["tool_choice"] == "required"


def test_expansion_has_shared_budget_but_never_exports_reasoning_or_unknown_fields(sdk):
    payload = {
        "resolved_question": "住宿标准",
        "expanded_queries": [],
        "unresolved_slots": [],
        "reasoning_content": PRIVATE + ":in-json",
        "internal_debug": PRIVATE + ":debug",
    }
    sdk.responses.append(completion(assistant(content=json.dumps(payload))))
    result = chat_module.DeepSeekChat(sdk.settings).expand("住宿标准", [])
    assert sdk.requests[0]["max_tokens"] == sdk.settings.deepseek_max_tokens
    assert not sdk.requests[0].get("tools")
    assert set(result) == {"resolved_question", "expanded_queries", "unresolved_slots"}
    assert PRIVATE not in json.dumps(result)


@pytest.mark.parametrize("method", ["decide", "expand"])
@pytest.mark.parametrize(
    "reason,code",
    [
        ("length", "MODEL_OUTPUT_TRUNCATED"),
        ("aborted", "MODEL_RESPONSE_INCOMPLETE"),
        ("insufficient_system_resource", "MODEL_RESPONSE_INCOMPLETE"),
        ("content_filter", "MODEL_RESPONSE_INCOMPLETE"),
    ],
)
def test_truncated_or_interrupted_response_cannot_be_accepted(sdk, method, reason, code):
    sdk.responses.append(completion(finish() if method == "decide" else None, reason))
    chat = chat_module.DeepSeekChat(sdk.settings)
    with pytest.raises(AppError) as error:
        chat.decide([]) if method == "decide" else chat.expand("test", [])
    assert error.value.code == code
    assert PRIVATE not in error.value.message


@pytest.mark.parametrize("method", ["decide", "expand"])
@pytest.mark.parametrize("timeout", [False, True])
def test_provider_error_body_is_never_exposed_to_public_error(sdk, method, timeout):
    request = httpx.Request("POST", "https://review.invalid/chat/completions")
    error = (
        APITimeoutError(request=request)
        if timeout
        else BadRequestError(
            PRIVATE,
            response=httpx.Response(400, request=request),
            body={"reasoning_content": PRIVATE},
        )
    )
    sdk.responses.append(error)
    chat = chat_module.DeepSeekChat(sdk.settings)
    with pytest.raises(AppError) as caught:
        chat.decide([]) if method == "decide" else chat.expand("test", [])
    assert caught.value.code == ("TIMEOUT" if timeout else "MODEL_API_ERROR")
    assert PRIVATE not in str(caught.value)
    assert caught.value.__suppress_context__


def assert_closed(messages):
    pending = set()
    for message in messages:
        role = message["role"]
        if role == "tool":
            assert message["tool_call_id"] in pending, "Unmatched or duplicate tool result"
            pending.remove(message["tool_call_id"])
        else:
            assert not pending, "A new message interrupted pending tool results"
            if role == "assistant":
                ids = [call["id"] for call in message.get("tool_calls", [])]
                assert len(ids) == len(set(ids)), "Duplicate tool IDs"
                pending.update(ids)
    assert not pending, "The next model request has missing tool results"


class SequenceChat:
    def __init__(self, responses):
        self.responses = deepcopy(responses)
        self.inputs = []
        self.expansions = 0

    def expand(self, question, history):
        self.expansions += 1
        return {"resolved_question": question, "expanded_queries": [], "unresolved_slots": []}

    def decide(self, messages):
        assert_closed(messages)
        self.inputs.append(deepcopy(messages))
        assert self.responses, "Unexpected extra model decision"
        return self.responses.pop(0)


class RecordingThinkingMixin:
    def __init__(self):
        super().__init__()
        self.inputs = []

    def call(self, name, args):
        result = super().call(name, args)
        result["reasoning_content"] = f"{PRIVATE}:{self.decisions}"
        return result

    def decide(self, messages):
        assert_closed(messages)
        self.inputs.append(deepcopy(messages))
        return super().decide(messages)


class ThinkingChat(RecordingThinkingMixin, ScriptedChat):
    pass


class ClarifyingThinkingChat(RecordingThinkingMixin, ClarifyingChat):
    pass


def saved_messages(h, run_id):
    with psycopg.connect(h.settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        record = PostgresSaver(conn).get_tuple({"configurable": {"thread_id": run_id}})
    assert record is not None
    return record.checkpoint["channel_values"]["messages"]


def setup_live(h, chat):
    kb = h.create()
    h.initial(kb)
    h.settings.provider_mode = "live"
    h.settings.pdf_backend = "mineru"
    runs = RunService(h.db, h.settings, h.embedder)
    conversation = runs.create_conversation(kb, h.key())["id"]
    request = runs.create_run(conversation, "上海住宿上限多少？", h.key())
    runtime = AgentRuntime(h.db, h.settings, runs, Retriever(h.db, h.embedder, MockReranker()), chat=chat)
    return runs, request, h.claim(request), runtime


@pytest.mark.integration
def test_plain_assistant_turn_is_preserved_and_not_used_as_final_answer(knowledge):
    plain = assistant(marker=PRIVATE + ":no-tool", content="未经校验的草稿，不应成为最终回答。")
    chat = SequenceChat([plain, finish()])
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    assert plain in chat.inputs[1]
    record = runs.get(request["run_id"])
    assert record["task_status"] == "incomplete"
    assert record["result"]["answer"] == "这是协议测试的公开结果。"
    assert plain in saved_messages(knowledge, request["run_id"])


@pytest.mark.integration
def test_multi_tool_response_is_closed_with_one_refusal_per_call(knowledge):
    calls = [
        tool_call("search_knowledge", {"queries": ["住宿"]}, "multi-search"),
        tool_call("request_clarification", {}, "multi-clarify"),
    ]
    batch = assistant(calls, PRIVATE + ":multi")
    chat = SequenceChat([batch, finish()])
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    next_messages = chat.inputs[1]
    assert batch in next_messages
    responses = [m for m in next_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in responses] == ["multi-search", "multi-clarify"]
    assert all(json.loads(m["content"]).get("error") for m in responses)
    record = runs.get(request["run_id"])
    assert record["tool_calls"] == 0 and record["clarification_count"] == 0
    assert record["task_status"] == "incomplete"


@pytest.mark.integration
@pytest.mark.parametrize(
    "bad_calls",
    [
        [{"type": "function", "function": {"name": "read_document", "arguments": "{}"}}],
        [tool_call("read_document", {}, "duplicate"), tool_call("read_document", {}, "duplicate")],
    ],
)
def test_invalid_tool_identifiers_fail_before_execution(knowledge, bad_calls):
    chat = SequenceChat([assistant(bad_calls)])
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["task_status"] == "incomplete"
    assert record["result"]["reason_code"] == "INVALID_MODEL_OUTPUT"
    assert record["tool_calls"] == 0


@pytest.mark.integration
def test_clarification_resume_keeps_private_reasoning_and_closed_tool_result(knowledge):
    chat = ClarifyingThinkingChat()
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    rid = request["run_id"]
    assert runs.get(rid)["execution_status"] == "waiting"
    before = saved_messages(knowledge, rid)
    assert_closed(before)
    resume = runs.clarify(rid, {"action": "submit", "content": "已转正"}, knowledge.key())
    runtime.run(knowledge.claim(resume))
    after = chat.inputs[-1]
    assert after[: len(before)] == before
    assert any(m["role"] == "user" and "已转正" in m["content"] for m in after[len(before) :])
    assert runs.get(rid)["task_status"] == "insufficient"


@pytest.mark.integration
def test_new_question_reuses_complete_private_history_including_plain_turn(knowledge):
    first_plain = assistant(marker=PRIVATE + ":first-plain", content="第一轮中间输出")
    first_chat = SequenceChat([first_plain, finish("first-finish")])
    runs, request, job, runtime = setup_live(knowledge, first_chat)
    runtime.run(job)
    previous = saved_messages(knowledge, request["run_id"])
    assert_closed(previous)
    cid = runs.get(request["run_id"])["conversation_id"]
    second = runs.create_run(cid, "继续问住宿标准", knowledge.key())
    second_chat = SequenceChat([finish("second-finish")])
    AgentRuntime(knowledge.db, knowledge.settings, runs, runtime.retriever, chat=second_chat).run(
        knowledge.claim(second)
    )
    messages = second_chat.inputs[0]
    assert messages[: len(previous)] == previous
    assert first_plain in messages
    assert any(m["role"] == "user" and "继续问住宿标准" in m["content"] for m in messages[len(previous) :])


@pytest.mark.integration
@pytest.mark.parametrize("change", ["version", "thinking"])
def test_new_question_starts_fresh_private_context_after_version_or_mode_change(knowledge, change):
    runs, request, job, runtime = setup_live(knowledge, SequenceChat([finish("old-context")]))
    runtime.run(job)
    previous = runs.get(request["run_id"])
    if change == "version":
        knowledge.build(knowledge.start(previous["kb_id"], previous["kb_version_id"]))
    else:
        knowledge.settings.deepseek_thinking = False
    second = runs.create_run(previous["conversation_id"], "新问题", knowledge.key())
    chat = SequenceChat([finish("fresh-context")])
    AgentRuntime(knowledge.db, knowledge.settings, runs, runtime.retriever, chat=chat).run(
        knowledge.claim(second)
    )
    messages = chat.inputs[0]
    assert not any(m["role"] == "assistant" for m in messages)
    assert PRIVATE not in json.dumps(messages)
    assert "这是协议测试的公开结果。" in json.dumps(messages, ensure_ascii=False)


@pytest.mark.parametrize(
    "old_messages",
    [
        [{"role": "assistant", "content": "old response without reasoning"}],
        [assistant([tool_call("read_document", {}, "unclosed")])],
        [
            assistant([tool_call("read_document", {}, "actual")]),
            {"role": "tool", "tool_call_id": "wrong", "content": "{}"},
        ],
    ],
)
def test_legacy_or_unclosed_previous_checkpoint_falls_back_to_fresh_context(old_messages):
    config = Settings(_env_file=None, provider_mode="live", pdf_backend="mineru").run_config()
    previous = {"id": "previous", "kb_version_id": "version", "config": config}
    db = SimpleNamespace(one=lambda *args: previous)
    runtime = AgentRuntime(db, None, None, None)
    graph = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(values={"messages": old_messages}, next=())
    )
    record = {
        "id": "new",
        "conversation_id": "conversation",
        "created_at": "2026-11-01",
        "kb_version_id": "version",
        "config": config,
        "question": "new question",
    }
    assert runtime.previous_messages(record, graph) == []


@pytest.mark.integration
def test_previous_run_evidence_cannot_authorize_new_answer(knowledge):
    runs, request, job, runtime = setup_live(knowledge, ThinkingChat())
    runtime.run(job)
    original = runs.get(request["run_id"])
    citation = original["result"]["citations"][0]
    second = runs.create_run(original["conversation_id"], "这个标准仍适用吗", knowledge.key())
    stale_finish = assistant(
        [
            tool_call(
                "finish_answer",
                {
                    "task_status": "answered",
                    "answer": "直接沿用旧引用",
                    "reason_code": "REVIEW",
                    "citations": [{"evidence_id": citation["evidence_id"], "quote": citation["quote"]}],
                    "information_gaps": [],
                },
                "stale-finish",
            )
        ]
    )
    chat = SequenceChat([stale_finish, finish("safe-finish")])
    AgentRuntime(knowledge.db, knowledge.settings, runs, runtime.retriever, chat=chat).run(
        knowledge.claim(second)
    )
    observations = [json.loads(m["content"]) for m in chat.inputs[1] if m["role"] == "tool"]
    assert observations[-1]["error"]["code"] == "UNREAD_CITATION"
    assert runs.get(second["run_id"])["result"]["citations"] == []


@pytest.mark.integration
def test_model_action_saved_before_checkpoint_replays_reasoning_without_recalling_provider(
    knowledge, monkeypatch
):
    chat = ThinkingChat()
    runs, request, job, runtime = setup(knowledge, chat)
    complete = runs.complete_action
    crashed = False

    class CrashAfterAction(Exception):
        pass

    def crash_once(job, action_id, output):
        nonlocal crashed
        complete(job, action_id, output)
        if not crashed and action_id.startswith("model:"):
            crashed = True
            raise CrashAfterAction()

    monkeypatch.setattr(runs, "complete_action", crash_once)
    with pytest.raises(CrashAfterAction):
        runtime.run(job)
    assert chat.decisions == 1
    runtime.run(knowledge.claim(request))
    assert chat.decisions == 3 and chat.expansions == 1
    assert runs.get(request["run_id"])["task_status"] == "answered"
    assert any(m.get("reasoning_content") == PRIVATE + ":1" for m in chat.inputs[1])


@pytest.mark.integration
def test_private_reasoning_stays_out_of_run_conversation_job_and_trace_api(knowledge):
    chat = ThinkingChat()
    runs, request, job, runtime = setup(knowledge, chat)
    runtime.run(job)
    rid = request["run_id"]
    assert PRIVATE in json.dumps(saved_messages(knowledge, rid))
    actions = knowledge.db.all("SELECT output FROM run_actions WHERE run_id=%s AND kind='model'", (rid,))
    assert PRIVATE in json.dumps(actions)
    cid = runs.get(rid)["conversation_id"]
    with TestClient(
        create_app(settings=knowledge.settings, db=knowledge.db, embedder=knowledge.embedder)
    ) as client:
        for path in [
            f"/api/v1/runs/{rid}",
            f"/api/v1/runs/{rid}/events",
            f"/api/v1/conversations/{cid}",
            f"/api/v1/jobs/{request['job_id']}",
        ]:
            response = client.get(path)
            assert response.status_code == 200
            assert PRIVATE not in response.text
            assert "reasoning_content" not in response.text


@pytest.mark.integration
@pytest.mark.parametrize(
    "field,value",
    [
        ("deepseek_thinking", False),
        ("deepseek_reasoning_effort", "low"),
        ("deepseek_max_tokens", 8192),
    ],
)
def test_live_resume_rejects_changed_thinking_configuration(knowledge, field, value):
    chat = SequenceChat([finish()])
    runs, request, job, runtime = setup_live(knowledge, chat)
    original = runs.get(request["run_id"])["config"][field]
    assert original != value
    setattr(knowledge.settings, field, value)
    runtime.run(job)
    record = runs.get(request["run_id"])
    assert record["result"]["reason_code"] == "CONFIG_MISMATCH"
    assert not chat.inputs and chat.expansions == 0
    assert record["config"][field] == original


@pytest.mark.integration
def test_old_live_snapshot_cannot_silently_enable_thinking(knowledge):
    chat = SequenceChat([finish()])
    runs, request, job, runtime = setup_live(knowledge, chat)
    with knowledge.db.tx() as connection:
        connection.execute(
            "UPDATE runs SET config=config-'deepseek_thinking' WHERE id=%s", (request["run_id"],)
        )
    runtime.run(job)
    assert runs.get(request["run_id"])["result"]["reason_code"] == "CONFIG_MISMATCH"
    assert not chat.inputs and chat.expansions == 0
