import json
from typing import TypedDict

import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from xiaoying.domain import AppError, LeaseLost
from xiaoying.providers.bounded import bounded
from xiaoying.providers.chat import (
    THINKING_CONFIG_KEYS,
    DeepSeekChat,
    MockChat,
    validate_tool_calls,
    validate_tool_history,
)
from xiaoying.services.runs import final_result
from xiaoying.storage.checkpoints import FencedPostgresSaver


class State(TypedDict, total=False):
    question: str
    history: list
    messages: list
    turn: int
    expanded: dict
    pending: dict
    route: str
    result: dict
    errors: dict
    evidence_epoch: int


class AgentRuntime:
    def __init__(self, db, settings, runs, retriever, chat=None, faults=None):
        self.db, self.settings, self.runs, self.retriever = db, settings, runs, retriever
        self.chat = chat
        self.faults = faults or {}
        self.read_attempts = 0

    def run(self, job):
        record = self.runs.start(job)
        if record is None:
            return
        rid = record["id"]
        cfg = record["config"]
        try:
            if (
                cfg["provider_mode"] != self.settings.provider_mode
                or cfg["deepseek_model"] != self.settings.deepseek_model
                or cfg["reranker_model"] != self.settings.reranker_model
                or cfg["reranker_revision"] != self.settings.reranker_revision
                or (
                    cfg["provider_mode"] == "live"
                    and any(cfg.get(key) != getattr(self.settings, key) for key in THINKING_CONFIG_KEYS)
                )
            ):
                raise AppError("CONFIG_MISMATCH", "Worker 的模型配置与任务快照不符", 409)
            chat = self.chat or (
                MockChat() if self.settings.provider_mode == "mock" else DeepSeekChat(self.settings)
            )
            with psycopg.connect(
                self.settings.database_url,
                autocommit=True,
                row_factory=dict_row,
                options="-c lock_timeout=2000 -c statement_timeout=10000 -c idle_in_transaction_session_timeout=10000",
            ) as conn:
                saver = FencedPostgresSaver(conn, job["id"], job["lease_token"])
                graph = self.make_graph(job, chat).compile(checkpointer=saver)
                config = {"configurable": {"thread_id": rid}, "recursion_limit": cfg["max_graph_steps"] + 8}
                snapshot = graph.get_state(config)
                if snapshot.values:
                    clar = self.db.one("SELECT status,response FROM clarifications WHERE run_id=%s", (rid,))
                    interrupted = any(t.interrupts for t in snapshot.tasks)
                    input_value = (
                        Command(resume=clar["response"])
                        if interrupted and clar and clar["status"] == "submitted"
                        else None
                    )
                else:
                    history = self.db.all(
                        "SELECT role,content FROM messages WHERE conversation_id=%s AND run_id!=%s ORDER BY id DESC LIMIT 12",
                        (record["conversation_id"], rid),
                    )
                    history = list(reversed(history))
                    input_value = {
                        "question": record["question"],
                        "history": history,
                        "messages": self.previous_messages(record, graph),
                        "turn": 0,
                        "errors": {},
                        "evidence_epoch": record["evidence_epoch"],
                    }
                output = graph.invoke(input_value, config=config, durability="sync")
                snapshot = graph.get_state(config)
                if snapshot.next:
                    if any(t.interrupts for t in snapshot.tasks):
                        clar = self.db.one("SELECT payload FROM clarifications WHERE run_id=%s", (rid,))
                        self.runs.waiting(job, clar["payload"])
                    else:
                        raise AppError("GRAPH_UNFINISHED", "状态图未完成且未进入补充等待", 500)
                else:
                    self.runs.finish(job, output["result"])
        except LeaseLost:
            raise
        except AppError as exc:
            self.runs.finish(job, final_result("incomplete", exc.code, exc.message))

    def previous_messages(self, record, graph):
        """Continue a compatible, closed provider conversation from its private checkpoint.

        Each question still has its own run, budget and evidence reads. Legacy/incomplete
        conversations or a version/model switch start fresh with public history in expand().
        """
        cfg = record["config"]
        if cfg["provider_mode"] != "live":
            return []
        previous = self.db.one(
            """SELECT id,kb_version_id,config FROM runs
            WHERE conversation_id=%s AND id!=%s AND execution_status='finished' AND created_at<%s
            ORDER BY created_at DESC LIMIT 1""",
            (record["conversation_id"], record["id"], record["created_at"]),
        )
        keys = ("provider_mode", "deepseek_model", *THINKING_CONFIG_KEYS)
        if (
            not previous
            or previous["kb_version_id"] != record["kb_version_id"]
            or any(previous["config"].get(key) != cfg.get(key) for key in keys)
        ):
            return []
        snapshot = graph.get_state({"configurable": {"thread_id": str(previous["id"])}})
        messages = snapshot.values.get("messages", [])
        if snapshot.next or not messages:
            return []
        try:
            validate_tool_history(messages, cfg["deepseek_thinking"])
        except AppError:
            return []
        return messages + [{"role": "user", "content": record["question"]}]

    def make_graph(self, job, chat):
        rid = job["resource_id"]
        cfg = self.runs.get(rid)["config"]

        def step(state, name):
            key = f"step:{state['turn']}:{name}"
            cached, _ = self.runs.reserve_action(job, key, "step", {"node": name})
            if not cached:
                self.runs.complete_action(job, key, {})

        def model_call(action_id, body, call):
            cached, output = self.runs.reserve_action(job, action_id, "model", body)
            if cached:
                return output
            output = bounded(call, min(cfg["model_timeout_seconds"], self.runs.remaining(rid)))
            self.runs.complete_action(job, action_id, output)
            return output

        def expand(state):
            step(state, "expand")
            expanded = model_call(
                f"expand:{state['turn']}",
                {"question": state["question"], "history": state["history"]},
                lambda: chat.expand(state["question"], state["history"]),
            )
            if not isinstance(expanded, dict) or not isinstance(expanded.get("resolved_question"), str):
                raise AppError("INVALID_EXPANSION", "模型问题扩展格式无效", 502)
            queries = expanded.get("expanded_queries", [])
            if not isinstance(queries, list) or any(not isinstance(x, str) for x in queries):
                raise AppError("INVALID_EXPANSION", "扩展查询格式无效", 502)
            expanded["expanded_queries"] = list(
                dict.fromkeys([state["question"][:2000], expanded["resolved_question"][:2000]] + queries)
            )[:3]
            self.runs.trace(job, f"expanded:{state['turn']}", "query.expanded", expanded)
            messages = state["messages"] or [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"history": state["history"], "question": state["question"]}, ensure_ascii=False
                    ),
                }
            ]
            messages = messages + [
                {
                    "role": "system",
                    "content": json.dumps(
                        {
                            **expanded,
                            "run_id": str(rid),
                            "kb_version_id": str(self.runs.get(rid)["kb_version_id"]),
                            "evidence_epoch": state["evidence_epoch"],
                            "clarification_used": self.runs.get(rid)["clarification_count"],
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
            return {"expanded": expanded, "messages": messages}

        def decide(state):
            step(state, "decide")
            response = model_call(
                f"model:{state['turn']}", state["messages"], lambda: chat.decide(state["messages"])
            )
            calls = validate_tool_calls(response)
            if len(calls) != 1:
                # Keep reasoning even without tools. Close every rejected parallel call by ID.
                messages = state["messages"] + [response]
                for call in calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {
                                    "error": {
                                        "code": "MULTIPLE_TOOL_CALLS_UNSUPPORTED",
                                        "message": "本轮所有工具均未执行，请每次只选择一个工具",
                                        "retryable": True,
                                    }
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                self.runs.trace(
                    job,
                    f"decision:{state['turn']}",
                    "agent.decision",
                    {
                        "step": state["turn"],
                        "tool_count": len(calls),
                        "accepted": False,
                    },
                )
                return {
                    "messages": messages,
                    "pending": {"invalid": "必须且只能选择一个工具；最终回答也须调用 finish_answer"},
                    "route": "execute",
                }
            self.runs.trace(
                job,
                f"decision:{state['turn']}",
                "agent.decision",
                {
                    "tool": calls[0].get("function", {}).get("name"),
                    "step": state["turn"],
                    "tool_call_id": calls[0].get("id"),
                },
            )
            return {"messages": state["messages"] + [response], "pending": calls[0], "route": "execute"}

        def execute(state):
            step(state, "execute")
            pending = state["pending"]
            turn = state["turn"]
            if pending.get("invalid"):
                return {
                    "turn": turn + 1,
                    "route": "decide",
                    "messages": state["messages"] + [{"role": "system", "content": pending["invalid"]}],
                }
            name = pending.get("function", {}).get("name")
            action_id = f"tool:{turn}"
            errors = dict(state["errors"])
            route = "decide"
            result = None
            try:
                args = json.loads(pending["function"].get("arguments", "{}"))
                if not isinstance(args, dict):
                    raise AppError("INVALID_ARGUMENTS", "工具参数须为对象", 422)
                reason = str(args.pop("reason", ""))[:300]
                if name not in {
                    "search_knowledge",
                    "read_document",
                    "request_clarification",
                    "finish_answer",
                }:
                    raise AppError("UNKNOWN_TOOL", "未授权的工具", 422)
                self.runs.trace(
                    job,
                    f"tool.started:{turn}",
                    "tool.started",
                    {
                        "tool": name,
                        "arguments": args,
                        "reason": reason,
                        "tool_call_id": pending["id"],
                        "step": turn,
                    },
                )
                if name in {"search_knowledge", "read_document"}:
                    cached, output = self.runs.reserve_action(
                        job, action_id, "tool", {"name": name, "args": args}
                    )
                    if not cached:
                        try:
                            run = self.runs.get(rid)
                            if name == "search_knowledge":
                                output = bounded(
                                    lambda: self.retriever.search(run["kb_id"], run["kb_version_id"], **args),
                                    min(cfg["tool_timeout_seconds"], self.runs.remaining(rid)),
                                )
                            else:
                                self.read_attempts += 1
                                fault = self.faults.get("read", 0)
                                if fault == "always" or self.read_attempts <= int(fault):
                                    raise AppError("TIMEOUT", "测试注入：原文读取超时", 504, True)
                                output = bounded(
                                    lambda: self.runs.read(job, action_id, args),
                                    min(cfg["read_timeout_seconds"], self.runs.remaining(rid)),
                                )
                        except LeaseLost:
                            raise
                        except Exception as exc:
                            output = {
                                "error": {
                                    "code": getattr(exc, "code", type(exc).__name__),
                                    "message": getattr(exc, "message", "工具执行失败"),
                                    "retryable": getattr(exc, "retryable", False),
                                }
                            }
                        self.runs.complete_action(job, action_id, output)
                    if output.get("error"):
                        errors[name] = output["error"]
                    else:
                        errors.pop(name, None)
                        if name == "read_document" and output.get("evidence"):
                            # Direct document reads can recover a failed search; actual evidence is now available.
                            errors.pop("search_knowledge", None)
                elif name == "request_clarification":
                    if errors:
                        raise AppError(
                            "UNRESOLVED_TOOL_FAILURE", "必要工具尚未成功，请补救或结束为执行未完成", 422
                        )
                    output = self.runs.prepare_clarification(job, action_id, args)
                    route = "wait"
                else:
                    if errors and args.get("task_status") in {"answered", "insufficient"}:
                        raise AppError(
                            "UNRESOLVED_TOOL_FAILURE", "工具故障尚未恢复，不能宣称完成必要查证", 422
                        )
                    result = self.runs.validate_result(rid, args)
                    output = {"accepted": True}
                    route = "finish"
                summary = {
                    "tool": name,
                    "tool_call_id": pending["id"],
                    "step": turn,
                    "degraded": output.get("degraded", False),
                }
                if "evidence" in output:
                    summary["evidence"] = [
                        {k: e[k] for k in ("evidence_id", "document_id", "chunk_id", "locator")}
                        for e in output["evidence"]
                    ]
                elif "candidates" in output:
                    summary.update(output)
                elif output.get("error"):
                    summary["error"] = output["error"]
                else:
                    summary["result"] = output
                self.runs.trace(
                    job,
                    f"tool.completed:{turn}",
                    "tool.failed" if output.get("error") else "tool.completed",
                    summary,
                )
            except LeaseLost:
                raise
            except AppError as exc:
                if exc.code == "EXECUTION_LIMIT":
                    raise
                output = {"error": {"code": exc.code, "message": exc.message, "retryable": exc.retryable}}
                self.runs.trace(
                    job, f"tool.failed:{turn}", "tool.failed", {"tool": name, "step": turn, **output}
                )
            except (ValueError, KeyError, TypeError):
                output = {
                    "error": {
                        "code": "INVALID_ARGUMENTS",
                        "message": "工具调用格式不合法",
                        "retryable": False,
                    }
                }
                self.runs.trace(
                    job, f"tool.failed:{turn}", "tool.failed", {"tool": name, "step": turn, **output}
                )
            update = {
                "turn": turn + 1,
                "route": route,
                "errors": errors,
                "messages": state["messages"]
                + [
                    {
                        "role": "tool",
                        "tool_call_id": pending["id"],
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                ],
            }
            if result:
                update["result"] = result
            if route == "wait":
                update["pending"] = output
            return update

        def wait(state):
            # interrupt replays this node. No non-idempotent work before it.
            response = interrupt(state["pending"])
            step(state, "resume")
            changed = response["evidence_epoch"] != state["evidence_epoch"]
            text = response.get("content", "")
            question = state["question"] + "\n用户补充：" + text
            messages = (
                [] if changed else state["messages"] + [{"role": "user", "content": "本次唯一补充：" + text}]
            )
            return {
                "question": question,
                "messages": messages,
                "turn": state["turn"] + 1,
                "evidence_epoch": response["evidence_epoch"],
                "errors": {} if changed else state["errors"],
            }

        graph = StateGraph(State)
        graph.add_node("expand", expand)
        graph.add_node("decide", decide)
        graph.add_node("execute", execute)
        graph.add_node("wait", wait)
        graph.add_edge(START, "expand")
        graph.add_edge("expand", "decide")
        graph.add_edge("decide", "execute")
        graph.add_conditional_edges(
            "execute", lambda s: s["route"], {"decide": "decide", "wait": "wait", "finish": END}
        )
        graph.add_edge("wait", "expand")
        return graph
