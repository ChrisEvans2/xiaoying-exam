import { useEffect, useState } from "react";
import type { ModelInvocation, Run, Trace } from "./types";

const toolLabels: Record<string, string> = {
  search_knowledge: "检索知识库",
  read_document: "读取原文",
  finish_answer: "提交答复",
  request_clarification: "请求补充",
};

const eventLabels: Record<string, string> = {
  "run.queued": "问题已提交",
  "run.started": "开始处理",
  "intent.detected": "需求判断",
  "query.expanded": "问题理解",
  "agent.decision": "模型决策",
  "model.started": "调用模型",
  "model.completed": "模型返回",
  "model.failed": "模型异常",
  "tool.started": "工具调用",
  "tool.completed": "工具返回",
  "tool.failed": "工具异常",
  "clarification.requested": "等待补充",
  "clarification.submitted": "收到补充",
  "run.finished": "最终结果",
  "evidence.version_changed": "切换资料版本",
  "action.unknown": "恢复执行",
};

type Tone = "input" | "tool" | "model" | "error" | "system";

export function eventTone(event: Trace): Tone {
  if (
    event.type === "tool.failed" ||
    event.type === "model.failed" ||
    (event.type === "run.finished" &&
      event.payload.task_status === "incomplete")
  )
    return "error";
  if (
    ["intent.detected", "query.expanded", "clarification.submitted"].includes(
      event.type,
    )
  )
    return "input";
  if (event.type.startsWith("tool.")) return "tool";
  if (event.type.startsWith("model.")) return "model";
  if (
    ["agent.decision", "run.finished", "clarification.requested"].includes(
      event.type,
    )
  )
    return "model";
  return "system";
}

const strings = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];

const intentLabels: Record<string, string> = {
  greeting: "问候 · 直接回应",
  thanks: "致谢 · 直接回应",
  farewell: "告别 · 直接回应",
  capabilities: "能力介绍 · 直接回应",
  unrelated: "非业务需求 · 引导咨询",
  knowledge: "业务咨询 · 进入查证",
};

const contentText = (value: unknown) => {
  if (typeof value !== "string") return JSON.stringify(value, null, 2);
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
};

function ModelContext({
  runId,
  callNumber,
}: {
  runId: string;
  callNumber: number;
}) {
  const [record, setRecord] = useState<ModelInvocation | null>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  const load = async () => {
    if (record || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `/api/v1/runs/${runId}/model-calls/${callNumber}`,
      );
      const value = await response.json();
      if (!response.ok)
        throw new Error(value.error?.message ?? "模型输入加载失败");
      setRecord(value);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };
  const roles: Record<string, string> = {
    system: "系统提示",
    user: "用户与历史",
    assistant: "模型消息",
    tool: "工具结果",
  };
  return (
    <details
      className="event-details model-context"
      onToggle={(e) => {
        if (e.currentTarget.open) void load();
      }}
    >
      <summary>输入上下文</summary>
      {loading && <p role="status">加载中…</p>}
      {error && <p role="alert">{error}</p>}
      {record && (
        <>
          <p className="context-note">
            {record.adapter === record.model
              ? record.model
              : `${record.adapter} · ${record.model}`}{" "}
            · 已隐藏内部思考字段
          </p>
          {record.request.messages?.map((message, i) => (
            <details
              key={i}
              className={`context-message context-${message.role}`}
            >
              <summary>
                {i + 1}. {roles[String(message.role)] ?? String(message.role)}
              </summary>
              {message.content != null && (
                <pre tabIndex={0}>{contentText(message.content)}</pre>
              )}
              {!!message.tool_calls && (
                <pre tabIndex={0}>
                  {JSON.stringify(message.tool_calls, null, 2)}
                </pre>
              )}
              {!!message.tool_call_id && (
                <p>工具调用 ID：{String(message.tool_call_id)}</p>
              )}
            </details>
          ))}
          <details className="context-message">
            <summary>请求参数</summary>
            <pre tabIndex={0}>{JSON.stringify(record.request, null, 2)}</pre>
          </details>
        </>
      )}
    </details>
  );
}

function EventContent({
  event,
  question,
  runId,
}: {
  event: Trace;
  question: string;
  runId: string;
}) {
  const payload = event.payload;
  const tool =
    typeof payload.tool === "string"
      ? (toolLabels[payload.tool] ?? payload.tool)
      : null;
  const args = payload.arguments as Record<string, unknown> | undefined;
  const error = payload.error as Record<string, unknown> | undefined;
  const queries = strings(
    event.type === "query.expanded" ? payload.expanded_queries : args?.queries,
  );

  return (
    <>
      {event.type.startsWith("model.") && (
        <>
          <p className="event-body">
            第 {String(payload.call_number)} 次 ·{" "}
            {payload.phase === "expand" ? "需求判断与问题理解" : "决策与答复"} ·{" "}
            {String(payload.model)}
          </p>
          {typeof payload.elapsed_ms === "number" && (
            <p className="event-body">
              耗时 {(payload.elapsed_ms / 1000).toFixed(2)} 秒
              {payload.recovered ? "（含恢复间隔）" : ""}
            </p>
          )}
          {event.type === "model.started" &&
            typeof payload.call_number === "number" && (
              <ModelContext runId={runId} callNumber={payload.call_number} />
            )}
          {payload.response != null && (
            <details className="event-details">
              <summary>返回内容</summary>
              <pre>{JSON.stringify(payload.response, null, 2)}</pre>
            </details>
          )}
        </>
      )}
      {tool && <p className="event-tool-name">{tool}</p>}
      {event.type === "intent.detected" &&
        typeof payload.intent === "string" && (
          <p className="event-body">
            {intentLabels[payload.intent] ?? payload.intent}
          </p>
        )}
      {event.type === "query.expanded" &&
        typeof payload.resolved_question === "string" &&
        payload.resolved_question.trim() !== question.trim() && (
          <p className="event-body">{payload.resolved_question}</p>
        )}
      {event.type === "clarification.submitted" &&
        typeof payload.content === "string" && (
          <p className="event-body">{payload.content}</p>
        )}
      {typeof payload.reason === "string" && payload.reason && (
        <p className="event-body">{payload.reason}</p>
      )}
      {queries.length > 0 && (
        <details className="event-details">
          <summary>查询语句 · {queries.length}</summary>
          <ul>
            {queries.map((query, i) => (
              <li key={i}>{query}</li>
            ))}
          </ul>
        </details>
      )}
      {typeof args?.document_id === "string" && (
        <p className="event-body">文档 {args.document_id}</p>
      )}
      {Array.isArray(payload.candidates) && (
        <p className="event-body">
          返回 {payload.candidates.length} 个候选片段
        </p>
      )}
      {Array.isArray(payload.evidence) && (
        <p className="event-body">读取 {payload.evidence.length} 条原文依据</p>
      )}
      {error && typeof error.message === "string" && (
        <p className="event-body">{error.message}</p>
      )}
      {event.type === "run.finished" && typeof payload.answer === "string" && (
        <details className="event-details">
          <summary>查看答复</summary>
          <p className="event-body">{payload.answer}</p>
        </details>
      )}
      <details className="event-details event-data">
        <summary>事件数据</summary>
        <pre>{JSON.stringify(payload, null, 2)}</pre>
      </details>
    </>
  );
}

export default function TraceView({
  sessionId,
  runs,
  selectedId,
  rounds,
  connection,
  onSelect,
}: {
  sessionId: string | null;
  runs: Run[];
  selectedId: string | null;
  rounds: { run: Run; events: Trace[] }[];
  connection: string;
  onSelect: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(selectedId ? [selectedId] : []),
  );
  useEffect(() => {
    if (selectedId) setExpanded((value) => new Set([...value, selectedId]));
  }, [selectedId]);
  useEffect(() => {
    if (selectedId)
      document
        .getElementById(`trace-round-${selectedId}`)
        ?.scrollIntoView({ block: "nearest" });
  }, [selectedId, rounds.length]);
  const statuses: Record<string, string> = {
    answered: "已回答",
    incomplete: "执行未完成",
    insufficient: "资料不足",
    awaiting_user: "待补充",
    queued: "排队中",
    running: "查证中",
  };
  const modelCalls = runs.reduce(
    (total, run) => total + (run.model_calls ?? 0),
    0,
  );
  const toolCalls = runs.reduce(
    (total, run) => total + (run.tool_calls ?? 0),
    0,
  );

  return (
    <section className="trace-page" aria-label="查证过程">
      <div className="trace-toolbar">
        <div className="trace-session-heading">
          <strong>本会话 · {runs.length}轮提问</strong>
          <span>
            模型 {modelCalls} 次 · 工具 {toolCalls} 次
          </span>
        </div>
        {sessionId && (
          <span className="trace-sync" role="status">
            <span
              className={
                !runs.some((run) => run.execution_status !== "finished")
                  ? "sync-dot"
                  : "sync-dot syncing"
              }
              aria-hidden="true"
            />
            {connection || "加载中…"}
          </span>
        )}
      </div>
      <ul className="trace-legend" aria-label="过程颜色说明">
        <li className="tone-input">
          <span aria-hidden="true" />
          提问与问题理解
        </li>
        <li className="tone-tool">
          <span aria-hidden="true" />
          工具
        </li>
        <li className="tone-model">
          <span aria-hidden="true" />
          决策与结果
        </li>
        <li className="tone-error">
          <span aria-hidden="true" />
          异常
        </li>
      </ul>
      <div className="trace-scroll" tabIndex={0} aria-label="查证步骤">
        {rounds.map((round, index) => {
          const run =
            runs.find((item) => item.id === round.run.id) ?? round.run;
          const events = round.events;
          const capturedCalls = events.filter(
            (event) => event.type === "model.started",
          );
          const missingContext =
            (run.execution_status === "finished" &&
              capturedCalls.length < (run.model_calls ?? 0)) ||
            (capturedCalls.length > 0 &&
              Math.min(
                ...capturedCalls.map((event) =>
                  Number(event.payload.call_number),
                ),
              ) > 1);
          const isOpen = expanded.has(run.id);
          const systemEvents = events.filter((event) =>
            ["run.queued", "run.started"].includes(event.type),
          );
          const steps = events.filter(
            (event) => !["run.queued", "run.started"].includes(event.type),
          );
          return (
            <section
              key={run.id}
              id={`trace-round-${run.id}`}
              className="trace-round"
              aria-label={`第${index + 1}轮提问`}
            >
              <button
                className="trace-round-summary"
                aria-expanded={isOpen}
                aria-controls={`trace-body-${run.id}`}
                onClick={() => {
                  setExpanded((value) => {
                    const next = new Set(value);
                    if (next.has(run.id)) next.delete(run.id);
                    else next.add(run.id);
                    return next;
                  });
                  if (!isOpen) onSelect(run.id);
                }}
              >
                <span className="round-number">第{index + 1}轮</span>
                <span className="round-question" title={run.question}>
                  {run.question}
                </span>
                <span className="round-counts">
                  模型 {run.model_calls ?? 0} · 工具 {run.tool_calls ?? 0}
                </span>
                <span className={`status ${run.task_status ?? ""}`}>
                  {statuses[run.task_status ?? run.execution_status] ??
                    run.execution_status}
                </span>
                <span aria-hidden="true">{isOpen ? "−" : "+"}</span>
              </button>
              {isOpen && (
                <div id={`trace-body-${run.id}`} className="trace-round-body">
                  {missingContext && (
                    <p className="legacy-context-note">
                      旧记录未保存完整模型输入，可查看已有决策及工具链。
                    </p>
                  )}
                  <ol className="trace-timeline">
                    <li className="trace-event tone-input">
                      <span className="event-node" aria-hidden="true" />
                      <article className="event-card">
                        <div className="event-heading">
                          <h2>用户提问</h2>
                        </div>
                        <p className="event-body">{run.question}</p>
                      </article>
                    </li>
                    {steps.map((event) => (
                      <li
                        key={event.seq}
                        className={`trace-event tone-${eventTone(event)}`}
                      >
                        <span className="event-node" aria-hidden="true" />
                        <article className="event-card">
                          <div className="event-heading">
                            <h2>{eventLabels[event.type] ?? event.type}</h2>
                            <time dateTime={event.occurred_at}>
                              {new Date(event.occurred_at).toLocaleTimeString(
                                "zh-CN",
                                {
                                  hour12: false,
                                },
                              )}
                            </time>
                          </div>
                          <EventContent
                            event={event}
                            question={run.question}
                            runId={run.id}
                          />
                        </article>
                      </li>
                    ))}
                    {!events.length && (
                      <li className="trace-pending">正在载入记录…</li>
                    )}
                  </ol>
                  {systemEvents.length > 0 && (
                    <details className="system-events event-details">
                      <summary>系统事件 · {systemEvents.length}</summary>
                      {systemEvents.map((event) => (
                        <div key={event.seq}>
                          <p>{eventLabels[event.type]}</p>
                          <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                        </div>
                      ))}
                    </details>
                  )}
                  <details className="system-events event-details">
                    <summary>会话与轮次标识</summary>
                    <pre>
                      {JSON.stringify(
                        {
                          session_id: sessionId,
                          run_id: run.id,
                          kb_version_id: run.kb_version_id,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </div>
              )}
            </section>
          );
        })}
        {!rounds.length && (
          <div className="trace-empty">
            {runs.length ? connection || "正在载入会话记录…" : "暂无查证记录"}
          </div>
        )}
      </div>
    </section>
  );
}
