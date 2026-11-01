import type { Run, Trace } from "./types";

const toolLabels: Record<string, string> = {
  search_knowledge: "检索知识库",
  read_document: "读取原文",
  finish_answer: "提交答复",
  request_clarification: "请求补充",
};

const eventLabels: Record<string, string> = {
  "run.queued": "问题已提交",
  "run.started": "开始查证",
  "query.expanded": "问题理解",
  "agent.decision": "模型决策",
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
    (event.type === "run.finished" &&
      event.payload.task_status === "incomplete")
  )
    return "error";
  if (["query.expanded", "clarification.submitted"].includes(event.type))
    return "input";
  if (event.type.startsWith("tool.")) return "tool";
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

function EventContent({ event, question }: { event: Trace; question: string }) {
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
      {tool && <p className="event-tool-name">{tool}</p>}
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
  runs,
  selectedId,
  events,
  connection,
  onSelect,
}: {
  runs: Run[];
  selectedId: string | null;
  events: Trace[];
  connection: string;
  onSelect: (id: string) => void;
}) {
  const selected = runs.find((run) => run.id === selectedId);
  const systemEvents = events.filter((event) =>
    ["run.queued", "run.started"].includes(event.type),
  );
  const steps = events.filter(
    (event) => !["run.queued", "run.started"].includes(event.type),
  );

  return (
    <section className="trace-page" aria-label="查证过程">
      <div className="trace-toolbar">
        <div className="trace-selector">
          <label htmlFor="trace-question-select">选择问题</label>
          <select
            id="trace-question-select"
            value={selectedId ?? ""}
            disabled={!runs.length}
            onChange={(e) => onSelect(e.target.value)}
          >
            {!selected && <option value="">暂无问题</option>}
            {runs.map((run, i) => (
              <option key={run.id} value={run.id}>
                {i + 1}. {run.question}
              </option>
            ))}
          </select>
        </div>
        {selected && (
          <span className="trace-sync" role="status">
            <span
              className={
                selected.execution_status === "finished"
                  ? "sync-dot"
                  : "sync-dot syncing"
              }
              aria-hidden="true"
            />
            {selected.execution_status === "finished"
              ? "已结束"
              : connection || "同步中"}
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
        {selected ? (
          <ol className="trace-timeline">
            <li className="trace-event tone-input">
              <span className="event-node" aria-hidden="true" />
              <article className="event-card">
                <div className="event-heading">
                  <h2>用户提问</h2>
                </div>
                <p className="event-body">{selected.question}</p>
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
                      {new Date(event.occurred_at).toLocaleTimeString("zh-CN", {
                        hour12: false,
                      })}
                    </time>
                  </div>
                  <EventContent event={event} question={selected.question} />
                </article>
              </li>
            ))}
            {!events.length && <li className="trace-pending">正在载入记录…</li>}
          </ol>
        ) : (
          <div className="trace-empty">暂无查证记录</div>
        )}
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
      </div>
    </section>
  );
}
