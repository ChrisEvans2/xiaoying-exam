import React, { Suspense, lazy, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import TraceView from "./TraceView";
const PDFPreview = lazy(() => import("./PDFPreview"));

import type {
  KB,
  Version,
  ChunkConfig,
  Doc,
  Citation,
  Run,
  Conversation,
  ConversationSummary,
  Trace,
  SessionTrace,
  Job,
  Pending,
} from "./types";
const labels: Record<string, string> = {
  answered: "已回答",
  awaiting_user: "待用户确认",
  incomplete: "执行未完成",
  insufficient: "资料不足",
  queued: "排队中",
  running: "查证中",
  waiting: "待补充",
  finished: "已结束",
};
const short = (id: string | null | undefined) =>
  id ? id.slice(0, 8) : "未发布";
const historyDate = (value: string) =>
  new Date(value).toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

const documentDraftKey = "xiaoying:document-drafts";
function readDocumentDrafts(): Record<string, Pending[]> {
  try {
    const saved = JSON.parse(localStorage.getItem(documentDraftKey) ?? "{}");
    if (!saved || typeof saved !== "object" || Array.isArray(saved)) return {};
    return Object.fromEntries(
      Object.entries(saved).map(([id, items]) => [
        id,
        Array.isArray(items)
          ? items.filter(
              (item): item is Pending =>
                item &&
                typeof item.document_id === "string" &&
                typeof item.name === "string" &&
                ["add", "replace", "remove"].includes(item.action) &&
                (item.action === "remove" ||
                  typeof item.upload_id === "string"),
            )
          : [],
      ]),
    );
  } catch {
    return {};
  }
}

async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Idempotency-Key"] = crypto.randomUUID();
    if (!(body instanceof FormData))
      headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    method: method ?? (body === undefined ? "GET" : "POST"),
    headers,
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error?.message ?? "请求失败，请稍后重试");
  return value;
}

function App() {
  const [kbs, setKbs] = useState<KB[]>([]),
    [kbId, setKbId] = useState(""),
    [versions, setVersions] = useState<Version[]>([]),
    [docs, setDocs] = useState<Doc[]>([]);
  const [view, setView] = useState<"chat" | "trace" | "knowledge">("chat"),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [mode, setMode] = useState("");
  const [name, setName] = useState(""),
    [creating, setCreating] = useState(false),
    [documentDrafts, setDocumentDrafts] = useState(readDocumentDrafts),
    [knowledgeLoading, setKnowledgeLoading] = useState(false),
    [jobId, setJobId] = useState<string | null>(null),
    [job, setJob] = useState<Job | null>(null);
  const [config, setConfig] = useState<ChunkConfig>({
      target_tokens: 400,
      max_tokens: 800,
      overlap_tokens: 80,
      semantic_threshold: 0.62,
    }),
    [metadata, setMetadata] = useState("{}");
  const [conversationId, setConversationId] = useState<string | null>(null),
    [conversations, setConversations] = useState<ConversationSummary[]>([]),
    [conversationLoading, setConversationLoading] = useState(false),
    [historyLoading, setHistoryLoading] = useState(false),
    [historyError, setHistoryError] = useState(""),
    [runs, setRuns] = useState<Run[]>([]),
    [question, setQuestion] = useState(""),
    [busy, setBusy] = useState(false),
    [uploading, setUploading] = useState(false);
  const [traceId, setTraceId] = useState<string | null>(null),
    [sessionTrace, setSessionTrace] = useState<SessionTrace | null>(null),
    [connection, setConnection] = useState(""),
    [supplement, setSupplement] = useState(""),
    [supplementVersion, setSupplementVersion] = useState("");
  const [citation, setCitation] = useState<Citation | null>(null),
    [source, setSource] = useState<{
      name: string;
      file_url: string;
      blocks: {
        block_id: string;
        text: string;
        locator: Record<string, unknown>;
      }[];
    } | null>(null),
    [clock, setClock] = useState(Date.now());
  const fileInput = useRef<HTMLInputElement>(null),
    bottom = useRef<HTMLDivElement>(null);
  const selectedKb = useRef(kbId),
    jobKb = useRef(""),
    knowledgeRequest = useRef(0),
    conversationRequest = useRef(0),
    historyRequest = useRef(0),
    drafts = useRef(new Map<string, string>());
  selectedKb.current = kbId;
  const pending = documentDrafts[kbId] ?? [];
  const setPending = (
    update: Pending[] | ((items: Pending[]) => Pending[]),
    id = kbId,
  ) => {
    setDocumentDrafts((all) => ({
      ...all,
      [id]: typeof update === "function" ? update(all[id] ?? []) : update,
    }));
  };
  const current = kbs.find((k) => k.id === kbId);
  const active = runs.find((r) => r.execution_status !== "finished");
  const building =
    busy ||
    knowledgeLoading ||
    (!!jobId && !job) ||
    job?.status === "pending" ||
    job?.status === "running";
  const guard = async (fn: () => Promise<void>) => {
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const reloadKbs = async () => {
    const list = await api<KB[]>("/api/v1/knowledge-bases");
    setKbs(list);
    return list;
  };
  const reloadKnowledge = async (id = kbId, restoreConfig = true) => {
    if (!id) return;
    const request = ++knowledgeRequest.current;
    if (selectedKb.current === id) setKnowledgeLoading(true);
    try {
      const [list, all, builds] = await Promise.all([
        api<Version[]>(`/api/v1/knowledge-bases/${id}/versions`),
        api<KB[]>("/api/v1/knowledge-bases"),
        api<Job[]>(`/api/v1/knowledge-bases/${id}/builds`),
      ]);
      const selected = all.find((k) => k.id === id);
      const documents = selected?.current_version_id
        ? await api<Doc[]>(
            `/api/v1/knowledge-bases/${id}/documents?version_id=${selected.current_version_id}`,
          )
        : [];
      if (selectedKb.current !== id || request !== knowledgeRequest.current)
        return;
      setKbs(all);
      setVersions(list);
      setDocs(documents);
      const latestBuild = builds[0] ?? null;
      jobKb.current = id;
      setJob(latestBuild);
      setJobId(latestBuild?.id ?? null);
      const version = list.find(
        (v) =>
          v.id ===
          (latestBuild?.status === "pending" ||
          latestBuild?.status === "running"
            ? latestBuild.resource_id
            : selected?.current_version_id),
      );
      if (restoreConfig && version) setConfig(version.config.chunking);
      setPending(
        (items) =>
          items.map((item) =>
            item.action === "remove"
              ? item
              : {
                  ...item,
                  action: documents.some(
                    (doc) => doc.document_id === item.document_id,
                  )
                    ? "replace"
                    : "add",
                },
          ),
        id,
      );
    } finally {
      if (selectedKb.current === id && request === knowledgeRequest.current)
        setKnowledgeLoading(false);
    }
  };
  const reloadConversations = async (id = kbId) => {
    const request = ++historyRequest.current;
    try {
      const list = await api<ConversationSummary[]>(
        `/api/v1/knowledge-bases/${id}/conversations`,
      );
      if (selectedKb.current === id && request === historyRequest.current) {
        setConversations(list);
        setHistoryError("");
      }
      return list;
    } catch (e) {
      if (selectedKb.current === id && request === historyRequest.current)
        setHistoryError((e as Error).message);
      throw e;
    }
  };
  const openConversation = async (cid: string, id = kbId, showChat = true) => {
    const request = ++conversationRequest.current;
    setConversationLoading(true);
    if (showChat) setView("chat");
    setError("");
    try {
      const value = await api<Conversation>(`/api/v1/conversations/${cid}`);
      if (selectedKb.current !== id || request !== conversationRequest.current)
        return;
      if (value.kb_id !== id) throw new Error("会话不属于当前知识库");
      setConversationId(cid);
      setRuns(value.runs);
      setTraceId(value.runs.at(-1)?.id ?? null);
      setCitation(null);
      setSupplement("");
      setSupplementVersion("");
      setQuestion(drafts.current.get(`${id}:${cid}`) ?? "");
      localStorage.setItem(`xiaoying:conversation:${id}`, cid);
      if (showChat) setView("chat");
    } catch (e) {
      if (selectedKb.current === id && request === conversationRequest.current)
        setError((e as Error).message);
    } finally {
      if (selectedKb.current === id && request === conversationRequest.current)
        setConversationLoading(false);
    }
  };
  useEffect(() => {
    localStorage.setItem(documentDraftKey, JSON.stringify(documentDrafts));
  }, [documentDrafts]);
  useEffect(() => {
    void guard(async () => {
      const list = await reloadKbs();
      const saved = localStorage.getItem("xiaoying:kb");
      setKbId(list.find((k) => k.id === saved)?.id ?? list[0]?.id ?? "");
      setMode((await api<{ provider_mode: string }>("/health")).provider_mode);
    });
  }, []);
  useEffect(() => {
    let alive = true;
    ++conversationRequest.current;
    ++historyRequest.current;
    setVersions([]);
    setDocs([]);
    setError("");
    setNotice("");
    jobKb.current = "";
    setJobId(null);
    setJob(null);
    setRuns([]);
    setTraceId(null);
    setCitation(null);
    setSupplement("");
    setSupplementVersion("");
    setConversationId(null);
    setConversations([]);
    setHistoryError("");
    setConversationLoading(false);
    setHistoryLoading(!!kbId);
    setQuestion(drafts.current.get(`${kbId}:new`) ?? "");
    if (kbId) {
      localStorage.setItem("xiaoying:kb", kbId);
      void guard(() => reloadKnowledge(kbId));
      void (async () => {
        try {
          const list = await reloadConversations(kbId);
          if (!alive) return;
          const saved = localStorage.getItem(`xiaoying:conversation:${kbId}`);
          const cid =
            saved === ""
              ? null
              : (list.find((item) => item.id === saved)?.id ?? list[0]?.id);
          if (cid) await openConversation(cid, kbId, false);
        } catch {
          // The history panel provides the error and a retry without blocking knowledge access.
        } finally {
          if (alive) setHistoryLoading(false);
        }
      })();
    }
    return () => {
      alive = false;
    };
  }, [kbId]);
  useEffect(() => {
    if (view === "knowledge" && kbId)
      void guard(() => reloadKnowledge(kbId, false));
  }, [view]);
  useEffect(() => {
    if (!jobId || jobKb.current !== kbId) return;
    let alive = true,
      done = false,
      polling = false;
    const id = kbId;
    const poll = async () => {
      if (polling || done) return;
      polling = true;
      try {
        const value = await api<Job>(`/api/v1/jobs/${jobId}`);
        if (!alive || selectedKb.current !== id || jobKb.current !== id) return;
        setJob(value);
        if (value.status === "done" || value.status === "failed") {
          done = true;
          if (value.status === "done") {
            const submitted = value.payload?.changes ?? [];
            setPending(
              (items) =>
                items.filter(
                  (item) =>
                    !submitted.some(
                      (change) =>
                        change.document_id === item.document_id &&
                        (change.upload_id ?? null) === (item.upload_id ?? null),
                    ),
                ),
              id,
            );
            await reloadKnowledge(id);
            if (alive && selectedKb.current === id)
              setNotice(
                value.progress?.published
                  ? "新版本已发布，新问题将使用新资料。"
                  : "构建已就绪，请查看发布状态。",
              );
          } else setError(value.error?.message ?? "构建失败");
        }
      } catch (e) {
        if (alive) setError((e as Error).message);
      } finally {
        polling = false;
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 1500);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [jobId, kbId]);
  // Run lifecycle stays current even while the user inspects an older trace.
  useEffect(() => {
    if (!active) return;
    let alive = true,
      polling = false;
    const runId = active.id;
    const poll = async () => {
      if (polling) return;
      polling = true;
      try {
        const run = await api<Run>(`/api/v1/runs/${runId}`);
        if (alive)
          setRuns((items) =>
            items.map((item) => (item.id === run.id ? run : item)),
          );
      } catch (e) {
        if (alive) setError((e as Error).message);
      } finally {
        polling = false;
      }
    };
    const timer = setInterval(() => void poll(), 1000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [active?.id, kbId]);
  useEffect(() => {
    setSessionTrace(null);
    setConnection("");
    if (!conversationId || view !== "trace") return;
    let alive = true;
    let stream: EventSource | null = null;
    setConnection("加载会话记录…");
    void (async () => {
      try {
        const snapshot = await api<SessionTrace>(
          `/api/v1/conversations/${conversationId}/trace`,
        );
        if (!alive) return;
        setSessionTrace(snapshot);
        const running = snapshot.rounds.find(
          (item) => item.run.execution_status !== "finished",
        );
        if (!running) {
          setConnection("记录已完整保存");
          return;
        }
        const runId = running.run.id;
        const after = running.events.at(-1)?.seq ?? 0;
        stream = new EventSource(
          `/api/v1/runs/${runId}/events?after_seq=${after}`,
        );
        stream.onopen = () => {
          if (alive) setConnection("实时同步");
        };
        stream.onmessage = (e) => {
          if (!alive) return;
          const row = JSON.parse(e.data) as Trace;
          setSessionTrace(
            (value) =>
              value && {
                ...value,
                rounds: value.rounds.map((item) =>
                  item.run.id !== runId ||
                  item.events.some((event) => event.seq === row.seq)
                    ? item
                    : { ...item, events: [...item.events, row] },
                ),
              },
          );
          if (
            row.type === "run.finished" ||
            row.type === "clarification.requested"
          ) {
            void api<Run>(`/api/v1/runs/${runId}`)
              .then((run) => {
                if (alive)
                  setRuns((items) =>
                    items.map((item) => (item.id === run.id ? run : item)),
                  );
              })
              .catch((e) => {
                if (alive) setError(e.message);
              });
          }
          if (row.type === "run.finished") {
            stream?.close();
            setConnection("记录已完整保存");
            void reloadConversations().catch(() => {});
          }
        };
        stream.onerror = () => {
          if (alive) setConnection("正在恢复连接");
        };
      } catch (e) {
        if (alive) {
          setConnection("会话记录加载失败");
          setError((e as Error).message);
        }
      }
    })();
    return () => {
      alive = false;
      stream?.close();
    };
  }, [conversationId, view, active?.id, runs.length]);
  useEffect(() => {
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (view === "chat") {
      const region = bottom.current?.parentElement;
      region?.scrollTo({ top: region.scrollHeight, behavior: "smooth" });
    }
  }, [runs.length, conversationId, view]);
  useEffect(() => {
    let alive = true;
    setSource(null);
    if (citation)
      void guard(async () => {
        const value = await api<{
          name: string;
          file_url: string;
          blocks: {
            block_id: string;
            text: string;
            locator: Record<string, unknown>;
          }[];
        }>(
          `/api/v1/knowledge-bases/${kbId}/versions/${citation.kb_version_id}/documents/${citation.document_version_id}/source`,
        );
        if (alive) setSource(value);
      });
    return () => {
      alive = false;
    };
  }, [citation, kbId]);
  const createKB = () =>
    guard(async () => {
      setCreating(true);
      try {
        const k = await api<KB>("/api/v1/knowledge-bases", { name });
        await reloadKbs();
        setKbId(k.id);
        setName("");
        setView("knowledge");
      } finally {
        setCreating(false);
      }
    });
  const upload = async (files: FileList | null) => {
    if (!files || !kbId || uploading || building) return;
    const uploadKb = kbId;
    setUploading(true);
    await guard(async () => {
      const supplied = JSON.parse(metadata);
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        form.append("metadata", JSON.stringify(supplied));
        const item = await api<{
          upload_id: string;
          metadata: { document_id: string };
        }>(`/api/v1/knowledge-bases/${uploadKb}/uploads`, form);
        setPending(
          (list) => [
            ...list.filter((x) => x.document_id !== item.metadata.document_id),
            {
              upload_id: item.upload_id,
              document_id: item.metadata.document_id,
              name: file.name,
              action: docs.some(
                (d) => d.document_id === item.metadata.document_id,
              )
                ? "replace"
                : "add",
            },
          ],
          uploadKb,
        );
      }
    });
    setUploading(false);
    if (fileInput.current) fileInput.current.value = "";
  };
  const build = () =>
    guard(async () => {
      if (!kbId || building || uploading) return;
      setBusy(true);
      setNotice("");
      try {
        const result = await api<{ job_id: string }>(
          `/api/v1/knowledge-bases/${kbId}/builds`,
          {
            base_version_id: current?.current_version_id ?? null,
            changes: pending.map(({ name, ...p }) => p),
            config,
            auto_publish: true,
          },
        );
        if (selectedKb.current !== kbId) return;
        jobKb.current = kbId;
        setJobId(result.job_id);
        setJob(null);
      } catch (e) {
        if (selectedKb.current === kbId)
          await reloadKnowledge(kbId).catch(() => {});
        throw e;
      } finally {
        setBusy(false);
      }
    });
  const send = () =>
    guard(async () => {
      if (
        !question.trim() ||
        active ||
        busy ||
        conversationLoading ||
        historyLoading
      )
        return;
      setBusy(true);
      try {
        let cid = conversationId;
        if (!cid) {
          cid = (
            await api<{ id: string }>("/api/v1/conversations", { kb_id: kbId })
          ).id;
          if (selectedKb.current === kbId) setConversationId(cid);
          localStorage.setItem(`xiaoying:conversation:${kbId}`, cid);
        }
        const value = await api<{ run_id: string }>(
          `/api/v1/conversations/${cid}/runs`,
          { question },
        );
        const run = await api<Run>(`/api/v1/runs/${value.run_id}`);
        if (selectedKb.current !== kbId) return;
        setRuns((list) => [...list, run]);
        setTraceId(run.id);
        setQuestion("");
        drafts.current.delete(`${kbId}:${cid}`);
        if (!conversationId) drafts.current.delete(`${kbId}:new`);
        await reloadConversations().catch(() => {});
      } finally {
        setBusy(false);
      }
    });
  const clarify = (action: string) =>
    guard(async () => {
      if (!active || conversationLoading || historyLoading) return;
      setBusy(true);
      try {
        await api(`/api/v1/runs/${active.id}/clarification`, {
          action,
          content: supplement,
          ...(supplementVersion ? { version_id: supplementVersion } : {}),
        });
        const run = await api<Run>(`/api/v1/runs/${active.id}`);
        if (selectedKb.current !== kbId) return;
        setRuns((items) => items.map((r) => (r.id === run.id ? run : r)));
        setSupplement("");
      } finally {
        setBusy(false);
      }
    });
  const newConversation = () => {
    if (active || busy || historyLoading) return;
    ++conversationRequest.current;
    setConversationLoading(false);
    setError("");
    setConversationId(null);
    setRuns([]);
    setTraceId(null);
    setView("chat");
    setCitation(null);
    setSupplement("");
    setSupplementVersion("");
    setQuestion(drafts.current.get(`${kbId}:new`) ?? "");
    localStorage.setItem(`xiaoying:conversation:${kbId}`, "");
  };
  const remaining = active?.deadline_at
    ? Math.max(0, Math.ceil((Date.parse(active.deadline_at) - clock) / 1000))
    : 0;

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="/" aria-label="青屿知识问答首页">
          <span className="brand-mark">青</span>
          <span>
            青屿 <i>/</i> 知识问答
          </span>
        </a>
        {mode === "mock" && <span className="mode">离线演示</span>}
      </header>
      <nav className="workspace-tabs" aria-label="问答页面">
        <button
          className={view === "chat" ? "page-tab selected" : "page-tab"}
          aria-current={view === "chat" ? "page" : undefined}
          onClick={() => setView("chat")}
        >
          咨询对话
        </button>
        <button
          className={view === "trace" ? "page-tab selected" : "page-tab"}
          aria-current={view === "trace" ? "page" : undefined}
          onClick={() => setView("trace")}
        >
          查证过程
        </button>
      </nav>
      <div className="workspace">
        <aside className="sidebar">
          <div className="section-label">知识库</div>
          <label className="sr-only" htmlFor="knowledge-select">
            选择知识库
          </label>
          <select
            id="knowledge-select"
            value={kbId}
            disabled={busy || uploading || creating}
            onChange={(e) => setKbId(e.target.value)}
          >
            <option value="" disabled>
              选择知识库
            </option>
            {kbs.map((k) => (
              <option key={k.id} value={k.id}>
                {k.name}
              </option>
            ))}
          </select>
          <button
            className="secondary new-chat"
            disabled={!kbId || !!active || busy || historyLoading}
            onClick={newConversation}
          >
            ＋ 新会话
          </button>
          <nav aria-label="知识库管理">
            <button
              className={
                view === "knowledge" ? "nav-item selected" : "nav-item"
              }
              onClick={() => setView("knowledge")}
            >
              文档与版本 <small>{docs.length}</small>
            </button>
          </nav>
          {kbId && (
            <section
              className="history-panel"
              aria-labelledby="history-heading"
            >
              <div className="section-label" id="history-heading">
                历史会话
              </div>
              {historyLoading ? (
                <p className="history-status" role="status">
                  加载中…
                </p>
              ) : historyError ? (
                <div className="history-status" role="alert">
                  历史加载失败
                  <button
                    className="text-button"
                    onClick={() => void reloadConversations().catch(() => {})}
                  >
                    重试
                  </button>
                </div>
              ) : conversations.length ? (
                <>
                  <select
                    className="history-select"
                    aria-label="历史会话"
                    value={conversationId ?? ""}
                    disabled={busy}
                    onChange={(e) => {
                      if (e.target.value) void openConversation(e.target.value);
                    }}
                  >
                    <option value="" disabled>
                      选择历史会话
                    </option>
                    {conversations.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.title} · {item.input_count}轮 ·{" "}
                        {historyDate(item.last_activity_at)}
                      </option>
                    ))}
                  </select>
                  <nav className="history-list" aria-label="历史会话">
                    {conversations.map((item) => (
                      <button
                        key={item.id}
                        className={
                          item.id === conversationId
                            ? "history-item selected"
                            : "history-item"
                        }
                        aria-current={
                          item.id === conversationId ? "true" : undefined
                        }
                        title={item.title}
                        disabled={busy}
                        onClick={() => void openConversation(item.id)}
                      >
                        <span>{item.title}</span>
                        <time dateTime={item.last_activity_at}>
                          {item.input_count}轮 ·{" "}
                          {historyDate(item.last_activity_at)}
                        </time>
                      </button>
                    ))}
                  </nav>
                </>
              ) : (
                <p className="history-status">暂无历史会话</p>
              )}
            </section>
          )}
          <div className="sidebar-bottom">
            <details>
              <summary>新建知识库</summary>
              <label htmlFor="kb-name">知识库名称</label>
              <input
                id="kb-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="例如：产品手册"
              />
              <button onClick={createKB} disabled={!name.trim() || creating}>
                创建
              </button>
            </details>
          </div>
        </aside>
        <main className={`main main-${view}`}>
          {view === "knowledge" && (
            <div className="page-header">
              <h1>文档与版本</h1>
            </div>
          )}
          {error && (
            <div className="alert" role="alert">
              <span>{error}</span>
              <button className="text-button" onClick={() => setError("")}>
                关闭
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <span>{notice}</span>
              <button className="text-button" onClick={() => setNotice("")}>
                关闭
              </button>
            </div>
          )}
          {view === "chat" ? (
            <>
              <div className="chat-session-header">
                <span>{conversationId ? "当前会话" : "新会话"}</span>
                <span>{runs.length}轮提问</span>
              </div>
              <div className="conversation" aria-label="咨询对话" tabIndex={0}>
                {(conversationLoading || historyLoading) && (
                  <p className="conversation-loading" role="status">
                    加载对话…
                  </p>
                )}
                {!runs.length && !conversationLoading && !historyLoading && (
                  <div className="empty-state">
                    <h1>有什么需要查证？</h1>
                  </div>
                )}
                {!conversationLoading &&
                  !historyLoading &&
                  runs.map((run) => (
                    <article className="turn" key={run.id}>
                      <div className="user-message">
                        <small>你</small>
                        <p>{run.question}</p>
                      </div>
                      <div className="answer">
                        <div className="answer-header">
                          <span className="small-mark">青</span>
                          <strong>知识助手</strong>
                          <span className={`status ${run.task_status ?? ""}`}>
                            {labels[run.task_status ?? run.execution_status]}
                          </span>
                          <button
                            className="text-button"
                            onClick={() => {
                              setTraceId(run.id);
                              setView("trace");
                            }}
                          >
                            查看过程
                          </button>
                        </div>
                        {run.result ? (
                          <>
                            <p className="answer-text">{run.result.answer}</p>
                            {run.result.citations.length > 0 && (
                              <div className="citations">
                                {run.result.citations.map((c, i) => (
                                  <button
                                    key={c.evidence_id + ":" + i}
                                    className="citation"
                                    onClick={() => setCitation(c)}
                                  >
                                    <span>
                                      {String(i + 1).padStart(2, "0")}
                                    </span>
                                    <div>
                                      <strong>{c.document_id}</strong>
                                      <small>{c.version_date}</small>
                                    </div>
                                    <span>↗</span>
                                  </button>
                                ))}
                              </div>
                            )}
                            {run.result.information_gaps.length > 0 && (
                              <div className="gaps">
                                <strong>还缺少的信息</strong>
                                <ul>
                                  {run.result.information_gaps.map((gap, i) => (
                                    <li key={i}>{gap}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </>
                        ) : (
                          <div className="loading-line">
                            <span className="pulse" />
                            正在核对资料与原文…
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                <div ref={bottom} />
              </div>
              {!conversationLoading &&
              !historyLoading &&
              active?.execution_status === "waiting" ? (
                <section className="clarification">
                  <div className="row">
                    <h3>补充信息</h3>
                    <span className="timer">
                      剩余 {Math.floor(remaining / 60)}:
                      {String(remaining % 60).padStart(2, "0")}
                    </span>
                  </div>
                  <ul>
                    {active.result?.questions?.map((q) => (
                      <li key={q}>{q}</li>
                    ))}
                  </ul>
                  <label className="sr-only" htmlFor="supplement">
                    补充信息
                  </label>
                  <textarea
                    id="supplement"
                    value={supplement}
                    onChange={(e) => setSupplement(e.target.value)}
                    placeholder="填写必要信息，或到文档页上传补充制度"
                  />
                  <label>
                    使用的资料版本
                    <select
                      value={supplementVersion}
                      onFocus={() => void guard(() => reloadKnowledge())}
                      onChange={(e) => setSupplementVersion(e.target.value)}
                    >
                      <option value="">继续使用原版本</option>
                      {versions
                        .filter(
                          (v) =>
                            v.published_at && v.id !== active.kb_version_id,
                        )
                        .map((v) => (
                          <option key={v.id} value={v.id}>
                            版本 {short(v.id)}
                          </option>
                        ))}
                    </select>
                  </label>
                  <div className="actions">
                    <button
                      className="secondary"
                      disabled={busy || !remaining}
                      onClick={() => clarify("abandon")}
                    >
                      放弃补充
                    </button>
                    <button
                      disabled={
                        busy ||
                        !remaining ||
                        (!supplement.trim() && !supplementVersion)
                      }
                      onClick={() => clarify("submit")}
                    >
                      提交补充
                    </button>
                  </div>
                </section>
              ) : (
                <form
                  className="composer"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void send();
                  }}
                >
                  <label className="composer-label" htmlFor="question">
                    咨询问题
                  </label>
                  <textarea
                    id="question"
                    value={question}
                    disabled={conversationLoading || historyLoading || busy}
                    onChange={(e) => {
                      setQuestion(e.target.value);
                      drafts.current.set(
                        `${kbId}:${conversationId ?? "new"}`,
                        e.target.value,
                      );
                    }}
                    onKeyDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                        e.preventDefault();
                        void send();
                      }
                    }}
                    placeholder={
                      current?.current_version_id
                        ? "输入问题…"
                        : "先到“文档与版本”上传并发布资料"
                    }
                  />
                  <div className="composer-footer">
                    <small>
                      <kbd>⌘ / Ctrl</kbd> <kbd>Enter</kbd> 发送
                    </small>
                    <button
                      disabled={
                        busy ||
                        conversationLoading ||
                        historyLoading ||
                        !!active ||
                        !question.trim() ||
                        !current?.current_version_id
                      }
                      type="submit"
                    >
                      {busy ? "提交中" : "发送"}{" "}
                      <span aria-hidden="true">↑</span>
                    </button>
                  </div>
                </form>
              )}
            </>
          ) : view === "trace" ? (
            <TraceView
              key={conversationId}
              sessionId={conversationId}
              runs={runs}
              selectedId={traceId}
              rounds={
                sessionTrace?.session_id === conversationId
                  ? sessionTrace.rounds
                  : []
              }
              connection={connection}
              onSelect={setTraceId}
            />
          ) : (
            <div className="knowledge-page">
              <div
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  void upload(e.dataTransfer.files);
                }}
              >
                <span className="upload-symbol">↥</span>
                <h2>
                  {uploading
                    ? "正在保存文件…"
                    : knowledgeLoading
                      ? "正在恢复构建状态…"
                      : building
                        ? "正在构建，请完成后再上传"
                        : "拖动文档到这里"}
                </h2>
                <p>Markdown、Word、HTML、PDF、TXT · 单文件不超过 20 MiB</p>
                <button
                  className="secondary"
                  disabled={!kbId || uploading || building}
                  onClick={() => fileInput.current?.click()}
                >
                  选择文件
                </button>
                <input
                  hidden
                  multiple
                  type="file"
                  disabled={uploading || building}
                  ref={fileInput}
                  accept=".md,.docx,.html,.pdf,.txt"
                  onChange={(e) => void upload(e.target.files)}
                />
              </div>
              <details className="settings">
                <summary>元数据与分块设置</summary>
                <div className="settings-grid">
                  {(
                    [
                      "target_tokens",
                      "max_tokens",
                      "overlap_tokens",
                      "semantic_threshold",
                    ] as const
                  ).map((k, i) => (
                    <label key={k}>
                      {
                        [
                          "目标 tokens",
                          "最大 tokens",
                          "重叠 tokens",
                          "语义相似度阈值",
                        ][i]
                      }
                      <input
                        type="number"
                        disabled={uploading || building}
                        step={k === "semantic_threshold" ? 0.01 : 1}
                        value={config[k]}
                        onChange={(e) =>
                          setConfig({ ...config, [k]: Number(e.target.value) })
                        }
                      />
                    </label>
                  ))}
                </div>
                <label>
                  上传文件缺少元数据时填写
                  JSON（document_id、title、version_date、scope）
                  <textarea
                    value={metadata}
                    disabled={uploading || building}
                    onChange={(e) => setMetadata(e.target.value)}
                    spellCheck={false}
                  />
                </label>
              </details>
              {pending.length > 0 && (
                <section className="pending">
                  <h3>待构建变更 · {pending.length}</h3>
                  {pending.map((p) => (
                    <div className="row" key={p.document_id}>
                      <span>
                        <b>{p.document_id}</b> · {p.name}
                      </span>
                      <span>
                        {p.action === "replace"
                          ? "替换"
                          : p.action === "remove"
                            ? "移除"
                            : "新增"}{" "}
                        <button
                          className="text-button"
                          disabled={building || uploading}
                          onClick={() =>
                            setPending((items) =>
                              items.filter(
                                (i) => i.document_id !== p.document_id,
                              ),
                            )
                          }
                        >
                          撤回
                        </button>
                      </span>
                    </div>
                  ))}
                </section>
              )}
              <div className="actions build-actions">
                <button
                  disabled={
                    !kbId ||
                    building ||
                    uploading ||
                    (!pending.length && !docs.length)
                  }
                  onClick={build}
                >
                  {knowledgeLoading
                    ? "正在恢复状态…"
                    : building
                      ? "后台构建中…"
                      : "构建并发布新版本"}
                </button>
              </div>
              {job && (
                <div
                  className={`job ${job.status === "failed" ? "failed" : ""}`}
                  role="status"
                >
                  <strong>
                    {job.status === "done"
                      ? "构建完成"
                      : job.status === "failed"
                        ? "构建失败"
                        : "正在处理文档"}
                  </strong>
                  <p>
                    {job.progress?.stage === "ready"
                      ? "已完成"
                      : job.progress?.stage === "indexing"
                        ? "已处理"
                        : "等待处理"}{" "}
                    {job.progress?.completed !== undefined
                      ? `${job.progress.completed} / ${job.progress.total}`
                      : ""}
                  </p>
                  {job.progress?.reused !== undefined && (
                    <p>
                      复用 {job.progress.reused} 份 · 新构建{" "}
                      {job.progress.rebuilt ?? 0} 份
                    </p>
                  )}
                  {(job.status === "pending" || job.status === "running") && (
                    <p>任务在后台继续，可离开此页面，返回后自动恢复进度。</p>
                  )}
                  {job.progress?.publish_error && (
                    <p>{job.progress.publish_error.message}</p>
                  )}
                  {job.error && <p>{job.error.message}</p>}
                </div>
              )}
              <section className="document-list">
                <div className="row">
                  <h2>
                    当前文档 <small>{docs.length}</small>
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => void guard(() => reloadKnowledge())}
                  >
                    刷新
                  </button>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>文档</th>
                      <th>版本日期</th>
                      <th>修订 ID</th>
                      <th>
                        <span className="sr-only">操作</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.map((d) => (
                      <tr key={d.document_id}>
                        <td>
                          <b>{d.document_id}</b>
                          <span>{d.metadata.title}</span>
                          <small>{d.name}</small>
                        </td>
                        <td>{d.version_date}</td>
                        <td>
                          <code>{short(d.document_version_id)}</code>
                        </td>
                        <td>
                          <button
                            className="text-button"
                            disabled={building || uploading}
                            onClick={() =>
                              setPending((items) => [
                                ...items.filter(
                                  (i) => i.document_id !== d.document_id,
                                ),
                                {
                                  document_id: d.document_id,
                                  name: d.name,
                                  action: "remove",
                                },
                              ])
                            }
                          >
                            移除
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!docs.length && <p className="muted">还没有已发布的文档。</p>}
              </section>
              <details className="versions">
                <summary>版本记录 · {versions.length}</summary>
                {versions.map((v) => (
                  <div className="version-row" key={v.id}>
                    <code>{short(v.id)}</code>
                    <span>
                      {v.id === current?.current_version_id
                        ? "当前发布"
                        : v.published_at
                          ? "历史发布"
                          : v.status === "ready"
                            ? "就绪"
                            : v.status === "failed"
                              ? "失败"
                              : "构建中"}
                    </span>
                    <small>
                      {new Date(v.created_at).toLocaleString("zh-CN")}
                    </small>
                    {v.status === "ready" && !v.published_at && (
                      <button
                        className="text-button"
                        disabled={building || uploading}
                        onClick={() =>
                          void guard(async () => {
                            await api(
                              `/api/v1/knowledge-bases/${kbId}/versions/${v.id}/publish`,
                              {
                                expected_current_version_id:
                                  current?.current_version_id,
                              },
                            );
                            await reloadKnowledge();
                          })
                        }
                      >
                        发布
                      </button>
                    )}
                    {v.error && <small>{v.error.message}</small>}
                  </div>
                ))}
              </details>
            </div>
          )}
        </main>
      </div>
      {citation && (
        <div className="modal-backdrop" onClick={() => setCitation(null)}>
          <section
            className="source-modal"
            role="dialog"
            aria-modal="true"
            aria-label="引用原文"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="row">
              <div>
                <h2>
                  {citation.document_id} · {source?.name ?? "正在载入"}
                </h2>
              </div>
              <button className="secondary" onClick={() => setCitation(null)}>
                关闭
              </button>
            </div>
            <p className="muted">
              文档日期 {citation.version_date} · 知识库版本{" "}
              {short(citation.kb_version_id)}
            </p>
            <blockquote>{citation.quote}</blockquote>
            {source?.name.endsWith(".pdf") && (
              <Suspense fallback={<p>正在加载 PDF…</p>}>
                <PDFPreview url={source.file_url} citation={citation} />
              </Suspense>
            )}
            <details open>
              <summary>定位与上下文</summary>
              {source?.blocks
                .filter((b) =>
                  citation.locator.some((l) => l.block_id === b.block_id),
                )
                .map((b) => (
                  <div key={b.block_id} className="source-block">
                    <small>{JSON.stringify(b.locator)}</small>
                    <p>{b.text}</p>
                  </div>
                ))}
            </details>
            {source && (
              <a href={source.file_url} target="_blank" rel="noreferrer">
                打开原始文件 ↗
              </a>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
