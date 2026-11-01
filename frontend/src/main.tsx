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
  Trace,
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
    [pending, setPending] = useState<Pending[]>([]),
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
    [runs, setRuns] = useState<Run[]>([]),
    [question, setQuestion] = useState(""),
    [busy, setBusy] = useState(false),
    [uploading, setUploading] = useState(false);
  const [traceId, setTraceId] = useState<string | null>(null),
    [trace, setTrace] = useState<Trace[]>([]),
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
    knowledgeRequest = useRef(0);
  selectedKb.current = kbId;
  const current = kbs.find((k) => k.id === kbId);
  const active = runs.find((r) => r.execution_status !== "finished");
  const building =
    busy ||
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
  const reloadKnowledge = async (id = kbId) => {
    if (!id) return;
    const request = ++knowledgeRequest.current;
    const [list, all] = await Promise.all([
      api<Version[]>(`/api/v1/knowledge-bases/${id}/versions`),
      api<KB[]>("/api/v1/knowledge-bases"),
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
    const version = list.find((v) => v.id === selected?.current_version_id);
    if (version) setConfig(version.config.chunking);
  };
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
    setVersions([]);
    setDocs([]);
    setError("");
    setNotice("");
    setPending([]);
    setJobId(null);
    setJob(null);
    setRuns([]);
    setTraceId(null);
    setCitation(null);
    setSupplement("");
    setSupplementVersion("");
    setConversationId(null);
    if (kbId) {
      localStorage.setItem("xiaoying:kb", kbId);
      void guard(async () => {
        await reloadKnowledge(kbId);
        const cid = localStorage.getItem(`xiaoying:conversation:${kbId}`);
        if (cid) {
          const value = await api<{ runs: Run[] }>(
            `/api/v1/conversations/${cid}`,
          );
          if (alive) {
            if (selectedKb.current === kbId) setConversationId(cid);
            setRuns(value.runs);
            setTraceId(value.runs.at(-1)?.id ?? null);
          }
        }
      });
    }
    return () => {
      alive = false;
    };
  }, [kbId]);
  useEffect(() => {
    if (!jobId) return;
    let alive = true,
      done = false,
      polling = false;
    const id = kbId;
    const poll = async () => {
      if (polling || done) return;
      polling = true;
      try {
        const value = await api<Job>(`/api/v1/jobs/${jobId}`);
        if (!alive || selectedKb.current !== id) return;
        setJob(value);
        if (value.status === "done" || value.status === "failed") {
          done = true;
          if (value.status === "done") {
            setPending([]);
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
    setTrace([]);
    setConnection("");
    if (!traceId) return;
    const stream = new EventSource(`/api/v1/runs/${traceId}/events`);
    setConnection("连接中");
    let alive = true;
    const refresh = async () => {
      const run = await api<Run>(`/api/v1/runs/${traceId}`);
      if (alive)
        setRuns((list) => list.map((r) => (r.id === run.id ? run : r)));
    };
    stream.onopen = () => {
      if (alive) setConnection("实时同步");
    };
    stream.onmessage = (e) => {
      if (!alive) return;
      const row = JSON.parse(e.data) as Trace;
      setTrace((items) =>
        items.some((i) => i.seq === row.seq) ? items : [...items, row],
      );
      void refresh().catch((e) => {
        if (alive) setError(e.message);
      });
      if (row.type === "run.finished") {
        stream.close();
        setConnection("记录已完整保存");
      }
    };
    stream.onerror = () => {
      if (alive) setConnection("正在恢复连接");
    };
    void refresh().catch((e) => {
      if (alive) setError(e.message);
    });
    return () => {
      alive = false;
      stream.close();
    };
  }, [traceId]);
  useEffect(() => {
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (view === "chat") {
      const region = bottom.current?.parentElement;
      region?.scrollTo({ top: region.scrollHeight, behavior: "smooth" });
    }
  }, [runs.length, view]);
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
        if (selectedKb.current !== uploadKb) break;
        setPending((list) => [
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
        ]);
      }
    });
    setUploading(false);
    if (fileInput.current) fileInput.current.value = "";
  };
  const build = () =>
    guard(async () => {
      setBusy(true);
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
        setJobId(result.job_id);
        setJob(null);
      } finally {
        setBusy(false);
      }
    });
  const send = () =>
    guard(async () => {
      if (!question.trim() || active || busy) return;
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
      } finally {
        setBusy(false);
      }
    });
  const clarify = (action: string) =>
    guard(async () => {
      if (!active) return;
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
    if (active || busy) return;
    setConversationId(null);
    setRuns([]);
    setTraceId(null);
    setView("chat");
    localStorage.removeItem(`xiaoying:conversation:${kbId}`);
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
            disabled={!kbId || !!active || busy}
            onClick={newConversation}
          >
            ＋ 新对话
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
              <div className="conversation" aria-label="咨询对话" tabIndex={0}>
                {!runs.length && (
                  <div className="empty-state">
                    <h1>有什么需要查证？</h1>
                  </div>
                )}
                {runs.map((run) => (
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
                                  <span>{String(i + 1).padStart(2, "0")}</span>
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
              {active?.execution_status === "waiting" ? (
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
                    onChange={(e) => setQuestion(e.target.value)}
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
              runs={runs}
              selectedId={traceId}
              events={trace}
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
                  构建并发布新版本
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
                    {job.progress?.stage}{" "}
                    {job.progress?.completed !== undefined
                      ? `${job.progress.completed} / ${job.progress.total}`
                      : ""}
                  </p>
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
