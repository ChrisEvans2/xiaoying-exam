import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import psycopg
from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from xiaoying.config import load_settings
from xiaoying.domain import AppError, BuildRequest
from xiaoying.providers.embeddings import providers
from xiaoying.services.knowledge import KnowledgeService
from xiaoying.services.runs import RunService
from xiaoying.storage.db import Database, require, valid_id


class NameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ConversationRequest(BaseModel):
    kb_id: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


class PublishRequest(BaseModel):
    expected_current_version_id: str | None = None


class ClarificationRequest(BaseModel):
    action: str
    content: str = ""
    version_id: str | None = None


def create_app(settings=None, db=None, embedder=None):
    settings = settings or load_settings()
    owned = db is None

    @asynccontextmanager
    async def lifespan(app):
        app.state.db = db or Database(settings.database_url)
        encoder = embedder or providers(settings)[0]
        app.state.knowledge = KnowledgeService(app.state.db, settings, encoder)
        app.state.runs = RunService(app.state.db, settings, encoder)
        yield
        if owned:
            app.state.db.close()

    app = FastAPI(title="青屿知识问答", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_error(request, exc):
        return await app_error(request, AppError("INVALID_REQUEST", "请求字段或参数格式不正确", 422))

    @app.exception_handler(psycopg.Error)
    async def db_error(request, exc):
        transient = isinstance(
            exc, (psycopg.OperationalError, psycopg.errors.LockNotAvailable, psycopg.errors.DeadlockDetected)
        )
        return await app_error(
            request,
            AppError(
                "DATABASE_BUSY" if transient else "DATABASE_CONFLICT",
                "数据库暂时忙，请使用原幂等键重试" if transient else "数据约束冲突，请刷新后重试",
                503 if transient else 409,
                transient,
            ),
        )

    @app.get("/health")
    def health(request: Request):
        request.app.state.db.one("SELECT 1 ok")
        return {
            "status": "ok",
            "provider_mode": settings.provider_mode,
            "model_configured": bool(settings.deepseek_api_key),
            "pdf_backend": settings.pdf_backend,
            "local_single_user": True,
        }

    @app.get("/api/v1/knowledge-bases")
    def knowledge_bases(request: Request):
        return request.app.state.db.all("SELECT * FROM knowledge_bases ORDER BY created_at")

    @app.post("/api/v1/knowledge-bases", status_code=201)
    def create_knowledge(body: NameRequest, request: Request, idempotency_key: str | None = Header(None)):
        return request.app.state.knowledge.create(body.name, idempotency_key)

    @app.get("/api/v1/knowledge-bases/{kb_id}/versions")
    def versions(kb_id: str, request: Request):
        require(request.app.state.db.one("SELECT id FROM knowledge_bases WHERE id=%s", (valid_id(kb_id),)))
        return request.app.state.db.all(
            "SELECT * FROM kb_versions WHERE kb_id=%s ORDER BY created_at DESC", (kb_id,)
        )

    @app.get("/api/v1/knowledge-bases/{kb_id}/documents")
    def documents(kb_id: str, version_id: str, request: Request):
        return request.app.state.knowledge.documents(kb_id, version_id)

    @app.post("/api/v1/knowledge-bases/{kb_id}/uploads", status_code=201)
    async def upload(
        kb_id: str,
        request: Request,
        file: UploadFile = File(...),
        metadata: str = Form("{}"),
        idempotency_key: str | None = Header(None),
    ):
        try:
            meta = json.loads(metadata)
        except ValueError:
            raise AppError("INVALID_METADATA", "元数据必须为 JSON 对象", 422) from None
        if not isinstance(meta, dict):
            raise AppError("INVALID_METADATA", "元数据必须为 JSON 对象", 422)
        data = bytearray()
        try:
            while part := await file.read(1024 * 1024):
                data.extend(part)
                if len(data) > settings.max_upload_bytes:
                    raise AppError("FILE_TOO_LARGE", "文件超过大小限制", 413)
        finally:
            await file.close()
        return await asyncio.to_thread(
            request.app.state.knowledge.upload,
            kb_id,
            file.filename or "upload",
            bytes(data),
            meta,
            idempotency_key,
        )

    @app.post("/api/v1/knowledge-bases/{kb_id}/builds", status_code=202)
    def build(kb_id: str, body: BuildRequest, request: Request, idempotency_key: str | None = Header(None)):
        return request.app.state.knowledge.create_build(kb_id, body, idempotency_key)

    @app.post("/api/v1/knowledge-bases/{kb_id}/versions/{version_id}/publish")
    def publish(
        kb_id: str,
        version_id: str,
        body: PublishRequest,
        request: Request,
        idempotency_key: str | None = Header(None),
    ):
        return request.app.state.knowledge.publish(
            kb_id, version_id, body.expected_current_version_id, idempotency_key
        )

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str, request: Request):
        row = require(request.app.state.db.one("SELECT * FROM jobs WHERE id=%s", (valid_id(job_id),)))
        return {k: v for k, v in row.items() if k not in {"lease_token", "worker_id"}}

    @app.post("/api/v1/conversations", status_code=201)
    def conversation(body: ConversationRequest, request: Request, idempotency_key: str | None = Header(None)):
        return request.app.state.runs.create_conversation(body.kb_id, idempotency_key)

    @app.get("/api/v1/conversations/{cid}")
    def history(cid: str, request: Request):
        return request.app.state.runs.conversation(cid)

    @app.post("/api/v1/conversations/{cid}/runs", status_code=202)
    def question(
        cid: str, body: QuestionRequest, request: Request, idempotency_key: str | None = Header(None)
    ):
        return request.app.state.runs.create_run(cid, body.question, idempotency_key)

    @app.get("/api/v1/runs/{rid}")
    def run(rid: str, request: Request):
        return request.app.state.runs.get(rid)

    @app.post("/api/v1/runs/{rid}/clarification", status_code=202)
    def clarification(
        rid: str, body: ClarificationRequest, request: Request, idempotency_key: str | None = Header(None)
    ):
        return request.app.state.runs.clarify(rid, body.model_dump(exclude_none=True), idempotency_key)

    @app.get("/api/v1/runs/{rid}/events")
    async def events(
        rid: str, request: Request, after_seq: int = Query(0, ge=0), last_event_id: str | None = Header(None)
    ):
        rid = valid_id(rid)
        run = await asyncio.to_thread(request.app.state.runs.get, rid)
        try:
            cursor = int(last_event_id) if last_event_id is not None else after_seq
        except ValueError:
            raise AppError("INVALID_EVENT_ID", "事件序号应为非负整数", 422) from None
        if cursor < 0 or cursor > run["next_event_seq"]:
            raise AppError("INVALID_EVENT_ID", "事件序号超出当前范围", 422)

        async def stream():
            nonlocal cursor
            last_heartbeat = asyncio.get_running_loop().time()
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(
                    request.app.state.db.all,
                    "SELECT * FROM trace_events WHERE run_id=%s AND seq>%s ORDER BY seq LIMIT 100",
                    (rid, cursor),
                )
                for row in rows:
                    cursor = row["seq"]
                    yield f"id: {cursor}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                state = await asyncio.to_thread(request.app.state.runs.get, rid)
                if state["execution_status"] == "finished" and cursor >= state["next_event_seq"]:
                    break
                now = asyncio.get_running_loop().time()
                if now - last_heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/knowledge-bases/{kb_id}/versions/{version_id}/documents/{revision}/source")
    def source(kb_id: str, version_id: str, revision: str, request: Request):
        result = request.app.state.knowledge.source(kb_id, version_id, revision)
        result.pop("storage_path", None)
        result["file_url"] = (
            f"/api/v1/knowledge-bases/{kb_id}/versions/{version_id}/documents/{revision}/file"
        )
        return result

    @app.get("/api/v1/knowledge-bases/{kb_id}/versions/{version_id}/documents/{revision}/file")
    def original_file(kb_id: str, version_id: str, revision: str, request: Request):
        result = request.app.state.knowledge.source(kb_id, version_id, revision)
        path = Path(result["storage_path"])
        if not path.is_file():
            raise AppError("SOURCE_FILE_MISSING", "原始文件暂不可用", 503)
        pdf = path.suffix.lower() == ".pdf"
        return FileResponse(
            path,
            filename=result["name"],
            media_type="application/pdf" if pdf else "application/octet-stream",
            content_disposition_type="inline" if pdf else "attachment",
        )

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
