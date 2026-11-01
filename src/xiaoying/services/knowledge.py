import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
from psycopg.types.json import Jsonb

from xiaoying.domain import AppError, BuildRequest, ChunkConfig, digest
from xiaoying.ingestion.parsers import inspect_metadata, parse_document, validate_file
from xiaoying.ingestion.chunking import chunk_document
from xiaoying.providers.bounded import bounded
from xiaoying.providers.embeddings import terms
from xiaoying.storage.db import fence, idempotent, require, uid, valid_id


class KnowledgeService:
    def __init__(self, db, settings, embedder):
        self.db, self.settings, self.embedder = db, settings, embedder

    def create(self, name, key):
        if not name.strip() or len(name) > 120:
            raise AppError("INVALID_NAME", "知识库名称长度为 1—120", 422)
        with self.db.tx() as c:

            def op():
                return c.execute(
                    "INSERT INTO knowledge_bases(id,name) VALUES(%s,%s) RETURNING *", (uid(), name.strip())
                ).fetchone()

            return idempotent(c, "create-kb", key, {"name": name}, op)

    def upload(self, kb_id, name, data, metadata, key):
        kb_id = valid_id(kb_id)
        require(self.db.one("SELECT id FROM knowledge_bases WHERE id=%s", (kb_id,)))
        suffix = Path(name).suffix.lower()
        folder = self.settings.data_dir.resolve() / "objects"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, suffix=suffix, delete=False) as f:
            temp = Path(f.name)
            f.write(data)
        try:
            validate_file(temp, self.settings)
            meta = inspect_metadata(temp, metadata)
            sha = hashlib.sha256(data).hexdigest()
            path = folder / (sha + suffix)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        with self.db.tx() as c:

            def op():
                upload_id = uid()
                c.execute(
                    "INSERT INTO uploads(id,kb_id,name,file_hash,storage_path,metadata) VALUES(%s,%s,%s,%s,%s,%s)",
                    (upload_id, kb_id, Path(name).name, sha, str(path), Jsonb(meta)),
                )
                return {"upload_id": upload_id, "name": Path(name).name, "metadata": meta, "file_hash": sha}

            return idempotent(c, f"upload:{kb_id}", key, {"name": name, "hash": sha, "metadata": meta}, op)

    def create_build(self, kb_id, body: BuildRequest, key):
        kb_id = valid_id(kb_id)
        base = valid_id(body.base_version_id) if body.base_version_id else None
        with self.db.tx() as c:

            def op():
                kb = require(
                    c.execute("SELECT * FROM knowledge_bases WHERE id=%s FOR UPDATE", (kb_id,)).fetchone()
                )
                if str(kb["current_version_id"] or "") != str(base or ""):
                    raise AppError("VERSION_CONFLICT", "当前版本已变化，请刷新后重建", 409)
                active = c.execute(
                    """SELECT j.id FROM jobs j JOIN kb_versions v ON v.id=j.resource_id
                    WHERE j.type='build' AND v.kb_id=%s AND j.status IN ('pending','running') LIMIT 1""",
                    (kb_id,),
                ).fetchone()
                if active:
                    raise AppError("BUILD_IN_PROGRESS", "该知识库已有后台构建任务，请等待完成", 409)
                manifest = {}
                if base:
                    for row in c.execute(
                        "SELECT m.*,d.document_id FROM kb_version_documents m JOIN documents d ON d.id=m.document_fk WHERE kb_version_id=%s",
                        (base,),
                    ).fetchall():
                        manifest[row["document_id"]] = (
                            str(row["document_fk"]),
                            str(row["document_version_id"]),
                        )
                changes = []
                for change in body.changes:
                    exists = change.document_id in manifest
                    if change.action == "add" and exists:
                        raise AppError("DOCUMENT_EXISTS", "已有该文档 ID，请选择替换", 409)
                    if change.action in {"replace", "remove"} and not exists:
                        raise AppError("DOCUMENT_NOT_FOUND", "替换或移除的文档不在基线版本", 409)
                    if change.action == "remove":
                        del manifest[change.document_id]
                        changes.append({**change.model_dump(), "name": change.document_id})
                        continue
                    upload = require(
                        c.execute(
                            "SELECT * FROM uploads WHERE id=%s AND kb_id=%s",
                            (valid_id(change.upload_id), kb_id),
                        ).fetchone()
                    )
                    if upload["metadata"]["document_id"] != change.document_id:
                        raise AppError("DOCUMENT_ID_MISMATCH", "上传文件 ID 与变更清单不一致", 422)
                    changes.append({**change.model_dump(), "name": upload["name"]})
                    if exists:
                        original = c.execute(
                            """SELECT u.file_hash,u.storage_path,v.metadata FROM document_versions v
                            JOIN uploads u ON u.id=v.upload_id WHERE v.id=%s""",
                            (manifest[change.document_id][1],),
                        ).fetchone()
                        if (
                            original["file_hash"] == upload["file_hash"]
                            and original["metadata"] == upload["metadata"]
                            and Path(original["storage_path"]).suffix == Path(upload["storage_path"]).suffix
                        ):
                            continue
                    d = c.execute(
                        "SELECT id FROM documents WHERE kb_id=%s AND document_id=%s",
                        (kb_id, change.document_id),
                    ).fetchone()
                    doc_id = str(d["id"]) if d else uid()
                    if not d:
                        c.execute(
                            "INSERT INTO documents(id,kb_id,document_id) VALUES(%s,%s,%s)",
                            (doc_id, kb_id, change.document_id),
                        )
                    revision = uid()
                    c.execute(
                        "INSERT INTO document_versions(id,document_fk,kb_id,upload_id,metadata,version_date) VALUES(%s,%s,%s,%s,%s,%s)",
                        (
                            revision,
                            doc_id,
                            kb_id,
                            upload["id"],
                            Jsonb(upload["metadata"]),
                            upload["metadata"]["version_date"],
                        ),
                    )
                    manifest[change.document_id] = (doc_id, revision)
                if not manifest or len(manifest) > 100:
                    raise AppError("INVALID_MANIFEST", "知识库须包含 1—100 份文档", 422)
                version, job = uid(), uid()
                config = {
                    "chunking": body.config.model_dump(),
                    "embedding_identity": self.embedder.identity,
                    "provider_mode": self.settings.provider_mode,
                    "pdf_backend": self.settings.pdf_backend,
                    "parser_revision": "v1",
                    "tokenizer_revision": "jieba-0.42.1-policy-v1",
                }
                c.execute(
                    "INSERT INTO kb_versions(id,kb_id,base_version_id,status,config,manifest_hash) VALUES(%s,%s,%s,'pending',%s,%s)",
                    (version, kb_id, base, Jsonb(config), digest(manifest)),
                )
                for doc_id, revision in manifest.values():
                    c.execute(
                        "INSERT INTO kb_version_documents(kb_version_id,kb_id,document_fk,document_version_id) VALUES(%s,%s,%s,%s)",
                        (version, kb_id, doc_id, revision),
                    )
                c.execute(
                    "INSERT INTO jobs(id,type,resource_id,payload) VALUES(%s,'build',%s,%s)",
                    (
                        job,
                        version,
                        Jsonb(
                            {
                                "auto_publish": body.auto_publish,
                                "expected_current_version_id": base,
                                "build_timeout_seconds": self.settings.build_timeout_seconds,
                                "changes": changes,
                            }
                        ),
                    ),
                )
                return {"job_id": job, "version_id": version}

            return idempotent(c, f"build:{kb_id}", key, body.model_dump(), op)

    def builds(self, kb_id):
        kb_id = valid_id(kb_id)
        require(self.db.one("SELECT id FROM knowledge_bases WHERE id=%s", (kb_id,)))
        return self.db.all(
            """SELECT j.id,j.resource_id,j.status,j.progress,j.error,j.payload,j.created_at
            FROM jobs j JOIN kb_versions v ON v.id=j.resource_id
            WHERE j.type='build' AND v.kb_id=%s
            ORDER BY (j.status IN ('pending','running')) DESC,j.created_at DESC,j.id DESC LIMIT 20""",
            (kb_id,),
        )

    @staticmethod
    def _prepare_document(c, job, version, revision):
        fence(c, job["id"], job["lease_token"])
        state = c.execute("SELECT status FROM kb_versions WHERE id=%s", (version,)).fetchone()
        if state["status"] != "building":
            raise AppError("VERSION_IMMUTABLE", "版本已不可变", 409)
        c.execute("DELETE FROM chunks WHERE kb_version_id=%s AND document_version_id=%s", (version, revision))
        c.execute("DELETE FROM source_blocks WHERE kb_version_id=%s AND document_version_id=%s", (version, revision))

    def _reuse_document(self, job, version, base, revision, progress):
        with self.db.tx() as c:
            self._prepare_document(c, job, version, revision)
            ordinals = [row["ordinal"] for row in c.execute(
                "SELECT ordinal FROM chunks WHERE kb_version_id=%s AND document_version_id=%s ORDER BY ordinal",
                (base, revision),
            ).fetchall()]
            c.execute(
                """INSERT INTO source_blocks(kb_version_id,document_version_id,block_id,text,locator,block_type,ordinal)
                SELECT %s,document_version_id,block_id,text,locator,block_type,ordinal FROM source_blocks
                WHERE kb_version_id=%s AND document_version_id=%s""",
                (version, base, revision),
            )
            ids = [str(uuid5(UUID(version), f"{revision}:{ordinal}")) for ordinal in ordinals]
            c.execute(
                """INSERT INTO chunks(id,kb_version_id,document_version_id,ordinal,text,source_ranges,
                heading_path,search_vector,embedding,token_count)
                SELECT ids.id,%s,c.document_version_id,c.ordinal,c.text,c.source_ranges,
                c.heading_path,c.search_vector,c.embedding,c.token_count FROM chunks c
                JOIN unnest(%s::uuid[],%s::int[]) AS ids(id,ordinal) ON ids.ordinal=c.ordinal
                WHERE c.kb_version_id=%s AND c.document_version_id=%s""",
                (version, ids, ordinals, base, revision),
            )
            c.execute("UPDATE jobs SET progress=%s WHERE id=%s", (Jsonb(progress), job["id"]))

    @staticmethod
    def _publish(c, kb_id, version, expected):
        kb = require(c.execute("SELECT * FROM knowledge_bases WHERE id=%s FOR UPDATE", (kb_id,)).fetchone())
        v = require(
            c.execute("SELECT * FROM kb_versions WHERE id=%s AND kb_id=%s", (version, kb_id)).fetchone()
        )
        if str(kb["current_version_id"] or "") == str(version):
            return {"version_id": str(version), "published": True}
        if v["status"] != "ready":
            raise AppError("VERSION_NOT_READY", "只能发布完整就绪版本", 409)
        if str(kb["current_version_id"] or "") != str(expected or ""):
            raise AppError("VERSION_CONFLICT", "基线已变化，请基于新版本重新构建", 409)
        # A stale ready snapshot cannot override another build by supplying its newer current ID.
        if str(v["base_version_id"] or "") != str(expected or ""):
            raise AppError("VERSION_CONFLICT", "目标版本并非基于指定当前版本构建", 409)
        c.execute("UPDATE knowledge_bases SET current_version_id=%s WHERE id=%s", (version, kb_id))
        c.execute(
            "UPDATE kb_versions SET published_at=COALESCE(published_at,clock_timestamp()) WHERE id=%s",
            (version,),
        )
        return {"version_id": str(version), "published": True}

    def publish(self, kb_id, version, expected, key):
        kb_id, version = valid_id(kb_id), valid_id(version)
        if expected:
            expected = valid_id(expected)
        with self.db.tx() as c:
            return idempotent(
                c,
                f"publish:{kb_id}",
                key,
                {"version_id": version, "expected": expected},
                lambda: self._publish(c, kb_id, version, expected),
            )

    def documents(self, kb_id, version):
        require(
            self.db.one(
                "SELECT id FROM kb_versions WHERE id=%s AND kb_id=%s", (valid_id(version), valid_id(kb_id))
            )
        )
        return self.db.all(
            """SELECT d.document_id,v.id AS document_version_id,v.metadata,v.version_date,u.name
            FROM kb_version_documents m JOIN document_versions v ON v.id=m.document_version_id
            JOIN documents d ON d.id=m.document_fk JOIN uploads u ON u.id=v.upload_id
            WHERE m.kb_version_id=%s ORDER BY d.document_id""",
            (version,),
        )

    def build(self, job):
        version = job["resource_id"]
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            timing = c.execute(
                """UPDATE jobs SET payload=CASE WHEN payload ? 'first_started_at' THEN payload
                ELSE payload || jsonb_build_object('first_started_at',clock_timestamp()) END
                WHERE id=%s RETURNING EXTRACT(EPOCH FROM(clock_timestamp()-(payload->>'first_started_at')::timestamptz)) elapsed,
                COALESCE((payload->>'build_timeout_seconds')::float,%s) budget""",
                (job["id"], self.settings.build_timeout_seconds),
            ).fetchone()
        start = time.monotonic()
        initial_remaining = float(timing["budget"]) - float(timing["elapsed"])
        record = require(self.db.one("SELECT * FROM kb_versions WHERE id=%s", (version,)))
        if (
            record["config"]["embedding_identity"] != self.embedder.identity
            or record["config"]["pdf_backend"] != self.settings.pdf_backend
        ):
            raise AppError("CONFIG_MISMATCH", "Worker 模型或解析配置与构建快照不同", 409)

        def remaining():
            left = initial_remaining - (time.monotonic() - start)
            if left <= 0:
                raise AppError("BUILD_TIMEOUT", "构建时间预算耗尽", 504)
            return left

        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            if record["status"] != "ready":
                c.execute("UPDATE kb_versions SET status='building',error=NULL WHERE id=%s", (version,))
        docs = self.db.all(
            """SELECT v.*,u.storage_path FROM kb_version_documents m
            JOIN document_versions v ON v.id=m.document_version_id JOIN uploads u ON u.id=v.upload_id
            WHERE m.kb_version_id=%s ORDER BY m.document_fk""",
            (version,),
        )
        base = self.db.one(
            "SELECT id,config,status FROM kb_versions WHERE id=%s AND kb_id=%s",
            (record["base_version_id"], record["kb_id"]),
        ) if record["base_version_id"] else None
        reusable = set()
        if base and base["status"] == "ready" and base["config"] == record["config"]:
            reusable = {row["document_version_id"] for row in self.db.all(
                """SELECT m.document_version_id FROM kb_version_documents m WHERE m.kb_version_id=%s
                AND EXISTS (SELECT 1 FROM chunks c WHERE c.kb_version_id=m.kb_version_id
                            AND c.document_version_id=m.document_version_id)
                AND EXISTS (SELECT 1 FROM source_blocks s WHERE s.kb_version_id=m.kb_version_id
                            AND s.document_version_id=m.document_version_id)""",
                (base["id"],),
            )}
        progress = (job.get("progress") or {}) if record["status"] == "ready" else {}
        reused, rebuilt = progress.get("reused", 0), progress.get("rebuilt", 0)
        if record["status"] != "ready":
            for index, doc in enumerate(docs):
                remaining()
                progress = {"stage": "indexing", "completed": index + 1, "total": len(docs),
                            "reused": reused, "rebuilt": rebuilt}
                if doc["id"] in reusable:
                    reused += 1
                    progress["reused"] = reused
                    self._reuse_document(job, version, base["id"], doc["id"], progress)
                    continue
                artifacts = self.settings.data_dir / "parsed" / version / doc["id"] / str(job["lease_token"])
                ir = bounded(
                    lambda doc=doc, artifacts=artifacts: parse_document(
                        Path(doc["storage_path"]), doc["metadata"], self.settings, artifacts
                    ),
                    remaining(),
                )
                chunks = bounded(
                    lambda ir=ir: chunk_document(
                        ir, self.embedder, ChunkConfig(**record["config"]["chunking"])
                    ),
                    remaining(),
                )
                if not chunks:
                    raise AppError("EMPTY_CHUNKS", "文档没有可用分块", 422)
                vectors = bounded(
                    lambda chunks=chunks: self.embedder.encode([c.text for c in chunks]), remaining()
                )
                matrix = np.asarray(vectors)
                if matrix.shape != (len(chunks), 1024) or not np.isfinite(matrix).all():
                    raise AppError("EMBEDDING_INVALID", "向量数量、维度或数值错误", 422)
                remaining()
                rebuilt += 1
                progress.update(rebuilt=rebuilt, warnings=ir.warnings)
                with self.db.tx() as c:
                    # Only unpublished building data is replaceable; a ready snapshot is never rewritten.
                    self._prepare_document(c, job, version, doc["id"])
                    for b in ir.blocks:
                        c.execute(
                            "INSERT INTO source_blocks(kb_version_id,document_version_id,block_id,text,locator,block_type,ordinal) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                            (version, doc["id"], b.block_id, b.text, Jsonb(b.locator), b.type, b.order),
                        )
                    for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                        chunk_id = str(uuid5(UUID(version), f"{doc['id']}:{ordinal}"))
                        c.execute(
                            """INSERT INTO chunks(id,kb_version_id,document_version_id,ordinal,text,source_ranges,heading_path,search_vector,embedding,token_count)
                            VALUES(%s,%s,%s,%s,%s,%s,%s,to_tsvector('simple',%s),%s::vector,%s)""",
                            (
                                chunk_id,
                                version,
                                doc["id"],
                                ordinal,
                                chunk.text,
                                Jsonb(chunk.source_ranges),
                                Jsonb(chunk.heading_path),
                                " ".join(terms(doc["metadata"]["title"] + " " + chunk.text)),
                                json.dumps(vector),
                                chunk.token_count,
                            ),
                        )
                    c.execute(
                        "UPDATE jobs SET progress=%s WHERE id=%s",
                        (
                            Jsonb(progress),
                            job["id"],
                        ),
                    )
            remaining()
            with self.db.tx() as c:
                fence(c, job["id"], job["lease_token"])
                count = c.execute(
                    "SELECT count(DISTINCT document_version_id) n FROM chunks WHERE kb_version_id=%s",
                    (version,),
                ).fetchone()["n"]
                if count != len(docs):
                    raise AppError("INCOMPLETE_BUILD", "部分文档缺失索引", 422)
                c.execute("UPDATE kb_versions SET status='ready',error=NULL WHERE id=%s", (version,))
        published = False
        conflict = None
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            if job["payload"].get("auto_publish"):
                try:
                    with c.transaction():
                        self._publish(
                            c, record["kb_id"], version, job["payload"].get("expected_current_version_id")
                        )
                        published = True
                except AppError as e:
                    if e.code != "VERSION_CONFLICT":
                        raise
                    conflict = {"code": e.code, "message": e.message}
            c.execute(
                "UPDATE jobs SET status='done',lease_until=NULL,progress=%s WHERE id=%s",
                (Jsonb({"stage": "ready", "completed": len(docs), "total": len(docs),
                        "reused": reused, "rebuilt": rebuilt,
                        "published": published, "publish_error": conflict}), job["id"]),
            )

    def source(self, kb_id, version, revision):
        row = require(
            self.db.one(
                """SELECT v.metadata,u.storage_path,u.name FROM kb_version_documents m
            JOIN document_versions v ON v.id=m.document_version_id JOIN uploads u ON u.id=v.upload_id
            JOIN kb_versions k ON k.id=m.kb_version_id
            WHERE m.kb_id=%s AND m.kb_version_id=%s AND m.document_version_id=%s AND k.status='ready'""",
                (valid_id(kb_id), valid_id(version), valid_id(revision)),
            )
        )
        row["blocks"] = self.db.all(
            "SELECT block_id,text,locator FROM source_blocks WHERE kb_version_id=%s AND document_version_id=%s ORDER BY ordinal",
            (version, revision),
        )
        return row
