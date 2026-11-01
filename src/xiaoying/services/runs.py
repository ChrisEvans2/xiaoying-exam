from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from xiaoying.domain import AppError, LeaseLost, digest
from xiaoying.providers.embeddings import require_compatible_index
from xiaoying.storage.db import event, fence, idempotent, jsonable, require, uid, valid_id


def final_result(status, reason, answer, citations=None, gaps=None):
    return {
        "task_status": status,
        "reason_code": reason,
        "answer": answer,
        "citations": citations or [],
        "information_gaps": gaps or [],
    }


def finalize(conn, run_id, result):
    row = require(conn.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE", (run_id,)).fetchone())
    if row["execution_status"] == "finished":
        return row["result"]
    conn.execute(
        """UPDATE runs SET execution_status='finished',task_status=%s,result=%s,
        active_ms=active_ms+CASE WHEN segment_started_at IS NULL THEN 0 ELSE
        EXTRACT(EPOCH FROM (clock_timestamp()-segment_started_at))*1000 END,segment_started_at=NULL
        WHERE id=%s""",
        (result["task_status"], Jsonb(result), run_id),
    )
    event(conn, run_id, "run.finished", "run.finished", result)
    conn.execute(
        "INSERT INTO messages(conversation_id,run_id,role,content) VALUES(%s,%s,'assistant',%s)",
        (row["conversation_id"], run_id, Jsonb(result)),
    )
    return result


class RunService:
    def __init__(self, db, settings, embedder):
        self.db, self.settings, self.embedder = db, settings, embedder

    def check_index(self, conn, kb_id, version):
        record = require(
            conn.execute(
                "SELECT config FROM kb_versions WHERE id=%s AND kb_id=%s AND status='ready'",
                (version, kb_id),
            ).fetchone()
        )
        require_compatible_index(record["config"], self.embedder.identity)

    def create_conversation(self, kb_id, key):
        kb_id = valid_id(kb_id)
        with self.db.tx() as c:
            require(c.execute("SELECT id FROM knowledge_bases WHERE id=%s", (kb_id,)).fetchone())
            return idempotent(
                c,
                "conversation",
                key,
                {"kb_id": kb_id},
                lambda: c.execute(
                    "INSERT INTO conversations(id,kb_id) VALUES(%s,%s) RETURNING *", (uid(), kb_id)
                ).fetchone(),
            )

    def conversation(self, cid):
        row = require(self.db.one("SELECT * FROM conversations WHERE id=%s", (valid_id(cid),)))
        row["messages"] = self.db.all(
            "SELECT id,run_id,role,content,created_at FROM messages WHERE conversation_id=%s ORDER BY id",
            (cid,),
        )
        row["runs"] = self.db.all(
            "SELECT id,question,execution_status,task_status,result,deadline_at,kb_version_id,created_at FROM runs WHERE conversation_id=%s ORDER BY created_at",
            (cid,),
        )
        return row

    def create_run(self, cid, question, key):
        cid = valid_id(cid)
        if not question.strip() or len(question) > 8000:
            raise AppError("INVALID_QUESTION", "问题长度须为 1—8000", 422)
        with self.db.tx() as c:

            def op():
                conv = require(
                    c.execute("SELECT * FROM conversations WHERE id=%s FOR UPDATE", (cid,)).fetchone()
                )
                if c.execute(
                    "SELECT id FROM runs WHERE conversation_id=%s AND execution_status IN ('queued','running','waiting')",
                    (cid,),
                ).fetchone():
                    raise AppError("ACTIVE_RUN", "请先完成当前问题或放弃补充", 409)
                kb = require(
                    c.execute("SELECT * FROM knowledge_bases WHERE id=%s", (conv["kb_id"],)).fetchone()
                )
                if not kb["current_version_id"]:
                    raise AppError("KNOWLEDGE_NOT_READY", "请先构建并发布知识库", 409)
                self.check_index(c, conv["kb_id"], kb["current_version_id"])
                rid, jid = uid(), uid()
                c.execute(
                    "INSERT INTO runs(id,conversation_id,kb_id,kb_version_id,question,config) VALUES(%s,%s,%s,%s,%s,%s)",
                    (
                        rid,
                        cid,
                        conv["kb_id"],
                        kb["current_version_id"],
                        question.strip(),
                        Jsonb(self.settings.run_config()),
                    ),
                )
                c.execute(
                    "INSERT INTO messages(conversation_id,run_id,role,content) VALUES(%s,%s,'user',%s)",
                    (cid, rid, Jsonb({"text": question.strip()})),
                )
                c.execute("INSERT INTO jobs(id,type,resource_id) VALUES(%s,'run',%s)", (jid, rid))
                event(
                    c,
                    rid,
                    "run.queued",
                    "run.queued",
                    {
                        "kb_version_id": str(kb["current_version_id"]),
                        "provider_mode": self.settings.provider_mode,
                    },
                )
                return {"run_id": rid, "job_id": jid, "events_url": f"/api/v1/runs/{rid}/events"}

            return idempotent(c, f"run:{cid}", key, {"question": question}, op)

    def get(self, rid):
        return require(self.db.one("SELECT * FROM runs WHERE id=%s", (valid_id(rid),)))

    def start(self, job):
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            run = require(
                c.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE", (job["resource_id"],)).fetchone()
            )
            if run["execution_status"] == "finished":
                c.execute("UPDATE jobs SET status='done',lease_until=NULL WHERE id=%s", (job["id"],))
                return None
            self.check_index(c, run["kb_id"], run["kb_version_id"])
            now = c.execute("SELECT clock_timestamp() t").fetchone()["t"]
            # A dead worker's active segment still consumes wall-clock execution budget.
            if run["segment_started_at"]:
                c.execute(
                    "UPDATE runs SET active_ms=active_ms+EXTRACT(EPOCH FROM (%s-segment_started_at))*1000 WHERE id=%s",
                    (now, run["id"]),
                )
            c.execute(
                "UPDATE runs SET execution_status='running',task_status=NULL,segment_started_at=%s WHERE id=%s",
                (now, run["id"]),
            )
            event(
                c,
                run["id"],
                f"run.started:{job['id']}",
                "run.started",
                {"kb_version_id": str(run["kb_version_id"]), "evidence_epoch": run["evidence_epoch"]},
            )
        return self.get(job["resource_id"])

    def remaining(self, rid):
        with self.db.pool.connection() as c:
            r = require(
                c.execute("SELECT *,clock_timestamp() AS db_now FROM runs WHERE id=%s", (rid,)).fetchone()
            )
        elapsed = r["active_ms"] / 1000
        if r["segment_started_at"]:
            elapsed += (r["db_now"] - r["segment_started_at"]).total_seconds()
        return r["config"]["active_timeout_seconds"] - elapsed

    def reserve_action(self, job, action_id, kind, body):
        """Return cached output if done, else atomically reserve budget before external work."""
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            run = require(
                c.execute(
                    "SELECT *,clock_timestamp() db_now FROM runs WHERE id=%s FOR UPDATE",
                    (job["resource_id"],),
                ).fetchone()
            )
            if run["execution_status"] != "running":
                raise LeaseLost()
            old = c.execute(
                "SELECT * FROM run_actions WHERE run_id=%s AND action_id=%s", (run["id"], action_id)
            ).fetchone()
            if old and old["input_hash"] != digest(body):
                raise AppError("ACTION_CONFLICT", "恢复动作输入不一致", 409)
            if old and old["status"] == "done":
                return True, old["output"]
            elapsed = run["active_ms"] / 1000 + (run["db_now"] - run["segment_started_at"]).total_seconds()
            cfg = run["config"]
            column = {"model": "model_calls", "tool": "tool_calls", "step": "graph_steps"}[kind]
            limit = {"model": "max_model_calls", "tool": "max_tool_calls", "step": "max_graph_steps"}[kind]
            if elapsed >= cfg["active_timeout_seconds"] or run[column] >= cfg[limit]:
                raise AppError("EXECUTION_LIMIT", f"执行时间或 {column} 上限已达到", 409)
            # Column is chosen from a fixed internal mapping, never user supplied.
            c.execute(f"UPDATE runs SET {column}={column}+1 WHERE id=%s", (run["id"],))
            if old:
                event(
                    c,
                    run["id"],
                    f"unknown:{action_id}:{job['lease_token']}",
                    "action.unknown",
                    {"action_id": action_id, "message": "上次调用结果未确认，重试仍计入预算"},
                )
                c.execute(
                    "UPDATE run_actions SET status='reserved' WHERE run_id=%s AND action_id=%s",
                    (run["id"], action_id),
                )
            else:
                c.execute(
                    "INSERT INTO run_actions(run_id,action_id,kind,input_hash,status) VALUES(%s,%s,%s,%s,'reserved')",
                    (run["id"], action_id, kind, digest(body)),
                )
            return False, None

    def complete_action(self, job, action_id, output):
        # Only a successfully returned tool observation becomes citable evidence.
        # Timed-out background reads are pure reads and cannot commit evidence later.
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            run = require(
                c.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE", (job["resource_id"],)).fetchone()
            )
            action = require(
                c.execute(
                    "SELECT * FROM run_actions WHERE run_id=%s AND action_id=%s",
                    (job["resource_id"], action_id),
                ).fetchone()
            )
            if action["status"] == "done":
                return
            if action["kind"] == "tool" and not output.get("error"):
                for ev in output.get("evidence", []):
                    if ev["evidence_epoch"] != run["evidence_epoch"] or ev["kb_version_id"] != str(
                        run["kb_version_id"]
                    ):
                        raise AppError("EVIDENCE_EPOCH_CHANGED", "证据版本已变化", 409)
                    c.execute(
                        "INSERT INTO evidence_reads(id,run_id,evidence_epoch,kb_version_id,document_version_id,chunk_id,payload) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING",
                        (
                            ev["evidence_id"],
                            run["id"],
                            run["evidence_epoch"],
                            ev["kb_version_id"],
                            ev["document_version_id"],
                            ev["chunk_id"],
                            Jsonb(ev),
                        ),
                    )
            c.execute(
                "UPDATE run_actions SET status='done',output=%s WHERE run_id=%s AND action_id=%s",
                (Jsonb(jsonable(output)), job["resource_id"], action_id),
            )

    def trace(self, job, event_id, kind, payload):
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            event(c, job["resource_id"], event_id, kind, payload)

    def read(self, job, action_id, args):
        run = self.get(job["resource_id"])
        chunk_ids = args.get("chunk_ids") or []
        doc_id = args.get("document_id")
        revision = args.get("document_version_id")
        offset = args.get("offset", 0)
        limit = args.get("limit", 8)
        if not isinstance(offset, int) or not isinstance(limit, int) or offset < 0 or not 1 <= limit <= 8:
            raise AppError("INVALID_RANGE", "读取范围不合法", 422)
        if (
            not isinstance(chunk_ids, list)
            or len(chunk_ids) > 8
            or (not chunk_ids and not doc_id and not revision)
        ):
            raise AppError("INVALID_READ", "提供 chunk_ids 或文档 ID；每次最多 8 块", 422)
        chunk_ids = [valid_id(x) for x in chunk_ids]
        if revision:
            revision = valid_id(revision)
        rows = self.db.all(
            """SELECT c.*,d.document_id,v.version_date,v.metadata FROM chunks c
            JOIN document_versions v ON v.id=c.document_version_id JOIN documents d ON d.id=v.document_fk
            WHERE c.kb_version_id=%s AND d.kb_id=%s
            AND (cardinality(%s::uuid[])=0 OR c.id=ANY(%s::uuid[]))
            AND (%s::text IS NULL OR d.document_id=%s) AND (%s::uuid IS NULL OR v.id=%s)
            ORDER BY d.document_id,c.ordinal LIMIT %s OFFSET %s""",
            (
                run["kb_version_id"],
                run["kb_id"],
                chunk_ids,
                chunk_ids,
                doc_id,
                doc_id,
                revision,
                revision,
                limit + 1,
                offset,
            ),
        )
        if not rows:
            raise AppError("SOURCE_NOT_FOUND", "在本问题的知识库快照中未找到原文", 404)
        evidence = []
        total = 0
        for row in rows[:limit]:
            count = self.embedder.count_tokens(row["text"])
            if total + count > 6000:
                break
            total += count
            locations = []
            for part in row["source_ranges"]:
                block = require(
                    self.db.one(
                        "SELECT text,locator FROM source_blocks WHERE kb_version_id=%s AND document_version_id=%s AND block_id=%s",
                        (run["kb_version_id"], row["document_version_id"], part["block_id"]),
                    )
                )
                locations.append(
                    {
                        **block["locator"],
                        "block_id": part["block_id"],
                        "start": part["start"],
                        "end": part["end"],
                    }
                )
            payload = {
                "evidence_id": str(
                    uuid5(UUID(run["id"]), f"{run['evidence_epoch']}:{action_id}:{row['id']}")
                ),
                "evidence_epoch": run["evidence_epoch"],
                "kb_version_id": run["kb_version_id"],
                "document_id": row["document_id"],
                "document_version_id": row["document_version_id"],
                "version_date": row["version_date"],
                "chunk_id": row["id"],
                "locator": locations,
                "quote": row["text"],
                "scope": row["metadata"].get("scope"),
                "title": row["metadata"].get("title"),
            }
            evidence.append(payload)
        if not evidence:
            raise AppError("SOURCE_TOO_LARGE", "首个原文块超过单次读取预算", 422)
        return {
            "evidence": evidence,
            "next_offset": offset + len(evidence) if len(rows) > len(evidence) else None,
            "truncated": len(rows) > len(evidence),
        }

    def validate_result(self, rid, result, allow_wait=False):
        run = self.get(rid)
        if not isinstance(result, dict):
            raise AppError("INVALID_RESULT", "回答须为结构化对象", 422)
        status = result.get("task_status")
        if status not in {"answered", "insufficient", "incomplete"} | (
            {"awaiting_user"} if allow_wait else set()
        ):
            raise AppError("INVALID_STATUS", "无效的业务状态", 422)
        answer = result.get("answer")
        gaps = result.get("information_gaps", [])
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(gaps, list)
            or any(not isinstance(x, str) for x in gaps)
        ):
            raise AppError("INVALID_RESULT", "回答及资料缺口格式错误", 422)
        citations = result.get("citations", [])
        if not isinstance(citations, list) or len(citations) > 30:
            raise AppError("INVALID_CITATIONS", "引用格式错误或过多", 422)
        checked = []
        for cite in citations:
            if (
                not isinstance(cite, dict)
                or not isinstance(cite.get("quote"), str)
                or not cite["quote"].strip()
            ):
                raise AppError("INVALID_CITATION", "引用必须包含 evidence_id 和原文片段", 422)
            evidence = self.db.one(
                "SELECT payload FROM evidence_reads WHERE id=%s AND run_id=%s AND evidence_epoch=%s AND kb_version_id=%s",
                (valid_id(cite.get("evidence_id")), rid, run["evidence_epoch"], run["kb_version_id"]),
            )
            if not evidence or cite["quote"] not in evidence["payload"]["quote"]:
                raise AppError("UNREAD_CITATION", "引用未实际读取、版本不符或原文被改写", 422)
            checked.append({**evidence["payload"], "quote": cite["quote"]})
        if status == "answered" and not checked:
            raise AppError("EVIDENCE_REQUIRED", "制度结论必须有实际读取的原文引用", 422)
        if status == "insufficient" and run["clarification_count"] == 0:
            raise AppError("CLARIFICATION_REQUIRED", "请先提供唯一一次补充机会", 422)
        if status in {"insufficient", "awaiting_user"} and not gaps:
            raise AppError("GAPS_REQUIRED", "请说明具体资料缺口或冲突", 422)
        return final_result(
            status, result.get("reason_code", "MODEL_DECISION"), answer.strip(), checked, gaps
        )

    def prepare_clarification(self, job, action_id, args):
        questions = args.get("questions")
        if (
            not isinstance(questions, list)
            or not 1 <= len(questions) <= 6
            or any(not isinstance(x, str) or not x.strip() for x in questions)
        ):
            raise AppError("INVALID_CLARIFICATION", "集中提供 1—6 个必要问题", 422)
        result = self.validate_result(job["resource_id"], {**args, "task_status": "awaiting_user"}, True)
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            run = c.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE", (job["resource_id"],)).fetchone()
            old = c.execute("SELECT * FROM clarifications WHERE run_id=%s", (run["id"],)).fetchone()
            if old:
                if old["payload"].get("action_id") == action_id and old["status"] == "pending":
                    return jsonable(old["payload"])
                raise AppError("CLARIFICATION_EXHAUSTED", "已经用过补充机会；仍不足请结束为资料不足", 422)
            deadline = c.execute(
                "SELECT clock_timestamp()+(%s * interval '1 second') AS t",
                (run["config"]["clarification_seconds"],),
            ).fetchone()["t"]
            payload = {
                **result,
                "questions": questions,
                "deadline_at": deadline.isoformat(),
                "clarification_id": str(run["id"]),
                "action_id": action_id,
            }
            c.execute(
                "INSERT INTO clarifications(run_id,payload,deadline_at,status) VALUES(%s,%s,%s,'pending')",
                (run["id"], Jsonb(payload), deadline),
            )
            c.execute(
                "UPDATE runs SET clarification_count=1,deadline_at=%s WHERE id=%s", (deadline, run["id"])
            )
            return payload

    def waiting(self, job, payload):
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            c.execute(
                """UPDATE runs SET execution_status='waiting',task_status='awaiting_user',result=%s,
                active_ms=active_ms+EXTRACT(EPOCH FROM(clock_timestamp()-segment_started_at))*1000,
                segment_started_at=NULL WHERE id=%s AND execution_status='running'""",
                (Jsonb(payload), job["resource_id"]),
            )
            event(c, job["resource_id"], "clarification.requested", "clarification.requested", payload)
            c.execute("UPDATE jobs SET status='done',lease_until=NULL WHERE id=%s", (job["id"],))

    def finish(self, job, result):
        with self.db.tx() as c:
            fence(c, job["id"], job["lease_token"])
            finalize(c, job["resource_id"], result)
            c.execute("UPDATE jobs SET status='done',lease_until=NULL WHERE id=%s", (job["id"],))

    def clarify(self, rid, body, key):
        rid = valid_id(rid)
        action = body.get("action")
        content = body.get("content", "")
        if action not in {"submit", "abandon"} or not isinstance(content, str) or len(content) > 10000:
            raise AppError("INVALID_CLARIFICATION", "补充内容格式错误", 422)
        if action == "submit" and not content.strip() and not body.get("version_id"):
            raise AppError("EMPTY_CLARIFICATION", "请填写补充内容或选择新版本", 422)
        with self.db.tx() as c:

            def op():
                run = require(c.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE", (rid,)).fetchone())
                clar = c.execute("SELECT * FROM clarifications WHERE run_id=%s FOR UPDATE", (rid,)).fetchone()
                if run["execution_status"] != "waiting" or not clar or clar["status"] != "pending":
                    raise AppError("NOT_WAITING", "当前问题不再接受补充", 409)
                now = c.execute("SELECT clock_timestamp() t").fetchone()["t"]
                if now >= clar["deadline_at"]:
                    self.expire_one(c, run, clar)
                    return {"expired": True, "run_id": rid}
                if action == "abandon":
                    c.execute(
                        "UPDATE clarifications SET status='abandoned',response=%s WHERE run_id=%s",
                        (Jsonb(body), rid),
                    )
                    old = clar["payload"]
                    result = final_result(
                        "insufficient",
                        "USER_ABANDONED",
                        "已放弃补充，现有依据仍不足以确定答案。",
                        old.get("citations"),
                        old.get("information_gaps"),
                    )
                    finalize(c, rid, result)
                    return {"run_id": rid, "result": result}
                version = str(run["kb_version_id"])
                epoch = run["evidence_epoch"]
                if body.get("version_id"):
                    version = valid_id(body["version_id"])
                    require(
                        c.execute(
                            "SELECT * FROM kb_versions WHERE id=%s AND kb_id=%s AND status='ready' AND published_at IS NOT NULL",
                            (version, run["kb_id"]),
                        ).fetchone()
                    )
                    self.check_index(c, run["kb_id"], version)
                    if version != str(run["kb_version_id"]):
                        epoch += 1
                        event(
                            c,
                            rid,
                            "evidence.version_changed",
                            "evidence.version_changed",
                            {"from": str(run["kb_version_id"]), "to": version, "epoch": epoch},
                        )
                else:
                    self.check_index(c, run["kb_id"], version)
                response = {"content": content, "kb_version_id": version, "evidence_epoch": epoch}
                c.execute(
                    "UPDATE clarifications SET status='submitted',response=%s WHERE run_id=%s",
                    (Jsonb(response), rid),
                )
                c.execute(
                    "UPDATE runs SET execution_status='queued',task_status=NULL,result=NULL,kb_version_id=%s,evidence_epoch=%s WHERE id=%s",
                    (version, epoch, rid),
                )
                c.execute(
                    "INSERT INTO messages(conversation_id,run_id,role,content) VALUES(%s,%s,'user',%s)",
                    (run["conversation_id"], rid, Jsonb({"text": content, "kind": "clarification"})),
                )
                jid = uid()
                c.execute(
                    "INSERT INTO jobs(id,type,resource_id,payload) VALUES(%s,'run',%s,%s)",
                    (jid, rid, Jsonb({"resume": True})),
                )
                event(
                    c,
                    rid,
                    "clarification.submitted",
                    "clarification.submitted",
                    {"content": content, "kb_version_id": version},
                )
                return {"run_id": rid, "job_id": jid}

            response = idempotent(c, f"clarify:{rid}", key, body, op)
        if response.get("expired"):
            raise AppError("CLARIFICATION_EXPIRED", "等待补充信息超时，请创建新问题", 409)
        return response

    @staticmethod
    def expire_one(c, run, clar):
        c.execute("UPDATE clarifications SET status='expired' WHERE run_id=%s", (run["id"],))
        payload = clar["payload"]
        finalize(
            c,
            run["id"],
            final_result(
                "insufficient",
                "CLARIFICATION_TIMEOUT",
                "等待补充信息超时，现有依据仍不足。",
                payload.get("citations"),
                payload.get("information_gaps"),
            ),
        )

    def sweep(self):
        with self.db.tx() as c:
            rows = c.execute(
                "SELECT * FROM runs WHERE execution_status='waiting' AND deadline_at<=clock_timestamp() ORDER BY deadline_at FOR UPDATE SKIP LOCKED LIMIT 50"
            ).fetchall()
            for run in rows:
                clar = c.execute(
                    "SELECT * FROM clarifications WHERE run_id=%s FOR UPDATE", (run["id"],)
                ).fetchone()
                if clar and clar["status"] == "pending":
                    self.expire_one(c, run, clar)
        # Lock jobs before runs consistently with Worker paths.
        with self.db.tx() as c:
            jobs = c.execute(
                "SELECT * FROM jobs WHERE status IN ('pending','running') ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 100"
            ).fetchall()
            now = c.execute("SELECT clock_timestamp() t").fetchone()["t"]
            for job in jobs:
                expired = job["status"] == "running" and job["lease_until"] <= now
                run = None
                if job["type"] == "run":
                    run = c.execute(
                        "SELECT * FROM runs WHERE id=%s FOR UPDATE", (job["resource_id"],)
                    ).fetchone()
                queued_out = (
                    job["status"] == "pending"
                    and run
                    and (now - job["created_at"]).total_seconds() >= run["config"]["queue_timeout_seconds"]
                )
                if expired and job["attempts"] < job["max_attempts"]:
                    c.execute(
                        "UPDATE jobs SET status='pending',lease_until=NULL,worker_id=NULL,next_attempt_at=clock_timestamp() WHERE id=%s",
                        (job["id"],),
                    )
                elif expired or queued_out:
                    c.execute(
                        "UPDATE jobs SET status='failed',lease_until=NULL,error=%s WHERE id=%s",
                        (Jsonb({"code": "QUEUE_TIMEOUT" if queued_out else "LEASE_EXHAUSTED"}), job["id"]),
                    )
                    if run:
                        finalize(
                            c,
                            run["id"],
                            final_result(
                                "incomplete",
                                "QUEUE_TIMEOUT" if queued_out else "LEASE_EXHAUSTED",
                                "任务未能在执行期限内完成，请稍后重新提问。",
                            ),
                        )
                    else:
                        c.execute(
                            "UPDATE kb_versions SET status='failed',error=%s WHERE id=%s AND status!='ready'",
                            (Jsonb({"code": "LEASE_EXHAUSTED"}), job["resource_id"]),
                        )
