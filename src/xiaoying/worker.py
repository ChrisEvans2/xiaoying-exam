import argparse
import os
from pathlib import Path
import random
import threading
import time
from uuid import uuid4
from psycopg.types.json import Jsonb

from xiaoying.agent.runtime import AgentRuntime
from xiaoying.config import load_settings
from xiaoying.domain import LeaseLost
from xiaoying.providers.embeddings import providers
from xiaoying.rag.retrieval import Retriever
from xiaoying.services.knowledge import KnowledgeService
from xiaoying.services.runs import RunService, finalize, final_result
from xiaoying.storage.db import Database, fence, jsonable


class Worker:
    def __init__(self, db, settings, embedder=None, reranker=None, chat=None, faults=None):
        self.db, self.settings = db, settings
        self.embedder, self.reranker = (embedder, reranker) if embedder is not None else providers(settings)
        self.knowledge = KnowledgeService(db, settings, self.embedder)
        self.runs = RunService(db, settings, self.embedder)
        self.retriever = Retriever(db, self.embedder, self.reranker, settings.tool_timeout_seconds)
        self.chat, self.faults = chat, faults
        self.worker_id = str(uuid4())

    def claim(self, kind):
        with self.db.tx() as c:
            c.execute("SELECT pg_advisory_xact_lock(%s)", (831027 if kind == "run" else 831028,))
            running = c.execute(
                "SELECT count(*) n FROM jobs WHERE type=%s AND status='running' AND lease_until>clock_timestamp()",
                (kind,),
            ).fetchone()["n"]
            if running >= (2 if kind == "run" else 1):
                return None
            job = c.execute(
                """SELECT * FROM jobs WHERE type=%s AND status='pending' AND next_attempt_at<=clock_timestamp()
                AND attempts<max_attempts ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1""",
                (kind,),
            ).fetchone()
            if not job:
                return None
            row = c.execute(
                """UPDATE jobs SET status='running',attempts=attempts+1,worker_id=%s,lease_token=lease_token+1,
                lease_until=clock_timestamp()+(%s * interval '1 second') WHERE id=%s RETURNING *""",
                (self.worker_id, self.settings.lease_seconds, job["id"]),
            ).fetchone()
            return jsonable(row)

    def execute(self, job):
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(self.settings.heartbeat_seconds):
                try:
                    with self.db.tx() as c:
                        fence(c, job["id"], job["lease_token"])
                        c.execute(
                            "UPDATE jobs SET lease_until=clock_timestamp()+(%s * interval '1 second') WHERE id=%s",
                            (self.settings.lease_seconds, job["id"]),
                        )
                except Exception:
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if job["type"] == "build":
                self.knowledge.build(job)
            else:
                AgentRuntime(self.db, self.settings, self.runs, self.retriever, self.chat, self.faults).run(
                    job
                )
        except LeaseLost:
            pass
        except Exception as exc:
            error = {
                "code": getattr(exc, "code", type(exc).__name__),
                "message": getattr(exc, "message", "任务执行失败"),
            }
            try:
                with self.db.tx() as c:
                    current = fence(c, job["id"], job["lease_token"])
                    retry = getattr(exc, "retryable", False) and current["attempts"] < current["max_attempts"]
                    if retry:
                        delay = 2 ** current["attempts"] + random.random()
                        c.execute(
                            "UPDATE jobs SET status='pending',lease_until=NULL,next_attempt_at=clock_timestamp()+(%s * interval '1 second'),error=%s WHERE id=%s",
                            (delay, Jsonb(error), job["id"]),
                        )
                    else:
                        c.execute(
                            "UPDATE jobs SET status='failed',lease_until=NULL,error=%s WHERE id=%s",
                            (Jsonb(error), job["id"]),
                        )
                        if job["type"] == "build":
                            c.execute(
                                "UPDATE kb_versions SET status='failed',error=%s WHERE id=%s AND status!='ready'",
                                (Jsonb(error), job["resource_id"]),
                            )
                        else:
                            finalize(
                                c,
                                job["resource_id"],
                                final_result("incomplete", error["code"], error["message"]),
                            )
            except LeaseLost:
                pass
        finally:
            stopped.set()
            thread.join(timeout=2)

    def once(self, kind="run"):
        self.runs.sweep()
        job = self.claim(kind)
        if job:
            self.execute(job)
        return job

    def serve(self, kind):
        while True:
            if not self.once(kind):
                time.sleep(self.settings.scan_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["run", "build", "scheduler"], default="run")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = load_settings()
    db = Database(settings.database_url)
    if not 1 <= args.concurrency <= 4:
        parser.error("concurrency must be 1—4")
    worker = Worker(db, settings)
    if args.ready_file:
        args.ready_file.write_text(str(os.getpid()))
    if args.once:
        worker.runs.sweep() if args.kind == "scheduler" else worker.once(args.kind)
        db.close()
        return
    if args.kind == "scheduler":
        while True:
            worker.runs.sweep()
            time.sleep(settings.scan_seconds)
    if args.concurrency == 1:
        worker.serve(args.kind)
    threads = [threading.Thread(target=worker.serve, args=(args.kind,)) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
