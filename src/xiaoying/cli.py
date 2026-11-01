import argparse
import json
from pathlib import Path
from uuid import uuid4

from xiaoying.config import Settings, load_settings
from xiaoying.domain import BuildRequest
from xiaoying.providers.embeddings import providers
from xiaoying.services.knowledge import KnowledgeService
from xiaoying.services.runs import RunService
from xiaoying.storage.db import Database, create_isolated_database, jsonable, migrate
from xiaoying.worker import Worker


def key():
    return str(uuid4())


def claim_specific(db, settings, job_id):
    with db.tx() as c:
        row = c.execute("SELECT type FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if not row:
            return None
        kind = row["type"]
        c.execute("SELECT pg_advisory_xact_lock(%s)", (831027 if kind == "run" else 831028,))
        running = c.execute(
            "SELECT count(*) n FROM jobs WHERE type=%s AND status='running' AND lease_until>clock_timestamp()",
            (kind,),
        ).fetchone()["n"]
        if running >= (2 if kind == "run" else 1):
            return None
        return jsonable(
            c.execute(
                """UPDATE jobs SET status='running',attempts=attempts+1,
            lease_token=lease_token+1,worker_id='cli',lease_until=clock_timestamp()+(%s * interval '1 second')
            WHERE id=%s AND status='pending' AND attempts<max_attempts AND next_attempt_at<=clock_timestamp()
            RETURNING *""",
                (settings.lease_seconds, job_id),
            ).fetchone()
        )


def main():
    parser = argparse.ArgumentParser(description="企业知识问答 Agent")
    parser.add_argument("--mock", action="store_true", help="明确使用离线 Mock 与 PDF 文本测试适配器")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    seed = commands.add_parser("seed")
    seed.add_argument("--directory", default="data/policies")
    seed.add_argument("--name", default="青屿差旅制度")
    ask = commands.add_parser("ask")
    ask.add_argument("--kb-id", required=True)
    ask.add_argument("question")
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = load_settings(profile="mock" if args.mock else "live")
    if args.command == "init-db":
        if args.mock:
            create_isolated_database(settings.database_url, Settings().database_url)
        migrate(settings.database_url)
        print("数据库迁移完成")
        return
    db = Database(settings.database_url)
    encoder, reranker = providers(settings)
    knowledge = KnowledgeService(db, settings, encoder)
    worker = Worker(db, settings, encoder, reranker)
    try:
        if args.command == "seed":
            kb = knowledge.create(args.name, key())
            changes = []
            for path in sorted(Path(args.directory).glob("*")):
                if not path.is_file() or path.suffix.lower() not in {".md", ".docx", ".pdf", ".html", ".txt"}:
                    continue
                upload = knowledge.upload(kb["id"], path.name, path.read_bytes(), {}, key())
                changes.append(
                    {
                        "action": "add",
                        "document_id": upload["metadata"]["document_id"],
                        "upload_id": upload["upload_id"],
                    }
                )
            build = knowledge.create_build(kb["id"], BuildRequest(changes=changes), key())
            claimed = claim_specific(db, settings, build["job_id"])
            if claimed:
                worker.execute(claimed)
            result = db.one("SELECT status,error,progress FROM jobs WHERE id=%s", (build["job_id"],))
            print(
                json.dumps(
                    {"kb_id": kb["id"], **build, **result, "mode": settings.provider_mode},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if result["status"] != "done":
                raise SystemExit(1)
        elif args.command == "ask":
            runs = RunService(db, settings, encoder)
            conversation = runs.create_conversation(args.kb_id, key())
            request = runs.create_run(conversation["id"], args.question, key())
            claimed = claim_specific(db, settings, request["job_id"])
            if claimed:
                worker.execute(claimed)
            print(json.dumps(runs.get(request["run_id"]), ensure_ascii=False, indent=2))
        elif args.command == "export":
            run = RunService(db, settings, encoder).get(args.run_id)
            events = db.all("SELECT * FROM trace_events WHERE run_id=%s ORDER BY seq", (args.run_id,))
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "run": run,
                        "trace": events,
                        "evidence_type": "offline_mock"
                        if run["config"]["provider_mode"] == "mock"
                        else "live",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(path)
    finally:
        db.close()


if __name__ == "__main__":
    main()
