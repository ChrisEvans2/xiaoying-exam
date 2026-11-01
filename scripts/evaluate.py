"""Run the 20 cases with a real model. Only status is machine-checked; facts require review."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from xiaoying.cli import claim_specific
from xiaoying.config import load_settings
from xiaoying.providers.embeddings import providers
from xiaoying.storage.db import Database
from xiaoying.worker import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--case", default="all", help="all or QA01..QA20")
    parser.add_argument("--output", default="examples/live")
    args = parser.parse_args()
    settings = load_settings(profile="live")
    if settings.provider_mode != "live" or not settings.deepseek_api_key:
        parser.error("真实评估需要 PROVIDER_MODE=live 和 DEEPSEEK_API_KEY；离线记录请使用 demo_cases.py")
    dataset = json.loads(Path("tests/fixtures/qa_cases.json").read_text())
    cases = [c for c in dataset["cases"] if args.case in {"all", c["id"]}]
    if not cases:
        parser.error("未知用例 ID")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    embedder, reranker = providers(settings)
    db = Database(settings.database_url)
    try:
        baseline = db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (args.kb_id,))
        if not baseline or not baseline["current_version_id"]:
            raise RuntimeError("评估知识库尚未发布")
        version = baseline["current_version_id"]
        manifest = db.all(
            """SELECT d.document_id,u.file_hash FROM kb_version_documents m
            JOIN documents d ON d.id=m.document_fk JOIN document_versions v ON v.id=m.document_version_id
            JOIN uploads u ON u.id=v.upload_id WHERE m.kb_version_id=%s""",
            (version,),
        )
        expected = {x["document_id"]: x["sha256"] for x in dataset["source_snapshot"]}
        if {x["document_id"]: x["file_hash"] for x in manifest} != expected:
            raise RuntimeError("评估库必须是20题对应的8份原始文件快照；文档ID或SHA不匹配")
        config = db.one("SELECT config FROM kb_versions WHERE id=%s", (version,))["config"]
        if (
            config["provider_mode"] != "live"
            or config["pdf_backend"] != "mineru"
            or config["embedding_identity"] != embedder.identity
        ):
            raise RuntimeError("评估库必须使用相同BGE修订及MinerU构建真实索引")

        def ensure_baseline():
            if (
                db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (args.kb_id,))[
                    "current_version_id"
                ]
                != version
            ):
                raise RuntimeError("评估过程中发布版本已变化，本次评估已停止，请使用固定的隔离知识库")

        # Warm models before starting the bounded per-question execution clock.
        embedder.encode(["模型预热"])
        reranker.rank("模型预热", ["模型预热"])
        for case in cases:
            ensure_baseline()
            setup = case.get("setup", {})
            cfg = settings.model_copy(
                update={"max_tool_calls": setup.get("max_tool_calls", settings.max_tool_calls)}
            )
            fault = "always" if case["id"] == "QA20" else 1 if case["id"] == "QA19" else 0
            worker = Worker(db, cfg, embedder, reranker, faults={"read": fault})
            # QA19/20 deliberately expose only D04 metadata in search results.
            if setup:

                class FilteredRetriever:
                    def __init__(self, base):
                        self.base = base

                    def search(self, kb, version, **kwargs):
                        kwargs["document_ids"] = ["D04"]
                        return self.base.search(kb, version, **kwargs)

                worker.retriever = FilteredRetriever(worker.retriever)
            runs = worker.runs
            key = lambda: str(uuid4())
            cid = runs.create_conversation(args.kb_id, key())["id"]

            def execute(request, worker=worker, cfg=cfg, runs=runs):
                job = claim_specific(db, cfg, request["job_id"])
                if not job:
                    raise RuntimeError("评估任务未能领取；请停止后台 Worker 或使用隔离数据库")
                worker.execute(job)
                record = runs.get(request["run_id"])
                ensure_baseline()
                if record["kb_version_id"] != version:
                    raise RuntimeError("问题使用了非基线版本，拒绝输出无效评估记录")
                return record

            history = []
            # Re-run prior user questions; never inject expected assistant answers into the model.
            for message in case["input"]["history"]:
                if message["role"] == "user":
                    prior = execute(runs.create_run(cid, message["content"], key()))
                    history.append(prior)
                    if prior["execution_status"] == "waiting":
                        runs.clarify(prior["id"], {"action": "abandon"}, key())
            request = runs.create_run(cid, case["input"]["question"], key())
            initial = execute(request)
            final = initial
            if initial["execution_status"] == "waiting" and case.get("followup"):
                resume = runs.clarify(initial["id"], {"action": "submit", **case["followup"]["input"]}, key())
                final = execute(resume)
            report = {
                "evidence_type": "live_model",
                "baseline_version_id": version,
                "case_id": case["id"],
                "input": case["input"],
                "history_runs": history,
                "initial": initial,
                "final": final,
                "trace": db.all("SELECT * FROM trace_events WHERE run_id=%s ORDER BY seq", (initial["id"],)),
                "initial_status_matches": initial["task_status"] == case["expected"]["status"],
                "final_status_matches": final["task_status"]
                == case.get("followup", {}).get("expected_status", case["expected"]["status"]),
                "semantic_verdict": "needs_human_review",
                "expected": case["expected"],
            }
            (output / (case["id"] + ".json")).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(case["id"], initial["task_status"], "→", final["task_status"], "facts/quotes: needs review")
    finally:
        db.close()


if __name__ == "__main__":
    main()
