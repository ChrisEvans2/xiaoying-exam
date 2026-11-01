"""Real-model exam evidence in a fresh database; never writes to the business database.

Run from the repository root. Source review is recorded separately from status checks.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from uuid import uuid4

from docx import Document
from psycopg.conninfo import make_conninfo

from xiaoying.cli import claim_specific
from xiaoying.config import MockSettings, load_settings, require_separate_database
from xiaoying.domain import BuildRequest
from xiaoying.providers.embeddings import providers
from xiaoying.storage.db import Database, create_isolated_database, migrate
from xiaoying.worker import Worker

DEFAULT_CASES = ["QA01", "QA03", "QA04", "QA11", "QA16", "QA17", "QA18", "QA19", "QA20"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolated_settings(business, name, directory):
    if not re.fullmatch(r"xiaoying_acceptance_[a-z0-9_]+", name):
        raise ValueError("评估数据库名必须以 xiaoying_acceptance_ 开头")
    url = make_conninfo(business.database_url, dbname=name)
    require_separate_database(url, business.database_url)
    require_separate_database(url, MockSettings().database_url)
    live, target = business.data_dir.resolve(), directory.resolve()
    if target.is_relative_to(live) or live.is_relative_to(target):
        raise ValueError("评估文件目录不得与业务文件目录重叠")
    return business.model_copy(update={"database_url": url, "data_dir": target})


def execute(worker, request):
    job = claim_specific(worker.db, worker.settings, request["job_id"])
    if not job:
        raise RuntimeError("无法领取独立评估任务；拒绝让其他 Worker 代跑")
    worker.execute(job)
    job = worker.db.one("SELECT status,error,progress FROM jobs WHERE id=%s", (request["job_id"],))
    if job["status"] != "done":
        raise RuntimeError(f"评估任务未完成：{job}")
    return job


class FaultSearch:
    """Fault cases see D04 candidate metadata only; the decision model stays real."""

    def __init__(self, base):
        self.base = base

    def search(self, kb, version, **kwargs):
        return self.base.search(kb, version, **{**kwargs, "document_ids": ["D04"]})


def export_run(worker, rid):
    db = worker.db
    record = worker.runs.get(rid)
    reads = db.all("SELECT payload FROM evidence_reads WHERE run_id=%s ORDER BY id", (rid,))
    by_id = {row["payload"]["evidence_id"]: row["payload"] for row in reads}
    citations = (record.get("result") or {}).get("citations", [])
    return {
        "run": record,
        "trace": db.all("SELECT * FROM trace_events WHERE run_id=%s ORDER BY seq", (rid,)),
        "read_evidence": [row["payload"] for row in reads],
        "model_calls": db.all(
            """SELECT call_number,phase,adapter,model,status,elapsed_ms,error
            FROM model_invocations WHERE run_id=%s ORDER BY call_number""", (rid,),
        ),
        "citation_integrity": all(
            cite["evidence_id"] in by_id
            and cite["quote"] in by_id[cite["evidence_id"]]["quote"]
            and cite["kb_version_id"] == record["kb_version_id"]
            for cite in citations
        ),
    }


def run_case(db, settings, embedder, reranker, kb, case):
    setup = case.get("setup", {})
    cfg = settings.model_copy(update={"max_tool_calls": setup.get("max_tool_calls", settings.max_tool_calls)})
    fault = "always" if case["id"] == "QA20" else 1 if case["id"] == "QA19" else 0
    worker = Worker(db, cfg, embedder, reranker, faults={"read": fault})
    if fault:
        worker.retriever = FaultSearch(worker.retriever)
    runs = worker.runs
    cid = runs.create_conversation(kb, str(uuid4()))["id"]
    history = []
    for message in case["input"]["history"]:
        if message["role"] != "user":
            continue
        request = runs.create_run(cid, message["content"], str(uuid4()))
        execute(worker, request)
        record = runs.get(request["run_id"])
        history.append(export_run(worker, request["run_id"]))
        if record["execution_status"] == "waiting":
            runs.clarify(record["id"], {"action": "abandon"}, str(uuid4()))
    request = runs.create_run(cid, case["input"]["question"], str(uuid4()))
    execute(worker, request)
    initial = runs.get(request["run_id"])
    if initial["execution_status"] == "waiting" and case.get("followup"):
        response = {"action": "submit", **case["followup"]["input"]}
        execute(worker, runs.clarify(initial["id"], response, str(uuid4())))
    report = {
        "evidence_type": "live_model",
        "case_id": case["id"],
        "input": case["input"],
        "followup_input": case.get("followup", {}).get("input"),
        "fault_injection": {"read": fault, "search_document_ids": ["D04"] if fault else None},
        "history_runs": history,
        "initial": initial,
        **export_run(worker, request["run_id"]),
        "expected": case["expected"],
        "expected_final_status": case.get("followup", {}).get("expected_status", case["expected"]["status"]),
        "semantic_verdict": "pending_source_review",
    }
    report["initial_status_matches"] = initial["task_status"] == case["expected"]["status"]
    report["final_status_matches"] = report["run"]["task_status"] == report["expected_final_status"]
    return report


def freeze_sources(dataset, output, revision):
    """Bind evaluation to exact bytes, including when another session edits the workspace."""
    paths = {}
    for entry in dataset["source_snapshot"]:
        path = Path(entry["path"])
        data = subprocess.check_output(["git", "show", f"{revision}:{path.as_posix()}"]) if revision else path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"所选资料快照哈希不匹配：{path}")
        frozen = output / "sources/baseline" / path.name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists() and frozen.read_bytes() != data:
            raise RuntimeError("拒绝覆盖既有评估资料快照")
        frozen.write_bytes(data)
        paths[entry["document_id"]] = frozen
    return paths


def build_baseline(worker, dataset, paths):
    service = worker.knowledge
    kb = service.create("考试真实模型验收（隔离）", str(uuid4()))["id"]
    changes = []
    for entry in dataset["source_snapshot"]:
        path = paths[entry["document_id"]]
        if sha(path) != entry["sha256"]:
            raise RuntimeError(f"原始资料哈希变化：{path}")
        upload = service.upload(kb, path.name, path.read_bytes(), {}, str(uuid4()))
        changes.append({"action": "add", "document_id": entry["document_id"], "upload_id": upload["upload_id"]})
    request = service.create_build(kb, BuildRequest(changes=changes), str(uuid4()))
    job = execute(worker, request)
    return kb, request["version_id"], job


def replacement(worker, kb, baseline, output, case, original):
    service = worker.knowledge
    document = Document(original)
    replacements = 0
    for paragraph in document.paragraphs:
        if "500" in paragraph.text:
            paragraph.text = paragraph.text.replace("500", "600")
            replacements += 1
    if replacements != 1:
        raise RuntimeError("替换实验要求只修改一处500→600")
    stream = io.BytesIO()
    document.save(stream)
    revised_path = output / "sources/D04-after.docx"
    revised_path.parent.mkdir(parents=True, exist_ok=True)
    revised_path.write_bytes(stream.getvalue())
    upload = service.upload(kb, "D04-accommodation.docx", stream.getvalue(), {}, str(uuid4()))
    request = service.create_build(kb, BuildRequest(
        base_version_id=baseline,
        changes=[{"action": "replace", "document_id": "D04", "upload_id": upload["upload_id"]}],
    ), str(uuid4()))
    job = execute(worker, request)
    variant = json.loads(json.dumps(case))
    variant["id"] = "REPLACEMENT"
    variant["expected"] = json.loads(json.dumps(case["expected"], ensure_ascii=False).replace("500", "600"))
    report = run_case(worker.db, worker.settings, worker.embedder, worker.reranker, kb, variant)
    report["replacement"] = {
        "before_version_id": baseline, "after_version_id": request["version_id"],
        "source_file": "sources/D04-after.docx", "source_sha256": sha(revised_path),
        "change": "D04上海等城市住宿上限500→600；原始仓库资料保持原值", "build": job,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/live/exam-20261102"))
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--source-revision", help="可选的40位Git提交SHA；只读提取已知资料快照，保留工作区改动")
    parser.add_argument("--resume", action="store_true", help="继续同目录已经导出的用例，不覆盖原记录")
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--source-revision需要完整40位提交SHA")
    business = load_settings(profile="live")
    if not business.deepseek_api_key:
        parser.error("需要本机真实DeepSeek密钥；不接受Mock冒充验收")
    dataset = json.loads(Path("tests/fixtures/qa_cases.json").read_text())
    cases = {case["id"]: case for case in dataset["cases"]}
    if any(case not in cases for case in args.cases) or "QA01" not in args.cases:
        parser.error("请选择已有QA编号；替换对照要求包含QA01")
    manifest_path = args.output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["case_ids"] != args.cases or manifest["source_snapshot"] != dataset["source_snapshot"]
                or manifest.get("source_revision") != args.source_revision):
            parser.error("恢复时不允许更换用例或资料快照")
        if manifest["code_snapshot"] != {str(p): sha(p) for p in sorted(Path("src").rglob("*.py"))}:
            parser.error("源码已变化；请新建输出目录，保留旧尝试并重新验收")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("输出目录非空；请换目录或显式--resume，保留既有尝试")
        manifest = {
            "evidence_type": "live_model_exam_acceptance",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "database_name": "xiaoying_acceptance_" + uuid4().hex[:12],
            "case_ids": args.cases, "source_snapshot": dataset["source_snapshot"],
            "source_revision": args.source_revision,
            "business_database_writes": False,
            "code_snapshot": {str(p): sha(p) for p in sorted(Path("src").rglob("*.py"))},
            "runtime_versions": {name: package_version(name) for name in (
                "langgraph", "langgraph-checkpoint-postgres", "openai", "psycopg", "sentence-transformers",
            )},
        }
        save(manifest_path, manifest)
    source_paths = freeze_sources(dataset, args.output, args.source_revision)
    directory = Path(".runtime/exam-acceptance") / manifest["database_name"] / "files"
    settings = isolated_settings(business, manifest["database_name"], directory)
    create_isolated_database(settings.database_url, business.database_url)
    migrate(settings.database_url)
    db = Database(settings.database_url)
    embedder, reranker = providers(settings)
    try:
        print("正在加载真实BGE向量与重排模型", flush=True)
        embedder.encode(["模型预热"])
        reranker.rank("模型预热", ["模型预热"])
        worker = Worker(db, settings, embedder, reranker)
        if "kb_id" not in manifest:
            kb, version, job = build_baseline(worker, dataset, source_paths)
            manifest.update(kb_id=kb, baseline_version_id=version, baseline_build=job)
            manifest["models"] = {"chat": settings.deepseek_model, "embedding": embedder.identity,
                                  "reranker": settings.reranker_model, "reranker_revision": settings.reranker_revision,
                                  "pdf_backend": settings.pdf_backend, "run_config": settings.run_config()}
            save(manifest_path, manifest)
            print("八份原始制度真实索引就绪", flush=True)
        current = db.one("SELECT current_version_id FROM knowledge_bases WHERE id=%s", (manifest["kb_id"],))
        for case_id in args.cases:
            path = args.output / (case_id + ".json")
            if path.exists():
                continue
            if current["current_version_id"] != manifest["baseline_version_id"]:
                raise RuntimeError("基线版本已变化，拒绝将替换后数据用于原始用例")
            report = run_case(db, settings, embedder, reranker, manifest["kb_id"], cases[case_id])
            save(path, report)
            print(case_id, report["initial"]["task_status"], "→", report["run"]["task_status"],
                  "status_matches=", report["initial_status_matches"] and report["final_status_matches"], flush=True)
        if not (args.output / "REPLACEMENT.json").exists():
            report = replacement(worker, manifest["kb_id"], manifest["baseline_version_id"], args.output,
                                 cases["QA01"], source_paths["D04"])
            save(args.output / "REPLACEMENT.json", report)
            print("REPLACEMENT", report["run"]["task_status"], flush=True)
        for entry in dataset["source_snapshot"]:
            if sha(source_paths[entry["document_id"]]) != entry["sha256"]:
                raise RuntimeError("验收期间冻结资料被修改")
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["frozen_sources_unchanged"] = True
        manifest["workspace_differences"] = [entry["path"] for entry in dataset["source_snapshot"]
                                             if sha(Path(entry["path"])) != entry["sha256"]]
        save(manifest_path, manifest)
    finally:
        db.close()


if __name__ == "__main__":
    main()
