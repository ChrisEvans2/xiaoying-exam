"""Execute declared scripted fixtures against real PG + LangGraph. Not a model evaluation."""

import argparse
import io
import json
import re
from pathlib import Path
from uuid import uuid4

from docx import Document

from xiaoying.cli import claim_specific
from xiaoying.config import Settings, load_settings
from xiaoying.domain import BuildRequest
from xiaoying.providers.embeddings import providers
from xiaoying.storage.db import Database, create_isolated_database, migrate
from xiaoying.worker import Worker


class FixtureChat:
    """A deliberately scripted test double; each decision observes actual tool outputs."""

    def __init__(self, conflict=False):
        self.conflict = conflict
        self.calls = 0

    def expand(self, question, history):
        return {"resolved_question": question, "expanded_queries": [], "unresolved_slots": []}

    def decide(self, messages):
        observations = [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]
        docs = ["D01", "D02"] if self.conflict else ["D04"]
        evidence = [e for o in observations for e in o.get("evidence", [])]
        read_docs = {e["document_id"] for e in evidence}
        if not observations:
            name, args = (
                "search_knowledge",
                {"queries": ["报销提交期限" if self.conflict else "上海住宿标准"], "document_ids": docs},
            )
        elif any(o.get("task_status") == "awaiting_user" for o in observations):
            name, args = (
                "finish_answer",
                {
                    "task_status": "insufficient",
                    "answer": "补充后仍无优先级说明，无法在30天与15天之间确定期限。",
                    "citations": evidence,
                    "information_gaps": ["两份制度的优先级或替代关系"],
                    "reason_code": "SCRIPTED_CONFLICT",
                },
            )
        elif set(docs) - read_docs:
            name, args = "read_document", {"document_id": next(d for d in docs if d not in read_docs)}
        elif self.conflict:
            quotes = [e for e in evidence if "自然日" in e["quote"]]
            name, args = (
                "request_clarification",
                {
                    "answer": "D01与D02的期限分别为30天、15天，现有资料没有说明优先级。",
                    "citations": quotes,
                    "information_gaps": ["两份制度的优先级或替代关系"],
                    "questions": ["能否补充财务对两份制度适用顺序的说明？"],
                },
            )
        else:
            quotes = [e for e in evidence if "上海" in e["quote"] or "含税住宿" in e["quote"]]
            amount = re.search(r"每人每晚\s*(\d+)\s*元", quotes[0]["quote"]).group(1)
            name, args = (
                "finish_answer",
                {
                    "task_status": "answered",
                    "answer": f"脚本夹具从原文提取：上海每人每晚上限{amount}元；按含税实际费用报销，超出自付。",
                    "citations": quotes,
                    "information_gaps": [],
                    "reason_code": "SCRIPTED_FIXTURE",
                },
            )
        args["reason"] = "脚本夹具：根据已返回的工具观察选择测试下一步"
        self.calls += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"fixture-{self.calls}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }


def execute(worker, request):
    claimed = claim_specific(worker.db, worker.settings, request["job_id"])
    if not claimed:
        raise RuntimeError("任务未能领取；请停止后台 Worker 后重试演示，或使用隔离数据库")
    worker.execute(claimed)


def save(db, runs, request, path, **extra):
    record = {
        "evidence_type": "scripted_fixture_not_model_evaluation",
        "run": runs.get(request["run_id"]),
        "trace": db.all("SELECT * FROM trace_events WHERE run_id=%s ORDER BY seq", (request["run_id"],)),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/scripted")
    args = parser.parse_args()
    settings = load_settings(profile="mock")
    create_isolated_database(settings.database_url, Settings().database_url)
    migrate(settings.database_url)
    db = Database(settings.database_url)
    embedder, reranker = providers(settings)
    worker = Worker(db, settings, embedder, reranker)
    service, runs = worker.knowledge, worker.runs
    key = lambda: str(uuid4())
    try:
        kb = service.create("离线验收夹具（非真实模型）", key())["id"]
        changes = []
        for path in sorted(Path("data/policies").iterdir()):
            if path.suffix not in {".md", ".pdf", ".docx", ".html", ".txt"}:
                continue
            upload = service.upload(kb, path.name, path.read_bytes(), {}, key())
            changes.append(
                {
                    "action": "add",
                    "document_id": upload["metadata"]["document_id"],
                    "upload_id": upload["upload_id"],
                }
            )
        build = service.create_build(kb, BuildRequest(changes=changes), key())
        execute(worker, build)
        assert db.one("SELECT status FROM jobs WHERE id=%s", (build["job_id"],))["status"] == "done"
        output = Path(args.output)
        summary = []
        for name, conflict, fault in [
            ("normal", False, 0),
            ("conflict", True, 0),
            ("read-recovered", False, 1),
            ("read-failed", False, "always"),
        ]:
            worker.chat, worker.faults = FixtureChat(conflict), {"read": fault}
            cid = runs.create_conversation(kb, key())["id"]
            request = runs.create_run(
                cid, "报销提交期限多久？" if conflict else "上海住宿每晚最多报销多少？", key()
            )
            execute(worker, request)
            initial = runs.get(request["run_id"])["result"]
            expected_initial = (
                "awaiting_user" if conflict else "incomplete" if fault == "always" else "answered"
            )
            assert initial["task_status"] == expected_initial, (name, initial)
            if conflict:
                resumed = runs.clarify(
                    request["run_id"], {"action": "submit", "content": "没有进一步优先级说明"}, key()
                )
                execute(worker, resumed)
            record = save(db, runs, request, output / (name + ".json"), initial_result=initial)
            assert record["run"]["task_status"] == ("insufficient" if conflict else expected_initial)
            if name == "read-recovered":
                assert sum(event["type"] == "tool.failed" for event in record["trace"]) == 1
            if name == "read-failed":
                assert record["run"]["result"]["citations"] == []
            summary.append(
                {"case": name, "status": record["run"]["task_status"], "run_id": request["run_id"]}
            )
        # Isolated replacement experiment: original corpus files are never overwritten.
        document = Document("data/policies/D04-accommodation.docx")
        for paragraph in document.paragraphs:
            if "500" in paragraph.text:
                paragraph.text = paragraph.text.replace("500", "600")
        stream = io.BytesIO()
        document.save(stream)
        upload = service.upload(kb, "D04-accommodation.docx", stream.getvalue(), {}, key())
        revised = service.create_build(
            kb,
            BuildRequest(
                base_version_id=build["version_id"],
                changes=[{"action": "replace", "document_id": "D04", "upload_id": upload["upload_id"]}],
            ),
            key(),
        )
        execute(worker, revised)
        worker.chat, worker.faults = FixtureChat(), {}
        cid = runs.create_conversation(kb, key())["id"]
        request = runs.create_run(cid, "上海住宿每晚最多报销多少？", key())
        execute(worker, request)
        record = save(
            db,
            runs,
            request,
            output / "replacement.json",
            previous_version_id=build["version_id"],
            new_version_id=revised["version_id"],
        )
        assert "600元" in record["run"]["result"]["answer"]
        summary.append(
            {"case": "replacement", "status": record["run"]["task_status"], "run_id": request["run_id"]}
        )
        (output / "summary.json").write_text(
            json.dumps(
                {"evidence_type": "scripted_fixture_not_model_evaluation", "kb_id": kb, "cases": summary},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
