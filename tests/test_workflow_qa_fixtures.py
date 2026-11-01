"""Offline integrity checks for the WF development set; no KB writes or model calls."""

import ast
import hashlib
import json
import operator
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from xiaoying.config import Settings
from xiaoying.ingestion.parsers import inspect_metadata, parse_document

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "fixtures" / "workflow_qa_cases.json"
EXPECTED_CATEGORIES = {"可回答": 10, "证据不足": 10, "超出范围": 10, "错误前提": 10}


def compact(text):
    return re.sub(r"\s+", "", text)


def read_dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return read_dataset()


def test_workflow_development_set_has_40_unique_questions_with_balanced_coverage(dataset):
    cases = dataset["cases"]
    assert [case["id"] for case in cases] == [f"WQA{n:02d}" for n in range(1, 41)]
    assert len({compact(case["input"]["question"]) for case in cases}) == 40
    assert Counter(case["category"] for case in cases) == EXPECTED_CATEGORIES
    assert dataset["category_counts"] == EXPECTED_CATEGORIES
    assert dataset["corpus_root"] == "data/workflow_manuals"
    assert dataset["company"] == "澄岳示例科技有限公司"
    assert dataset["schema_version"] == 2
    assert dataset["execution_status"] == "not_run"
    assert dataset["business_document_changes"] == []
    for case in cases:
        assert case["execution_status"] == "not_run"
        assert case["input"]["question"] and case["input"]["history"] == []
        assert case["expected"]["reference_answer"]
        assert case["expected"]["required_facts"]
        assert case["expected"]["forbidden_claims"]
        assert case["expected"]["evidence"]


def test_missing_evidence_outside_scope_and_wrong_premises_have_distinct_oracles(dataset):
    statuses = Counter()
    for case in dataset["cases"]:
        expected = case["expected"]
        category = case["category"]
        statuses[expected["status"]] += 1
        if category in {"可回答", "错误前提"}:
            assert expected["status"] == "answered"
            assert expected["information_gaps"] == []
            assert expected["gap_type"] == "none"
        elif category == "证据不足":
            assert expected["information_gaps"]
            if expected["gap_type"] in {"incident_facts", "manual_review"}:
                assert expected["status"] == "awaiting_user"
            else:
                assert expected["status"] == "insufficient"
        elif category == "超出范围":
            assert expected["status"] == "insufficient"
            assert expected["gap_type"] == "outside_scope"
            assert expected["information_gaps"]
            assert all(ev["evidence_role"] == "scope_boundary" for ev in expected["evidence"])
        if category == "错误前提":
            assert expected["error_premise"]
        else:
            assert "error_premise" not in expected
    assert statuses == {"answered": 20, "awaiting_user": 8, "insufficient": 12}


def test_workflow_corpus_snapshot_matches_all_eight_unchanged_deliverables(dataset):
    manifest = json.loads((ROOT / "data" / "workflow_manuals" / "manifest.json").read_text())
    by_id = {item["document_id"]: item for item in manifest["documents"]}
    snapshots = dataset["source_snapshot"]
    assert len(snapshots) == len(by_id) == 8
    assert {item["document_id"] for item in snapshots} == {f"WF{n:02d}" for n in range(1, 9)}
    suffixes = set()
    for item in snapshots:
        path = ROOT / item["path"]
        assert path.parent == ROOT / "data" / "workflow_manuals"
        assert path.name == by_id[item["document_id"]]["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["sha256"] == by_id[item["document_id"]]["sha256"]
        assert by_id[item["document_id"]]["body_han_characters"] > 10000
        suffixes.add(path.suffix)
    assert suffixes == {".txt", ".md", ".docx", ".html", ".pdf"}


@pytest.mark.parametrize("document_id", [f"WF{n:02d}" for n in range(1, 9)])
def test_every_gold_quote_and_locator_exists_in_the_actual_source(document_id, dataset):
    snapshot = next(item for item in dataset["source_snapshot"] if item["document_id"] == document_id)
    path = ROOT / snapshot["path"]
    settings = Settings(_env_file=None, provider_mode="mock", pdf_backend="test_text")
    metadata = inspect_metadata(path)
    document = parse_document(path, metadata, settings)
    assert metadata["document_id"] == document_id
    assert metadata["company"] == dataset["company"]
    quotes = [
        (case["id"], ev)
        for case in dataset["cases"]
        for ev in case["expected"]["evidence"]
        if ev["document_id"] == document_id
    ]
    assert quotes, f"{document_id} has no evidence coverage"
    for case_id, evidence in quotes:
        assert evidence["source_path"] == snapshot["path"]
        assert evidence["locator"]
        assert evidence["quote"]
        assert any(
            block.locator == evidence["source_locator"]
            and compact(evidence["quote"]) in compact(block.text)
            for block in document.blocks
        ), (case_id, evidence)
        if path.suffix == ".pdf":
            assert evidence["source_locator"]["page"] >= 1
            assert f"第 {evidence['source_locator']['page']} 页" in evidence["locator"]
    if path.suffix == ".pdf":
        assert any("TEST_ONLY" in warning for warning in document.warnings)


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(node):
    if isinstance(node, ast.Expression):
        return calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](calculate(node.left), calculate(node.right))
    raise AssertionError("Reference calculations only accept numeric arithmetic")


def test_reference_calculations_are_correct_and_present_in_the_answer(dataset):
    calculations = []
    for case in dataset["cases"]:
        for check in case["expected"].get("calculations", []):
            actual = calculate(ast.parse(check["expression"], mode="eval"))
            assert actual == Decimal(check["result"]), case["id"]
            assert check["result"] in case["expected"]["reference_answer"], case["id"]
            calculations.append(check)
    assert len(calculations) == 9


def test_human_review_catalog_contains_exactly_the_same_40_questions_and_answers(dataset):
    review = (ROOT / "examples" / "workflow-qa-cases.md").read_text(encoding="utf-8")
    assert re.findall(r"^\| (WQA\d{2}) \|", review, re.M) == [c["id"] for c in dataset["cases"]]
    for case in dataset["cases"]:
        assert case["input"]["question"] in review
        assert case["expected"]["reference_answer"] in review
    assert "全部 `not_run`" in review
    assert "均未改动" in review


def test_evaluation_cli_advertises_optional_dataset_without_running_a_model():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert "--cases-file" in result.stdout
    assert "--case" in result.stdout
    assert not result.stderr
