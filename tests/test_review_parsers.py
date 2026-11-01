"""Source preservation and malformed input contracts; PDF fixture mode is explicitly offline."""

import json
import math
import sys
from pathlib import Path

import pytest
from docx import Document

from xiaoying.config import Settings
from xiaoying.domain import AppError, ChunkConfig
from xiaoying.ingestion.chunking import chunk_document
from xiaoying.ingestion.parsers import inspect_metadata, mineru_blocks, parse_document, parse_pdf
from xiaoying.providers.embeddings import MockEmbedder

ROOT = Path(__file__).resolve().parents[1]
POLICIES = sorted((ROOT / "data" / "policies").glob("D*"))


@pytest.fixture
def settings():
    return Settings(_env_file=None, provider_mode="mock", pdf_backend="test_text")


@pytest.mark.parametrize("path", POLICIES, ids=lambda path: path.stem)
def test_original_fixture_quotes_are_preserved_in_parsed_sources(path, settings):
    metadata = inspect_metadata(path)
    document = parse_document(path, metadata, settings)
    cases = json.loads((ROOT / "tests" / "fixtures" / "qa_cases.json").read_text())["cases"]
    quotes = {
        evidence["quote"]
        for case in cases
        for evidence in case["expected"]["evidence"]
        if evidence["document_id"] == metadata["document_id"]
    }
    assert quotes
    for quote in quotes:
        # PDF text extraction contains physical line wraps; preserve them in source storage,
        # but do not count a visual wrap as a changed business statement here.
        assert any(quote in block.text.replace("\n", "") for block in document.blocks), quote
    assert [block.order for block in document.blocks] == list(range(len(document.blocks)))
    assert len({block.block_id for block in document.blocks}) == len(document.blocks)
    assert all(block.locator for block in document.blocks)
    if path.suffix == ".pdf":
        assert document.warnings and "TEST_ONLY" in document.warnings[0]


def test_markdown_table_keeps_its_section_and_source_line(tmp_path, settings):
    path = tmp_path / "table.md"
    path.write_text("# 差旅\n\n## 住宿\n\n| 城市 | 金额 |\n| --- | --- |\n| 上海 | 500 |\n\n后续说明。\n")
    document = parse_document(path, {}, settings)
    table = next(block for block in document.blocks if block.type == "table")
    assert table.heading_path == ["差旅", "住宿"]
    assert table.locator["line_start"] == 5
    assert document.blocks[-1].text == "后续说明。"


def test_long_markdown_table_does_not_separate_amounts_from_headers(tmp_path, settings):
    path = tmp_path / "table.md"
    path.write_text("# 住宿\n\n| 城市 | 金额 |\n| --- | --- |\n" + "".join(f"| 上海{i} | {500+i} |\n" for i in range(20)))
    document = parse_document(path, {}, settings)
    chunks = chunk_document(document, MockEmbedder(), ChunkConfig(target_tokens=32, max_tokens=64, overlap_tokens=0))
    assert len(chunks) > 1
    for chunk in chunks:
        if "上海" in chunk.text:
            assert "城市" in chunk.text and "金额" in chunk.text


def test_word_preserves_paragraph_table_interleaving_and_header(tmp_path, settings):
    path = tmp_path / "table.docx"
    document = Document()
    document.add_heading("住宿标准", level=1)
    document.add_paragraph("表格之前。")
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [("城市", "金额"), ("上海", "500"), ("杭州", "350")]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.add_paragraph("表格之后。")
    document.save(path)
    parsed = parse_document(path, {}, settings)
    assert [part.type for part in parsed.blocks] == ["paragraph", "table_row", "table_row", "paragraph"]
    assert parsed.blocks[0].text == "表格之前。"
    assert parsed.blocks[-1].text == "表格之后。"
    assert all("城市 | 金额" in part.text for part in parsed.blocks if part.type == "table_row")


def test_html_removes_executable_hidden_noise_and_preserves_nested_items(tmp_path, settings):
    path = tmp_path / "article.html"
    path.write_text("""<html><body><nav>导航噪声</nav><article><h1>规定</h1>
    <script>alert('代码不能成为正文')</script><p hidden>隐藏金额999</p>
    <p style="display: none">隐藏金额888</p><ol><li>交通凭证<ul><li>必须真实</li></ul></li></ol>
    <table><tr><th>城市</th><th>金额</th></tr><tr><td>上海</td><td>500</td></tr></table>
    </article></body></html>""")
    parsed = parse_document(path, {}, settings)
    text = "\n".join(part.text for part in parsed.blocks)
    assert "导航" not in text and "隐藏" not in text and "alert" not in text
    assert text.count("必须真实") == 1
    row = next(part for part in parsed.blocks if part.type == "table_row")
    assert "城市" in row.text and "金额" in row.text and "500" in row.text


def test_html_plain_div_body_is_not_silently_lost(tmp_path, settings):
    path = tmp_path / "simple.html"
    path.write_text("<html><body><div>住宿上限为每晚500元。</div></body></html>")
    parsed = parse_document(path, {}, settings)
    assert "住宿上限为每晚500元。" in "".join(part.text for part in parsed.blocks)


@pytest.mark.parametrize("item", [
    {"type": "text", "text": "正文", "page_idx": -1, "bbox": [0, 0, 10, 10]},
    {"type": "text", "text": "正文", "page_idx": 0, "bbox": [10, 10, 0, 0]},
    {"type": "text", "text": "正文", "page_idx": 0, "bbox": [0, 0, math.nan, 10]},
    {"type": "text", "text": "正文", "page_idx": 0, "bbox": [0, 0, 1001, 10]},
])
def test_mineru_invalid_page_or_bbox_cannot_generate_false_location(item):
    with pytest.raises(AppError) as error:
        mineru_blocks([item])
    assert error.value.code == "MINERU_SCHEMA"


@pytest.mark.parametrize("item", [None, "unexpected", {"type": "text", "text": None, "page_idx": 0}])
def test_unknown_mineru_item_schema_is_rejected_explicitly(item):
    with pytest.raises(AppError) as error:
        mineru_blocks([item])
    assert error.value.code == "MINERU_SCHEMA"


def test_mineru_columns_are_kept_separate():
    blocks = mineru_blocks([
        {"type": "text", "text": "交通政策。", "page_idx": 0, "bbox": [10, 10, 450, 80]},
        {"type": "text", "text": "住宿政策。", "page_idx": 0, "bbox": [550, 10, 990, 80]},
    ])
    assert blocks[0].locator["page"] == 1
    assert blocks[1].hard_boundary


def test_mineru_process_has_bounded_timeout(tmp_path):
    config = Settings(
        _env_file=None,
        provider_mode="mock",
        pdf_backend="mineru",
        pdf_timeout_seconds=0.05,
        mineru_command=[sys.executable, "-c", "import time; time.sleep(5)"],
    )
    with pytest.raises(AppError) as error:
        parse_pdf(POLICIES[0], config, tmp_path / "parse")
    assert error.value.code == "PDF_TIMEOUT"


def test_missing_mineru_does_not_silently_fall_back_to_test_parser(tmp_path):
    config = Settings(
        _env_file=None,
        provider_mode="mock",
        pdf_backend="mineru",
        mineru_command=[str(tmp_path / "does-not-exist")],
    )
    with pytest.raises(AppError) as error:
        parse_pdf(POLICIES[0], config, tmp_path / "parse")
    assert error.value.code == "MINERU_UNAVAILABLE"
