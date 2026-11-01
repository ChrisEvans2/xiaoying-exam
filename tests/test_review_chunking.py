"""Independent algorithm contracts; deterministic vectors are not a BGE quality test."""

import math

import pytest
from pydantic import ValidationError

from xiaoying.domain import AppError, Block, ChunkConfig, DocumentIR
from xiaoying.ingestion.chunking import chunk_document


class TopicEmbedding:
    """Observable sentence vectors let tests vary similarity without a model download."""

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.0, 1.0] if "发票" in text else [1.0, 0.0] for text in texts]

    def count_tokens(self, text):
        return len(text)


def ir(*blocks):
    return DocumentIR(metadata={"document_id": "REVIEW"}, blocks=list(blocks), parser_version="review")


def block(identifier, text, **kwargs):
    return Block(block_id=identifier, text=text, **kwargs)


def test_topic_change_has_real_vector_boundary():
    embedder = TopicEmbedding()
    config = ChunkConfig(target_tokens=400, max_tokens=800, overlap_tokens=0, semantic_threshold=0.7)
    document = ir(
        block("a", "交通费用应当保存行程记录。" * 8, order=0),
        block("b", "交通费用按照审批标准报销。" * 8, order=1),
        block("c", "发票遗失应当补充有效凭证。" * 8, order=2),
    )
    chunks = chunk_document(document, embedder, config)
    assert embedder.calls, "Semantic chunking must actually obtain sentence/paragraph vectors"
    assert len(chunks) >= 2
    assert all(not ("交通" in part.text and "发票" in part.text) for part in chunks)


def test_similar_sentences_can_merge_without_crossing_sections():
    embedder = TopicEmbedding()
    config = ChunkConfig(target_tokens=400, max_tokens=800, overlap_tokens=0)
    document = ir(
        block("a", "交通费用应当保存行程记录。", heading_path=["交通"], order=0),
        block("b", "交通费用按照审批标准报销。", heading_path=["交通"], order=1),
        block("c", "交通费用说明是独立附录。", heading_path=["附录"], order=2),
    )
    chunks = chunk_document(document, embedder, config)
    assert any("行程记录" in part.text and "审批标准" in part.text for part in chunks)
    assert all(not ("附录" in part.text and "行程记录" in part.text) for part in chunks)


def test_hard_clause_boundaries_survive_identical_vectors():
    document = ir(
        block("a", "第一条：交通费用需要审批。", hard_boundary=True, order=0),
        block("b", "第二条：交通费用需要凭证。", hard_boundary=True, order=1),
    )
    chunks = chunk_document(document, TopicEmbedding(), ChunkConfig(overlap_tokens=0))
    assert len(chunks) == 2
    assert chunks[0].text == document.blocks[0].text
    assert chunks[1].text == document.blocks[1].text


def test_long_document_is_bounded_and_keeps_all_original_characters():
    # A paragraph longer than 10,000 characters, with distinct numbered sentences.
    original = "".join(f"交通记录第{i:04d}条应当保存相关费用凭证。" for i in range(650))
    assert len(original) > 10000
    config = ChunkConfig(target_tokens=64, max_tokens=96, overlap_tokens=0)
    chunks = chunk_document(ir(block("long", original)), TopicEmbedding(), config)
    assert len(chunks) > 100
    assert all(0 < part.token_count <= config.max_tokens for part in chunks)
    assert all(len(part.text) <= config.max_tokens for part in chunks)
    assert "".join(part.text.replace("\n", "") for part in chunks) == original
    assert all(part.source_ranges for part in chunks)


def test_unpunctuated_text_cannot_exceed_hard_limit():
    original = "交通记录" * 300
    config = ChunkConfig(target_tokens=32, max_tokens=64, overlap_tokens=0)
    chunks = chunk_document(ir(block("long", original)), TopicEmbedding(), config)
    assert all(part.token_count <= 64 for part in chunks)
    assert "".join(part.text for part in chunks) == original


def test_chunk_ranges_reconstruct_exact_source_with_overlap():
    source = ir(
        block("one", "交通记录应当完整。含税金额需要核实！审批记录不可省略？" * 4),
        block("two", "交通凭证包含\t日期与金额。请保留原文中的空格 和换行。\n", order=1),
    )
    by_id = {part.block_id: part.text for part in source.blocks}
    chunks = chunk_document(
        source, TopicEmbedding(), ChunkConfig(target_tokens=32, max_tokens=64, overlap_tokens=16)
    )
    for chunk in chunks:
        pieces = []
        previous = None
        for span in chunk.source_ranges:
            text = by_id[span["block_id"]]
            assert 0 <= span["start"] < span["end"] <= len(text)
            if previous is not None and previous != span["block_id"]:
                pieces.append("\n\n")
            pieces.append(text[span["start"]:span["end"]])
            previous = span["block_id"]
        assert "".join(pieces) == chunk.text


@pytest.mark.parametrize("vectors", [[[1.0, float("nan")]], [[1.0, 0.0], [1.0, 0.0]]])
def test_invalid_vector_output_fails_without_creating_chunks(vectors):
    class BrokenEmbedding(TopicEmbedding):
        def encode(self, texts):
            return vectors

    with pytest.raises(AppError) as error:
        chunk_document(ir(block("one", "交通费用。")), BrokenEmbedding(), ChunkConfig())
    assert error.value.code == "EMBEDDING_INVALID"


def test_oversized_table_row_never_detaches_amount_from_city_and_condition():
    table = block(
        "table", "城市 | 条件 | 金额\n上海 | " + "必须审批通过并保留凭证。" * 30 + " | 500元",
        type="table_row", hard_boundary=True,
    )
    try:
        chunks = chunk_document(ir(table), TopicEmbedding(), ChunkConfig(target_tokens=32, max_tokens=64, overlap_tokens=0))
    except AppError as error:
        # Explicitly rejecting a row that cannot be safely represented is preferable to false evidence.
        assert error.code in {"TABLE_ROW_TOO_LARGE", "TABLE_TOO_LARGE"}
        return
    amount_chunks = [chunk for chunk in chunks if "500元" in chunk.text]
    assert amount_chunks
    assert all("上海" in chunk.text and "必须审批通过" in chunk.text for chunk in amount_chunks)


@pytest.mark.parametrize(
    "settings",
    [
        {"target_tokens": 100, "max_tokens": 64},
        {"target_tokens": 80, "overlap_tokens": 80},
        {"semantic_threshold": math.nan},
    ],
)
def test_invalid_chunk_config_fails_before_processing(settings):
    with pytest.raises(ValidationError):
        ChunkConfig(**settings)
