"""Sentence vectors + semantic/structural boundaries + bounded, source-mapped chunks."""

import re
import numpy as np
from xiaoying.domain import AppError, Chunk, ChunkConfig, DocumentIR


def split_to_fit(text, count, limit):
    """Return character ranges without deleting punctuation, whitespace or tail content."""
    start = 0
    while start < len(text):
        low, high = 1, len(text) - start
        if count(text[start:]) <= limit:
            yield start, len(text)
            return
        while low < high:
            mid = (low + high + 1) // 2
            if count(text[start : start + mid]) <= limit:
                low = mid
            else:
                high = mid - 1
        if count(text[start : start + low]) > limit:
            raise AppError("TOKEN_LIMIT", "单个字符超过 tokenizer 限制", 422)
        yield start, start + low
        start += low


def _chunk_linear(ir: DocumentIR, embedder, config: ChunkConfig) -> list[Chunk]:
    units = []
    for block in ir.blocks:
        # Complete short clauses/items remain whole; prose is split at sentence punctuation.
        ranges = (
            [(0, len(block.text))]
            if block.hard_boundary and embedder.count_tokens(block.text) <= config.max_tokens
            else [
                (m.start(), m.end())
                for m in re.finditer(r"[^。！？!?\n]+[。！？!?\n]*|[。！？!?\n]+", block.text)
            ]
        )
        for start, end in ranges:
            for lo, hi in split_to_fit(block.text[start:end], embedder.count_tokens, config.max_tokens):
                units.append(
                    {
                        "block": block,
                        "start": start + lo,
                        "end": start + hi,
                        "text": block.text[start + lo : start + hi],
                    }
                )
    if not units:
        return []
    vectors = np.asarray(embedder.encode([u["text"] for u in units]), dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(units) or not np.isfinite(vectors).all():
        raise AppError("EMBEDDING_INVALID", "句段向量无效", 503)
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / np.maximum(norms[:, None], 1e-12)
    result, current = [], []

    def text_of(group):
        return "".join(
            ("\n\n" if i and group[i - 1]["block"].block_id != u["block"].block_id else "") + u["text"]
            for i, u in enumerate(group)
        )

    def emit(group):
        if not group:
            return
        text = text_of(group)
        ranges = []
        for u in group:
            r = {"block_id": u["block"].block_id, "start": u["start"], "end": u["end"]}
            if ranges and ranges[-1]["block_id"] == r["block_id"] and ranges[-1]["end"] == r["start"]:
                ranges[-1]["end"] = r["end"]
            else:
                ranges.append(r)
        result.append(
            Chunk(
                text=text,
                source_ranges=ranges,
                heading_path=group[0]["block"].heading_path,
                token_count=embedder.count_tokens(text),
            )
        )

    for i, u in enumerate(units):
        boundary = False
        if current:
            prev = current[-1]
            a, b = prev["block"], u["block"]
            structural = a.heading_path != b.heading_path or (
                a.block_id != b.block_id and (a.hard_boundary or b.hard_boundary)
            )
            semantic = float(normalized[i - 1] @ normalized[i]) < config.semantic_threshold
            too_long = embedder.count_tokens(text_of(current + [u])) > config.max_tokens
            target = embedder.count_tokens(text_of(current)) >= config.target_tokens
            boundary = structural or semantic
            if boundary or too_long or target:
                emit(current)
                overlap = []
                # Overlap only at size boundaries in the same structure/topic; never across semantic changes.
                if not boundary and config.overlap_tokens:
                    for old in reversed(current):
                        proposed = [old] + overlap
                        if embedder.count_tokens(text_of(proposed)) > config.overlap_tokens:
                            break
                        overlap = proposed
                    if embedder.count_tokens(text_of(overlap + [u])) > config.max_tokens:
                        overlap = []
                current = overlap
        current.append(u)
    emit(current)
    if any(c.token_count > config.max_tokens for c in result):
        raise AppError("CHUNK_TOO_LARGE", "分块超过硬上限", 422)
    return result


def chunk_document(ir: DocumentIR, embedder, config: ChunkConfig) -> list[Chunk]:
    result, prose = [], []

    def flush():
        if prose:
            result.extend(_chunk_linear(ir.model_copy(update={"blocks": list(prose)}), embedder, config))
            prose.clear()

    for block in ir.blocks:
        if block.type not in {"table", "table_row"} or embedder.count_tokens(block.text) <= config.max_tokens:
            prose.append(block)
            continue
        flush()
        lines = block.text.splitlines(keepends=True)
        if len(lines) < 2:
            raise AppError("TABLE_TOO_LARGE", "超长表格缺少可靠表头/行结构，不能安全分块", 422)
        header_len = len(lines[0])
        if block.type == "table" and len(lines) > 2 and re.fullmatch(r"[\s|:\-]+", lines[1]):
            header_len += len(lines[1])
        header = block.text[:header_len]
        budget = config.max_tokens - embedder.count_tokens(header) - 4
        if budget < 1:
            raise AppError("TABLE_HEADER_TOO_LARGE", "表头超过分块上限，请调整分块配置", 422)
        body = block.text[header_len:]
        cursor = 0
        for row in body.splitlines(keepends=True):
            if embedder.count_tokens(header + row) > config.max_tokens:
                raise AppError(
                    "TABLE_ROW_TOO_LARGE",
                    "表格单行及表头超过分块上限，不能拆散金额与条件；请增大分块上限或整理表格",
                    422,
                )
            result.append(
                Chunk(
                    text=header + row,
                    heading_path=block.heading_path,
                    token_count=embedder.count_tokens(header + row),
                    source_ranges=[
                        {"block_id": block.block_id, "start": 0, "end": header_len},
                        {
                            "block_id": block.block_id,
                            "start": header_len + cursor,
                            "end": header_len + cursor + len(row),
                        },
                    ],
                )
            )
            cursor += len(row)
    flush()
    return result
