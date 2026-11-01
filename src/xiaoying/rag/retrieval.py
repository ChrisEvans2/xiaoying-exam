import json
import math
import time
from xiaoying.domain import AppError
from xiaoying.providers.bounded import bounded
from xiaoying.providers.embeddings import require_compatible_index, terms
from xiaoying.storage.db import require, valid_id


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return scores


class Retriever:
    def __init__(self, db, embedder, reranker, timeout=20):
        self.db, self.embedder, self.reranker, self.timeout = db, embedder, reranker, timeout

    def search(self, kb_id, version, queries, query=None, document_ids=None):
        if (
            not isinstance(queries, list)
            or not 1 <= len(queries) <= 3
            or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in queries)
        ):
            raise AppError("INVALID_QUERY", "需要 1—3 个长度不超过 2000 的查询", 422)
        cfg = require(
            self.db.one(
                "SELECT config FROM kb_versions WHERE id=%s AND kb_id=%s AND status='ready'",
                (valid_id(version), valid_id(kb_id)),
            )
        )["config"]
        require_compatible_index(cfg, self.embedder.identity)
        start = time.monotonic()
        rankings = []
        candidates = {}
        failures = []
        document_ids = document_ids or []
        if (
            not isinstance(document_ids, list)
            or len(document_ids) > 20
            or any(not isinstance(x, str) for x in document_ids)
        ):
            raise AppError("INVALID_FILTER", "无效的文档过滤条件", 422)
        common = """FROM chunks c JOIN document_versions v ON v.id=c.document_version_id
            JOIN documents d ON d.id=v.document_fk
            WHERE c.kb_version_id=%s AND d.kb_id=%s AND (cardinality(%s::text[])=0 OR d.document_id=ANY(%s::text[]))"""
        select = "SELECT c.id,c.document_version_id,c.ordinal,c.text,c.heading_path,d.document_id,v.metadata,v.version_date "
        filters = (version, kb_id, document_ids, document_ids)
        for q in dict.fromkeys(queries):
            for route in ("keyword", "vector"):
                try:
                    remaining = self.timeout - (time.monotonic() - start)
                    if route == "keyword":
                        words = list(dict.fromkeys(terms(q)))[:80]
                        if not words:
                            continue
                        # Each term becomes an independent plain query; tsquery syntax is never interpolated.
                        tsquery = " || ".join("plainto_tsquery('simple',%s)" for _ in words)
                        sql = (
                            select
                            + common
                            + f" AND c.search_vector @@ ({tsquery}) ORDER BY ts_rank_cd(c.search_vector,({tsquery})) DESC,c.id LIMIT 20"
                        )
                        rows = bounded(
                            lambda sql=sql, words=words: self.db.all(
                                sql, filters + tuple(words) + tuple(words)
                            ),
                            remaining,
                        )
                    else:
                        vector = bounded(lambda q=q: self.embedder.encode([q])[0], remaining)
                        if len(vector) != 1024 or not all(math.isfinite(x) for x in vector):
                            raise AppError("EMBEDDING_INVALID", "无效的查询向量", 503)
                        rows = bounded(
                            lambda vector=vector: self.db.all(
                                select + common + " ORDER BY c.embedding <=> %s::vector,c.id LIMIT 20",
                                filters + (json.dumps(vector),),
                            ),
                            self.timeout - (time.monotonic() - start),
                        )
                    rankings.append([r["id"] for r in rows])
                    for r in rows:
                        candidates[r["id"]] = r
                except Exception as e:
                    failures.append({"stage": route, "code": getattr(e, "code", type(e).__name__)})
        if not rankings:
            raise AppError("RETRIEVAL_FAILED", "关键词与向量检索均未完成", 503, True)
        scores = reciprocal_rank_fusion(rankings)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:60]
        rerank_scores = {}
        if ordered:
            try:
                values = bounded(
                    lambda: self.reranker.rank(query or queries[0], [candidates[i]["text"] for i in ordered]),
                    self.timeout - (time.monotonic() - start),
                )
                if len(values) != len(ordered) or not all(math.isfinite(v) for v in values):
                    raise AppError("RERANK_INVALID", "重排分数无效", 503)
                rerank_scores = dict(zip(ordered, values, strict=True))
                ordered.sort(key=lambda i: (-rerank_scores[i], -scores[i], i))
            except Exception as e:
                failures.append({"stage": "rerank", "code": getattr(e, "code", type(e).__name__)})
        return {
            "candidates": [
                {
                    "chunk_id": i,
                    "document_id": candidates[i]["document_id"],
                    "document_version_id": candidates[i]["document_version_id"],
                    "title": candidates[i]["metadata"]["title"],
                    "version_date": candidates[i]["version_date"],
                    "ordinal": candidates[i]["ordinal"],
                    "heading_path": candidates[i]["heading_path"],
                    "rrf_score": scores[i],
                    "rerank_score": rerank_scores.get(i),
                }
                for i in ordered[:8]
            ],
            "degraded": bool(failures),
            "failures": failures,
            "candidate_count": len(candidates),
            "query_count": len(set(queries)),
        }
