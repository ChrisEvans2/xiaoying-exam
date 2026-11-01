"""Live encoders are lazy. Deterministic encoders are explicitly test-only."""

import hashlib
import math
import re
import threading

import jieba
import numpy as np

from xiaoying.domain import AppError

jieba.setLogLevel(30)
for word in ("预借款", "试用期", "差旅", "报销", "网约车", "餐补"):
    jieba.add_word(word)


def terms(text):
    return [t.lower() for t in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", t)][:10000]


def require_compatible_index(config, identity):
    if config.get("embedding_identity") != identity:
        raise AppError(
            "INDEX_MODEL_MISMATCH",
            "该知识库索引与当前模型不兼容。请到“文档与版本”点击“构建并发布新版本”，完成后重新提问。",
            409,
        )


class MockEmbedder:
    identity = "mock-hashed-lexical-v1"

    def count_tokens(self, text):
        return len(text)

    def encode(self, texts):
        result = []
        for text in texts:
            v = [0.0] * 1024
            for token in terms(text):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 1024
                v[index] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1
            result.append([x / norm for x in v])
        return result


class BGEEmbedder:
    def __init__(self, model, revision):
        self.model_name, self.revision = model, revision
        self.identity = f"{model}@{revision}"
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise AppError("MODELS_NOT_INSTALLED", "请执行 uv sync --extra models", 503) from None
                self._model = SentenceTransformer(
                    self.model_name, revision=self.revision, trust_remote_code=False
                )
            return self._model

    def count_tokens(self, text):
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def encode(self, texts):
        if not texts:
            return []
        result = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        if result.shape[1] != 1024 or not np.isfinite(result).all():
            raise AppError("EMBEDDING_INVALID", "向量维度必须为 1024 且数值有限", 503)
        return result.tolist()


class MockReranker:
    def rank(self, query, texts):
        q = set(terms(query))
        return [len(q & set(terms(t))) / max(len(q), 1) for t in texts]


class BGEReranker:
    def __init__(self, model, revision):
        self.model_name, self.revision = model, revision
        self._model = None
        self._lock = threading.Lock()

    def rank(self, query, texts):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, revision=self.revision, trust_remote_code=False)
        return np.asarray(self._model.predict([(query, t) for t in texts], batch_size=8)).reshape(-1).tolist()


def providers(settings):
    if settings.provider_mode == "mock":
        return MockEmbedder(), MockReranker()
    return BGEEmbedder(settings.embedding_model, settings.embedding_revision), BGEReranker(
        settings.reranker_model, settings.reranker_revision
    )
