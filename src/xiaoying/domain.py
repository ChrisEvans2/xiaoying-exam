import hashlib
import json
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)


class LeaseLost(AppError):
    def __init__(self):
        super().__init__("LEASE_LOST", "任务执行权已失效", 409)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


class ChunkConfig(BaseModel):
    target_tokens: int = Field(400, ge=16, le=4096)
    max_tokens: int = Field(800, ge=32, le=8192)
    overlap_tokens: int = Field(80, ge=0, le=1024)
    semantic_threshold: float = Field(0.62, ge=-1, le=1)

    @model_validator(mode="after")
    def check_sizes(self):
        if not self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("require overlap < target <= maximum")
        return self


class Block(BaseModel):
    block_id: str
    text: str
    type: str = "paragraph"
    heading_path: list[str] = []
    order: int = 0
    locator: dict = {}
    hard_boundary: bool = False


class DocumentIR(BaseModel):
    metadata: dict
    blocks: list[Block]
    parser_version: str
    warnings: list[str] = []


class Chunk(BaseModel):
    text: str
    source_ranges: list[dict]
    heading_path: list[str]
    token_count: int


class BuildChange(BaseModel):
    action: Literal["add", "replace", "remove"]
    document_id: str = Field(min_length=1, max_length=120, pattern=r"^[\w.-]+$")
    upload_id: str | None = None


class BuildRequest(BaseModel):
    base_version_id: str | None = None
    changes: list[BuildChange] = Field(default_factory=list, max_length=100)
    config: ChunkConfig = Field(default_factory=ChunkConfig)
    auto_publish: bool = True

    @model_validator(mode="after")
    def changes_valid(self):
        ids = [c.document_id for c in self.changes]
        if len(set(ids)) != len(ids):
            raise ValueError("a document can only occur once in a build")
        if any(c.action != "remove" and not c.upload_id for c in self.changes):
            raise ValueError("add/replace requires upload_id")
        return self
