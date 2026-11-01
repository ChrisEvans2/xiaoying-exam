import os
import re
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from psycopg.conninfo import conninfo_to_dict, make_conninfo


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://xiaoying:xiaoying@localhost:55432/xiaoying"
    data_dir: Path = Path(".runtime/files")
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_thinking: bool = True
    deepseek_reasoning_effort: Literal["low", "high", "max"] = "high"
    deepseek_max_tokens: int = Field(16384, ge=1, le=393216)
    provider_mode: str = "live"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    mineru_command: list[str] = ["mineru", "-p", "{input}", "-o", "{output}", "-b", "pipeline"]
    pdf_backend: str = "mineru"
    model_timeout_seconds: float = Field(45, gt=0)
    tool_timeout_seconds: float = Field(20, gt=0)
    read_timeout_seconds: float = Field(5, gt=0)
    active_timeout_seconds: float = Field(180, gt=0)
    clarification_seconds: float = Field(600, gt=0)
    queue_timeout_seconds: float = Field(60, gt=0)
    lease_seconds: float = Field(60, gt=0)
    heartbeat_seconds: float = Field(10, gt=0)
    scan_seconds: float = Field(5, gt=0)
    max_model_calls: int = Field(12, ge=1, le=100)
    max_tool_calls: int = Field(8, ge=1, le=100)
    max_graph_steps: int = Field(40, ge=1, le=500)
    max_upload_bytes: int = Field(20 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(200, gt=0)
    build_timeout_seconds: float = Field(1800, gt=0)
    pdf_timeout_seconds: float = Field(600, gt=0)

    @model_validator(mode="after")
    def validate_modes(self):
        if self.provider_mode not in {"live", "mock"}:
            raise ValueError("PROVIDER_MODE must be live or mock")
        if self.pdf_backend not in {"mineru", "test_text"}:
            raise ValueError("PDF_BACKEND must be mineru or test_text")
        if self.pdf_backend == "test_text" and self.provider_mode != "mock":
            raise ValueError("test_text PDF backend is only permitted in explicit mock mode")
        if self.heartbeat_seconds >= self.lease_seconds / 2:
            raise ValueError("heartbeat must be less than half of lease")
        if self.provider_mode == "live" and any(
            not re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in (self.embedding_revision, self.reranker_revision)
        ):
            raise ValueError("Live model revisions must be immutable 40-character commit hashes")
        return self

    def run_config(self):
        return self.model_dump(
            mode="json",
            include={
                "provider_mode",
                "deepseek_model",
                "deepseek_thinking",
                "deepseek_reasoning_effort",
                "deepseek_max_tokens",
                "model_timeout_seconds",
                "tool_timeout_seconds",
                "read_timeout_seconds",
                "active_timeout_seconds",
                "clarification_seconds",
                "queue_timeout_seconds",
                "max_model_calls",
                "max_tool_calls",
                "max_graph_steps",
                "reranker_model",
                "reranker_revision",
            },
        )


class MockSettings(Settings):
    """Explicit offline profile; business .env and unprefixed variables are never inherited."""

    model_config = SettingsConfigDict(env_file=".env.mock", env_prefix="XIAOYING_MOCK_", extra="ignore")
    database_url: str = "postgresql://xiaoying:xiaoying@localhost:55432/xiaoying_mock"
    data_dir: Path = Path(".runtime/mock/files")
    provider_mode: str = "mock"
    pdf_backend: str = "test_text"


def require_separate_database(isolated_url, business_url):
    # Require distinct database names even when hosts differ, avoiding localhost aliases
    # and tunnel addresses accidentally bypassing the separation check.
    isolated = conninfo_to_dict(isolated_url).get("dbname")
    business = conninfo_to_dict(business_url).get("dbname")
    if not isolated or isolated == business:
        raise ValueError("测试/mock 必须使用独立数据库，不能指向业务数据库")


def load_settings(profile=None):
    profile = profile or os.getenv("XIAOYING_PROFILE", "live")
    if profile not in {"live", "mock"}:
        raise ValueError("XIAOYING_PROFILE must be live or mock")
    business = Settings()
    if profile == "live":
        if business.provider_mode != "live":
            raise ValueError("业务入口要求 PROVIDER_MODE=live；离线演示请显式使用 --mock")
        return business
    offline = MockSettings()
    if offline.provider_mode != "mock":
        raise ValueError("mock 环境只能使用 mock provider")
    require_separate_database(offline.database_url, business.database_url)
    live_files, mock_files = business.data_dir.resolve(), offline.data_dir.resolve()
    if live_files.is_relative_to(mock_files) or mock_files.is_relative_to(live_files):
        raise ValueError("mock 文件目录必须与业务文件目录分开，不能相同或互相包含")
    return offline


def get_test_database_url():
    business = Settings()
    url = os.getenv(
        "XIAOYING_TEST_DATABASE_URL", make_conninfo(business.database_url, dbname="xiaoying_test")
    )
    require_separate_database(url, business.database_url)
    require_separate_database(url, MockSettings().database_url)
    return url
