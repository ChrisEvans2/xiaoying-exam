"""Guardrails around diagnostic/offline modes and lease configuration."""

import pytest
from pydantic import ValidationError

from psycopg.conninfo import conninfo_to_dict

from xiaoying.config import Settings, get_test_database_url, load_settings


def test_live_mode_cannot_silently_use_test_pdf_parser():
    with pytest.raises(ValidationError, match="only permitted"):
        Settings(_env_file=None, provider_mode="live", pdf_backend="test_text")


def test_offline_pdf_parser_requires_explicit_mock_mode():
    config = Settings(_env_file=None, provider_mode="mock", pdf_backend="test_text")
    assert config.provider_mode == "mock"


def test_heartbeat_cannot_exhaust_lease_margin():
    with pytest.raises(ValidationError, match="heartbeat"):
        Settings(_env_file=None, heartbeat_seconds=30, lease_seconds=60)


def test_recorded_run_config_has_no_api_secret():
    secret = "review-only-do-not-log"
    config = Settings(_env_file=None, deepseek_api_key=secret)
    assert secret not in str(config.run_config())


@pytest.mark.parametrize("field", ["embedding_revision", "reranker_revision"])
def test_live_mode_rejects_mutable_model_revision(field):
    with pytest.raises(ValidationError, match="immutable"):
        Settings(_env_file=None, provider_mode="live", **{field: "main"})


def test_default_business_profile_is_live(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XIAOYING_PROFILE", raising=False)
    assert load_settings().provider_mode == "live"


def test_business_entrypoint_rejects_legacy_mock_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PROVIDER_MODE=mock\nPDF_BACKEND=test_text\n")
    with pytest.raises(ValueError, match="--mock"):
        load_settings(profile="live")


def test_mock_profile_ignores_business_dotenv_and_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=dotenv-business-secret\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost:55432/business")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "business-files"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-business-secret")
    monkeypatch.setenv("PROVIDER_MODE", "live")
    monkeypatch.setenv("PDF_BACKEND", "mineru")
    config = load_settings(profile="mock")
    assert conninfo_to_dict(config.database_url)["dbname"] == "xiaoying_mock"
    assert config.data_dir.resolve() == tmp_path / ".runtime/mock/files"
    assert config.provider_mode == "mock" and config.pdf_backend == "test_text"
    assert config.deepseek_api_key == ""


@pytest.mark.parametrize("via_file", [False, True])
def test_mock_profile_rejects_business_database_even_through_host_alias(monkeypatch, tmp_path, via_file):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:business-secret@localhost:55432/business")
    url = "postgresql://user:mock-secret@127.0.0.1:55432/business"
    if via_file:
        (tmp_path / ".env.mock").write_text(f"XIAOYING_MOCK_DATABASE_URL={url}\n")
    else:
        monkeypatch.setenv("XIAOYING_MOCK_DATABASE_URL", url)
    with pytest.raises(ValueError, match="独立数据库") as error:
        load_settings(profile="mock")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("relative", ["files", "files/mock", "."])
def test_mock_profile_rejects_overlapping_file_directories(monkeypatch, tmp_path, relative):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("XIAOYING_MOCK_DATA_DIR", str(tmp_path / relative))
    with pytest.raises(ValueError, match="文件目录"):
        load_settings(profile="mock")


def test_automated_test_database_is_separate_from_live_and_mock(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XIAOYING_TEST_DATABASE_URL", raising=False)
    assert conninfo_to_dict(get_test_database_url())["dbname"] == "xiaoying_test"
    for name in ("xiaoying", "xiaoying_mock"):
        monkeypatch.setenv("XIAOYING_TEST_DATABASE_URL", f"postgresql://user:secret@localhost/{name}")
        with pytest.raises(ValueError, match="独立数据库"):
            get_test_database_url()
