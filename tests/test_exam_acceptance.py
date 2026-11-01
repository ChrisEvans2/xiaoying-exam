"""Safety boundaries of the real-model evaluation entrypoint; no API calls."""

import hashlib

import pytest
from psycopg.conninfo import conninfo_to_dict

from scripts.exam_acceptance import FaultSearch, freeze_sources, isolated_settings
from xiaoying.config import Settings


def test_acceptance_uses_separate_database_and_preserves_live_configuration(tmp_path):
    live = Settings(_env_file=None, data_dir=tmp_path / "business", deepseek_api_key="test-secret")
    isolated = isolated_settings(live, "xiaoying_acceptance_test", tmp_path / "evaluation")
    assert conninfo_to_dict(isolated.database_url)["dbname"] == "xiaoying_acceptance_test"
    assert isolated.provider_mode == "live" and isolated.pdf_backend == "mineru"
    assert isolated.deepseek_api_key == live.deepseek_api_key
    assert live.data_dir == tmp_path / "business"
    assert conninfo_to_dict(live.database_url)["dbname"] != "xiaoying_acceptance_test"


@pytest.mark.parametrize("directory", ["business", "business/child", "."])
def test_acceptance_rejects_file_directory_overlap(tmp_path, directory):
    live = Settings(_env_file=None, data_dir=tmp_path / "business")
    with pytest.raises(ValueError, match="文件目录"):
        isolated_settings(live, "xiaoying_acceptance_test", tmp_path / directory)


@pytest.mark.parametrize("name", ["xiaoying", "xiaoying_mock", "postgres", "xiaoying_acceptance_test;DROP"])
def test_acceptance_rejects_non_evaluation_database_names(tmp_path, name):
    with pytest.raises(ValueError, match="数据库名"):
        isolated_settings(Settings(_env_file=None), name, tmp_path / "isolated")


def test_acceptance_cannot_reuse_business_database_even_with_valid_prefix(tmp_path):
    live = Settings(_env_file=None, database_url="postgresql://localhost/xiaoying_acceptance_existing")
    with pytest.raises(ValueError, match="独立数据库"):
        isolated_settings(live, "xiaoying_acceptance_existing", tmp_path / "isolated")


def test_fault_search_preserves_query_but_only_exposes_target_document():
    calls = []

    class Base:
        def search(self, kb, version, **kwargs):
            calls.append((kb, version, kwargs))
            return {"candidates": []}

    wrapped = FaultSearch(Base())
    result = wrapped.search("kb", "version", queries=["上海住宿"], document_ids=["D01", "D04"])
    assert result == {"candidates": []}
    assert calls == [("kb", "version", {"queries": ["上海住宿"], "document_ids": ["D04"]})]


def test_revision_snapshot_preserves_concurrent_workspace_edits(tmp_path, monkeypatch):
    source = tmp_path / "D07.txt"
    source.write_bytes(b"concurrent user edit")
    baseline = b"known original corpus"
    revision = "a" * 40
    calls = []

    def git_show(command):
        calls.append(command)
        return baseline

    monkeypatch.setattr("scripts.exam_acceptance.subprocess.check_output", git_show)
    dataset = {"source_snapshot": [{"document_id": "D07", "path": str(source),
                                    "sha256": hashlib.sha256(baseline).hexdigest()}]}
    paths = freeze_sources(dataset, tmp_path / "output", revision)
    assert paths["D07"].read_bytes() == baseline
    assert source.read_bytes() == b"concurrent user edit"
    assert calls == [["git", "show", f"{revision}:{source.as_posix()}"]]


def test_snapshot_hash_mismatch_stops_before_accepting_unverified_material(tmp_path):
    source = tmp_path / "D07.txt"
    source.write_bytes(b"changed")
    dataset = {"source_snapshot": [{"document_id": "D07", "path": str(source), "sha256": "0" * 64}]}
    with pytest.raises(RuntimeError, match="哈希不匹配"):
        freeze_sources(dataset, tmp_path / "output", None)
    assert not (tmp_path / "output").exists()
