from typing import Protocol


class CorrectionMemory(Protocol):
    """TODO: reviewed corrections with source provenance; never enabled implicitly."""

    def propose(self, run_id: str, correction: str, evidence_ids: list[str]) -> str: ...


class KnowledgeGraphBuilder(Protocol):
    """TODO: version-scoped, evidence-backed entities and relationships."""

    def build(self, kb_version_id: str) -> str: ...
