"""Validate and edit human-labelled Gold Evidence."""

from __future__ import annotations

from .projects import ProjectError, ProjectStore
from .question_sets import GoldEvidence, QuestionSet, QuestionSetService
from .source_metadata import SourceMetadata, SourceMetadataService


class GroundTruthService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService | None = None,
        source_metadata: SourceMetadataService | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)
        self.source_metadata = source_metadata or SourceMetadataService(projects)

    def available_sources(self, project_id: str) -> list[SourceMetadata]:
        return self.source_metadata.load(project_id)

    def save(
        self,
        project_id: str,
        question_set_id: str,
        question_id: str,
        value: object,
    ) -> QuestionSet:
        gold_evidence = self.question_sets.parse_gold_evidence(value, "gold_evidence")
        available = self.available_sources(project_id)
        pages_by_document: dict[str, set[int]] = {}
        chunks_by_document: dict[str, set[str]] = {}
        for source in available:
            pages_by_document.setdefault(source.document_id, set()).add(source.page)
            chunks_by_document.setdefault(source.document_id, set()).add(source.chunk_id)
        errors: list[str] = []
        for item in gold_evidence:
            if item.document_id not in pages_by_document:
                errors.append(f"找不到 PDF：{item.document_id}")
                continue
            missing_pages = set(item.pages) - pages_by_document[item.document_id]
            missing_chunks = set(item.chunk_ids) - chunks_by_document[item.document_id]
            if missing_pages:
                errors.append(f"{item.document_id} 找不到頁碼：{', '.join(map(str, sorted(missing_pages)))}")
            if missing_chunks:
                errors.append(f"{item.document_id} 找不到 Chunk：{', '.join(sorted(missing_chunks))}")
        if errors:
            raise ProjectError("Gold Evidence 驗證失敗：\n- " + "\n- ".join(errors))
        return self.question_sets.update_gold_evidence(
            project_id,
            question_set_id,
            question_id,
            gold_evidence,
        )
