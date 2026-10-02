"""Human review persistence and export for batch answers."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .projects import ProjectError, ProjectStore
from .question_sets import QuestionSetService


HUMAN_LABELS = {"correct", "partially_correct", "incorrect", "insufficient"}


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    question_id: str
    question: str
    answer: str
    answer_status: str
    human_label: str | None = None
    reviewer_note: str = ""
    reviewed_at: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    reviewed: int
    total: int


class ReviewService:
    def __init__(self, projects: ProjectStore, question_sets: QuestionSetService | None = None) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)

    def records(self, project_id: str, question_set_id: str) -> list[ReviewRecord]:
        question_set = self.question_sets.get(project_id, question_set_id)
        saved = self._read_saved(project_id, question_set_id)
        return [
            ReviewRecord(
                question_id=item.question_id,
                question=item.question,
                answer=item.answer,
                answer_status=item.status,
                human_label=saved.get(item.question_id, {}).get("human_label"),
                reviewer_note=saved.get(item.question_id, {}).get("reviewer_note", ""),
                reviewed_at=saved.get(item.question_id, {}).get("reviewed_at"),
            )
            for item in question_set.questions
        ]

    def save(
        self,
        project_id: str,
        question_set_id: str,
        question_id: str,
        human_label: str,
        reviewer_note: str = "",
    ) -> ReviewRecord:
        if human_label not in HUMAN_LABELS:
            raise ProjectError("請選擇有效的人工正確性標籤")
        records = self.records(project_id, question_set_id)
        record_by_id = {item.question_id: item for item in records}
        if question_id not in record_by_id:
            raise ProjectError(f"題目集找不到題號：{question_id}")
        now = datetime.now(timezone.utc).isoformat()
        saved = self._read_saved(project_id, question_set_id)
        saved[question_id] = {
            "human_label": human_label,
            "reviewer_note": reviewer_note.strip(),
            "reviewed_at": now,
        }
        self._write_saved(project_id, question_set_id, saved)
        original = record_by_id[question_id]
        return ReviewRecord(
            question_id=original.question_id,
            question=original.question,
            answer=original.answer,
            answer_status=original.answer_status,
            human_label=human_label,
            reviewer_note=reviewer_note.strip(),
            reviewed_at=now,
        )

    def progress(self, project_id: str, question_set_id: str) -> ReviewProgress:
        records = self.records(project_id, question_set_id)
        return ReviewProgress(sum(item.human_label is not None for item in records), len(records))

    def export(self, project_id: str, question_set_id: str) -> tuple[Path, Path]:
        question_set = self.question_sets.get(project_id, question_set_id)
        records = self.records(project_id, question_set_id)
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / f"{question_set_id}-human-review.json"
        csv_path = directory / f"{question_set_id}-human-review.csv"
        payload = {
            "question_set_id": question_set_id,
            "project_id": project_id,
            "name": question_set.name,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "reviews": [asdict(record) for record in records],
        }
        self._atomic_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(ReviewRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    def _review_path(self, project_id: str, question_set_id: str) -> Path:
        self.question_sets.get(project_id, question_set_id)
        return self.projects.path_for(project_id) / "runs" / f"reviews-{question_set_id}.json"

    def _read_saved(self, project_id: str, question_set_id: str) -> dict[str, dict[str, str]]:
        path = self._review_path(project_id, question_set_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ProjectError("人工評測紀錄格式錯誤") from exc
        reviews = value.get("reviews")
        if not isinstance(reviews, dict):
            raise ProjectError("人工評測紀錄格式錯誤")
        return reviews

    def _write_saved(self, project_id: str, question_set_id: str, reviews: dict[str, dict[str, str]]) -> None:
        path = self._review_path(project_id, question_set_id)
        payload = {"project_id": project_id, "question_set_id": question_set_id, "reviews": reviews}
        self._atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
