"""Question-set validation, batch execution, and export."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from .evidence import Evidence
from .projects import ProjectError, ProjectStore
from .querying import QueryResult, QueryService


QUESTION_STATUSES = {"PENDING", "RUNNING", "COMPLETED", "FAILED"}


class QuestionRunner(Protocol):
    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult: ...


@dataclass(frozen=True, slots=True)
class GoldEvidence:
    document_id: str
    pages: tuple[int, ...]
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchQuestion:
    question_id: str
    question: str
    reference_answer: str = ""
    status: str = "PENDING"
    answer: str = ""
    error: str | None = None
    duration_seconds: float | None = None
    completed_at: str | None = None
    gold_evidence: tuple[GoldEvidence, ...] = ()
    retrieved_evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionSet:
    question_set_id: str
    project_id: str
    name: str
    description: str
    method: str
    imported_at: str
    updated_at: str
    questions: tuple[BatchQuestion, ...]


@dataclass(frozen=True, slots=True)
class BatchSummary:
    total: int
    completed: int
    failed: int
    pending: int
    running: int


class QuestionSetService:
    def __init__(self, projects: ProjectStore, query_service: QuestionRunner | None = None) -> None:
        self.projects = projects
        self.query_service = query_service or QueryService(projects)

    def import_file(self, project_id: str, source: str | Path) -> QuestionSet:
        self.projects.get(project_id)
        path = Path(source)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError("找不到題目集檔案") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError(f"JSON 格式錯誤：第 {exc.lineno} 行第 {exc.colno} 欄，{exc.msg}") from exc
        name, description, questions = self._validate(value)
        now = datetime.now(timezone.utc).isoformat()
        question_set = QuestionSet(
            question_set_id=uuid.uuid4().hex,
            project_id=project_id,
            name=name,
            description=description,
            method="local",
            imported_at=now,
            updated_at=now,
            questions=tuple(questions),
        )
        self._write(question_set)
        return question_set

    def create(self, project_id: str, name: str, description: str = "") -> QuestionSet:
        self.projects.get(project_id)
        if not isinstance(name, str) or not name.strip():
            raise ProjectError("題目集名稱不可空白")
        if not isinstance(description, str):
            raise ProjectError("題目集說明格式錯誤")
        now = datetime.now(timezone.utc).isoformat()
        question_set = QuestionSet(
            question_set_id=uuid.uuid4().hex,
            project_id=project_id,
            name=name.strip(),
            description=description.strip(),
            method="local",
            imported_at=now,
            updated_at=now,
            questions=(),
        )
        self._write(question_set)
        return question_set

    def append_answered_question(
        self,
        project_id: str,
        question_set_id: str,
        question: str,
        answer: str,
        method: str = "local",
    ) -> QuestionSet:
        if not isinstance(question, str) or not question.strip():
            raise ProjectError("問題不可空白")
        if not isinstance(answer, str) or not answer.strip():
            raise ProjectError("系統回答不可空白，請先完成問答再加入題目集")
        question_set = self.get(project_id, question_set_id)
        if any(item.question.strip() == question.strip() for item in question_set.questions):
            raise ProjectError("題目集已存在相同問題")
        used_ids = {item.question_id for item in question_set.questions}
        used_numbers = {
            int(item_id[1:])
            for item_id in used_ids
            if item_id.startswith("Q") and item_id[1:].isdigit()
        }
        next_number = max(used_numbers, default=0) + 1
        while f"Q{next_number:04d}" in used_ids:
            next_number += 1
        now = datetime.now(timezone.utc).isoformat()
        answered = BatchQuestion(
            question_id=f"Q{next_number:04d}",
            question=question.strip(),
            reference_answer=answer.strip(),
            status="COMPLETED",
            answer=answer.strip(),
            completed_at=now,
        )
        updated = self._replace(question_set, [*question_set.questions, answered], method)
        self._write(updated)
        return updated

    def list(self, project_id: str) -> list[QuestionSet]:
        directory = self.projects.path_for(project_id) / "question_sets"
        result = [self._read(path) for path in directory.glob("*.json") if path.is_file()]
        return sorted(result, key=lambda item: item.imported_at, reverse=True)

    def get(self, project_id: str, question_set_id: str) -> QuestionSet:
        if not question_set_id or not question_set_id.isalnum():
            raise ProjectError("題目集 ID 格式不正確")
        path = self.projects.path_for(project_id) / "question_sets" / f"{question_set_id}.json"
        try:
            return self._read(path)
        except FileNotFoundError as exc:
            raise ProjectError(f"找不到題目集：{question_set_id}") from exc

    def run(
        self,
        project_id: str,
        question_set_id: str,
        method: str = "local",
        selected_question_ids: Iterable[str] | None = None,
        resume: bool = False,
    ) -> QuestionSet:
        project = self.projects.get(project_id)
        if project.status != "INDEXED":
            raise ProjectError(f"專案狀態 {project.status} 尚未完成建圖，無法批次查詢")
        question_set = self.get(project_id, question_set_id)
        selected = set(selected_question_ids or [])
        known_ids = {item.question_id for item in question_set.questions}
        unknown = selected - known_ids
        if unknown:
            raise ProjectError(f"題目集找不到題號：{', '.join(sorted(unknown))}")

        if resume:
            targets = {item.question_id for item in question_set.questions if item.status in {"PENDING", "RUNNING"}}
        elif selected:
            targets = selected
        else:
            targets = known_ids

        questions = list(question_set.questions)
        if not resume:
            questions = [
                BatchQuestion(
                    item.question_id,
                    item.question,
                    reference_answer=item.reference_answer,
                    gold_evidence=item.gold_evidence,
                )
                if item.question_id in targets
                else item
                for item in questions
            ]
        current = self._replace(question_set, questions, method)
        self._write(current)

        for index, item in enumerate(questions):
            if item.question_id not in targets:
                continue
            questions[index] = BatchQuestion(
                item.question_id,
                item.question,
                reference_answer=item.reference_answer,
                status="RUNNING",
                gold_evidence=item.gold_evidence,
            )
            current = self._replace(current, questions, method)
            self._write(current)
            try:
                result = self.query_service.ask(project_id, item.question, method)
                questions[index] = BatchQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    status=result.status,
                    answer=result.answer,
                    error=result.error,
                    duration_seconds=result.duration_seconds,
                    completed_at=result.completed_at,
                    gold_evidence=item.gold_evidence,
                    retrieved_evidence=result.evidence,
                )
            except Exception as exc:
                questions[index] = BatchQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    status="FAILED",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    gold_evidence=item.gold_evidence,
                )
            current = self._replace(current, questions, method)
            self._write(current)
        return current

    def summary(self, question_set: QuestionSet) -> BatchSummary:
        counts = {status: 0 for status in QUESTION_STATUSES}
        for item in question_set.questions:
            counts[item.status] += 1
        return BatchSummary(
            total=len(question_set.questions),
            completed=counts["COMPLETED"],
            failed=counts["FAILED"],
            pending=counts["PENDING"],
            running=counts["RUNNING"],
        )

    def export(self, project_id: str, question_set_id: str) -> tuple[Path, Path]:
        question_set = self.get(project_id, question_set_id)
        export_directory = self.projects.path_for(project_id) / "exports"
        json_path = export_directory / f"{question_set_id}.json"
        csv_path = export_directory / f"{question_set_id}.csv"
        self._atomic_text(json_path, json.dumps(self._to_dict(question_set), ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "question_id",
                "question",
                "reference_answer",
                "status",
                "answer",
                "error",
                "duration_seconds",
                "completed_at",
                "gold_evidence",
                "retrieved_evidence",
            ],
        )
        writer.writeheader()
        for item in question_set.questions:
            row = asdict(item)
            row["gold_evidence"] = json.dumps(row["gold_evidence"], ensure_ascii=False)
            row["retrieved_evidence"] = json.dumps(row["retrieved_evidence"], ensure_ascii=False)
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    def update_gold_evidence(
        self,
        project_id: str,
        question_set_id: str,
        question_id: str,
        gold_evidence: tuple[GoldEvidence, ...],
    ) -> QuestionSet:
        question_set = self.get(project_id, question_set_id)
        found = False
        questions: list[BatchQuestion] = []
        for item in question_set.questions:
            if item.question_id == question_id:
                found = True
                value = asdict(item)
                value["gold_evidence"] = gold_evidence
                value["retrieved_evidence"] = item.retrieved_evidence
                questions.append(BatchQuestion(**value))
            else:
                questions.append(item)
        if not found:
            raise ProjectError(f"題目集找不到題號：{question_id}")
        updated = self._replace(question_set, questions, question_set.method)
        self._write(updated)
        return updated

    @staticmethod
    def _validate(value: object) -> tuple[str, str, list[BatchQuestion]]:
        if not isinstance(value, dict):
            raise ProjectError("題目集根節點必須是 JSON object")
        errors: list[str] = []
        name = value.get("name")
        description = value.get("description", "")
        raw_questions = value.get("questions")
        if not isinstance(name, str) or not name.strip():
            errors.append("name：必須是非空白字串")
        if not isinstance(description, str):
            errors.append("description：必須是字串")
        if not isinstance(raw_questions, list) or not raw_questions:
            errors.append("questions：必須是至少包含一題的陣列")
            raw_questions = []

        questions: list[BatchQuestion] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_questions):
            location = f"questions[{index}]"
            if not isinstance(raw, dict):
                errors.append(f"{location}：必須是 object")
                continue
            question_id = raw.get("question_id")
            question = raw.get("question")
            reference_answer = raw.get("reference_answer", "")
            try:
                gold_evidence = QuestionSetService.parse_gold_evidence(raw.get("gold_evidence", []), location)
            except ProjectError as exc:
                errors.extend(str(exc).splitlines())
                gold_evidence = ()
            if not isinstance(question_id, str) or not question_id.strip():
                errors.append(f"{location}.question_id：必須是非空白字串")
            elif question_id.strip() in seen:
                errors.append(f"{location}.question_id：重複題號 {question_id.strip()}")
            else:
                seen.add(question_id.strip())
            if not isinstance(question, str) or not question.strip():
                errors.append(f"{location}.question：必須是非空白字串")
            if not isinstance(reference_answer, str):
                errors.append(f"{location}.reference_answer：必須是字串")
            if (
                isinstance(question_id, str)
                and question_id.strip()
                and isinstance(question, str)
                and question.strip()
                and isinstance(reference_answer, str)
            ):
                questions.append(
                    BatchQuestion(
                        question_id.strip(),
                        question.strip(),
                        reference_answer=reference_answer.strip(),
                        gold_evidence=gold_evidence,
                    )
                )
        if errors:
            raise ProjectError("題目集格式錯誤：\n- " + "\n- ".join(errors))
        return name.strip(), description.strip(), questions

    @staticmethod
    def parse_gold_evidence(value: object, location: str = "gold_evidence") -> tuple[GoldEvidence, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ProjectError(f"{location}.gold_evidence：必須是陣列")
        result: list[GoldEvidence] = []
        errors: list[str] = []
        for index, raw in enumerate(value):
            item_location = f"{location}.gold_evidence[{index}]"
            if not isinstance(raw, dict):
                errors.append(f"{item_location}：必須是 object")
                continue
            document_id = raw.get("document_id")
            pages = raw.get("pages", [])
            chunk_ids = raw.get("chunk_ids", [])
            if not isinstance(document_id, str) or not document_id.strip():
                errors.append(f"{item_location}.document_id：必須是非空白字串")
            if not isinstance(pages, list) or any(not isinstance(page, int) or page < 1 for page in pages):
                errors.append(f"{item_location}.pages：必須是正整數陣列")
                pages = []
            if not isinstance(chunk_ids, list) or any(
                not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in chunk_ids
            ):
                errors.append(f"{item_location}.chunk_ids：必須是非空白字串陣列")
                chunk_ids = []
            if not pages and not chunk_ids:
                errors.append(f"{item_location}：pages 與 chunk_ids 至少需要一項")
            if isinstance(document_id, str) and document_id.strip() and (pages or chunk_ids):
                result.append(
                    GoldEvidence(
                        document_id=document_id.strip(),
                        pages=tuple(dict.fromkeys(pages)),
                        chunk_ids=tuple(dict.fromkeys(chunk_id.strip() for chunk_id in chunk_ids)),
                    )
                )
        if errors:
            raise ProjectError("\n".join(errors))
        return tuple(result)

    def _write(self, question_set: QuestionSet) -> None:
        path = self.projects.path_for(question_set.project_id) / "question_sets" / f"{question_set.question_set_id}.json"
        self._atomic_text(path, json.dumps(self._to_dict(question_set), ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _read(path: Path) -> QuestionSet:
        value = json.loads(path.read_text(encoding="utf-8"))
        questions = []
        for item in value.pop("questions"):
            item["gold_evidence"] = tuple(
                GoldEvidence(
                    document_id=gold["document_id"],
                    pages=tuple(gold.get("pages", [])),
                    chunk_ids=tuple(gold.get("chunk_ids", [])),
                )
                for gold in item.get("gold_evidence", [])
            )
            item["retrieved_evidence"] = tuple(
                Evidence(**evidence) for evidence in item.get("retrieved_evidence", [])
            )
            questions.append(BatchQuestion(**item))
        return QuestionSet(**value, questions=tuple(questions))

    @staticmethod
    def _to_dict(question_set: QuestionSet) -> dict[str, object]:
        value = asdict(question_set)
        value["questions"] = [asdict(item) for item in question_set.questions]
        return value

    @staticmethod
    def _replace(question_set: QuestionSet, questions: list[BatchQuestion], method: str) -> QuestionSet:
        return QuestionSet(
            question_set_id=question_set.question_set_id,
            project_id=question_set.project_id,
            name=question_set.name,
            description=question_set.description,
            method=method,
            imported_at=question_set.imported_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=tuple(questions),
        )

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
