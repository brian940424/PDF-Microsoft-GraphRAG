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
from typing import Iterable, Mapping, Protocol, Sequence

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
    document_name: str = ""


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
    question_source_evidence: tuple[GoldEvidence, ...] = ()
    answer_source_evidence: tuple[GoldEvidence, ...] = ()
    source_documents: tuple[str, ...] = ()


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

    def create_with_questions(
        self,
        project_id: str,
        name: str,
        questions: Sequence[BatchQuestion],
        description: str = "",
        method: str = "local",
    ) -> QuestionSet:
        self.projects.get(project_id)
        if not name.strip() or not questions:
            raise ProjectError("題目集名稱與題目不可空白")
        question_ids: set[str] = set()
        normalized_questions: set[str] = set()
        for item in questions:
            normalized = "".join(item.question.casefold().split())
            if not item.question.strip() or not item.reference_answer.strip():
                raise ProjectError("自動問答題目必須同時包含問題與正確答案")
            if item.question_id in question_ids or normalized in normalized_questions:
                raise ProjectError("題目集內含重複題號或重複問題")
            question_ids.add(item.question_id)
            normalized_questions.add(normalized)
        now = datetime.now(timezone.utc).isoformat()
        question_set = QuestionSet(
            question_set_id=uuid.uuid4().hex,
            project_id=project_id,
            name=name.strip(),
            description=description.strip(),
            method=method,
            imported_at=now,
            updated_at=now,
            questions=tuple(questions),
        )
        self._write(question_set)
        return question_set

    def update_answers(
        self,
        project_id: str,
        question_set_id: str,
        results: Mapping[str, QueryResult],
        method: str,
    ) -> QuestionSet:
        question_set = self.get(project_id, question_set_id)
        questions: list[BatchQuestion] = []
        for item in question_set.questions:
            result = results.get(item.question_id)
            if result is None:
                questions.append(item)
                continue
            questions.append(
                BatchQuestion(
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
                    question_source_evidence=item.question_source_evidence,
                    answer_source_evidence=item.answer_source_evidence,
                    source_documents=item.source_documents,
                )
            )
        updated = self._replace(question_set, questions, method)
        self._write(updated)
        return updated

    def update_questions(
        self,
        project_id: str,
        question_set_id: str,
        questions: Sequence[BatchQuestion],
    ) -> QuestionSet:
        question_set = self.get(project_id, question_set_id)
        previous_ids = {item.question_id for item in question_set.questions}
        updated_ids = {item.question_id for item in questions}
        if previous_ids != updated_ids or len(updated_ids) != len(questions):
            raise ProjectError("編輯後的題號不可變更或重複")
        if any(not item.question.strip() or not item.reference_answer.strip() for item in questions):
            raise ProjectError("問題與正確答案不可空白")
        normalized = ["".join(item.question.casefold().split()) for item in questions]
        if len(set(normalized)) != len(normalized):
            raise ProjectError("題目集不可包含重複問題")
        updated = self._replace(question_set, list(questions), question_set.method)
        self._write(updated)
        return updated

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
                "correct_answer",
                "reference_answer",
                "question_sources",
                "answer_sources",
                "source_documents",
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
            row["correct_answer"] = item.reference_answer
            row.pop("question_source_evidence", None)
            row.pop("answer_source_evidence", None)
            row["question_sources"] = self._format_source_pages(
                item.question_source_evidence or item.gold_evidence
            )
            row["answer_sources"] = self._format_source_pages(
                item.answer_source_evidence or item.gold_evidence
            )
            row["source_documents"] = "; ".join(self._source_documents(item))
            row["gold_evidence"] = json.dumps(row["gold_evidence"], ensure_ascii=False)
            row["retrieved_evidence"] = json.dumps(row["retrieved_evidence"], ensure_ascii=False)
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    def export_json(self, project_id: str, question_set_id: str) -> Path:
        question_set = self.get(project_id, question_set_id)
        json_path = self.projects.path_for(project_id) / "exports" / f"{question_set_id}.json"
        self._atomic_text(json_path, json.dumps(self._to_dict(question_set), ensure_ascii=False, indent=2) + "\n")
        return json_path

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
            reference_answer = raw.get("correct_answer", raw.get("reference_answer", ""))
            answer = raw.get("answer", "")
            status = raw.get("status", "PENDING")
            error = raw.get("error")
            duration_seconds = raw.get("duration_seconds")
            completed_at = raw.get("completed_at")
            source_documents_raw = raw.get("source_documents", [])
            retrieved_raw = raw.get("retrieved_evidence", [])
            try:
                if not isinstance(retrieved_raw, list):
                    raise ValueError("retrieved_evidence 必須是陣列")
                retrieved_evidence = tuple(Evidence(**item) for item in retrieved_raw)
            except (TypeError, ValueError) as exc:
                errors.append(f"{location}.retrieved_evidence：格式錯誤（{exc}）")
                retrieved_evidence = ()
            try:
                gold_evidence = QuestionSetService.parse_gold_evidence(raw.get("gold_evidence", []), location)
            except ProjectError as exc:
                errors.extend(str(exc).splitlines())
                gold_evidence = ()
            question_sources_raw = raw.get(
                "question_sources",
                raw.get("question_source_evidence", raw.get("question_source_pages")),
            )
            answer_sources_raw = raw.get(
                "answer_sources",
                raw.get("answer_source_evidence", raw.get("answer_source_pages")),
            )
            try:
                question_sources = (
                    QuestionSetService._parse_source_pages(
                        question_sources_raw, location + ".question_sources"
                    )
                    if question_sources_raw is not None
                    else gold_evidence
                )
                answer_sources = (
                    QuestionSetService._parse_source_pages(
                        answer_sources_raw, location + ".answer_sources"
                    )
                    if answer_sources_raw is not None
                    else gold_evidence
                )
            except ProjectError as exc:
                errors.extend(str(exc).splitlines())
                question_sources = answer_sources = gold_evidence
            if not gold_evidence:
                gold_evidence = answer_sources
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
            if not isinstance(answer, str):
                errors.append(f"{location}.answer：必須是字串")
            if not isinstance(source_documents_raw, list) or any(
                not isinstance(document, str) or not document.strip() for document in source_documents_raw
            ):
                errors.append(f"{location}.source_documents：必須是非空白文件名稱陣列")
                source_documents_raw = []
            if status not in QUESTION_STATUSES:
                errors.append(f"{location}.status：不支援的狀態")
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
                        status=status if status in QUESTION_STATUSES else "PENDING",
                        answer=answer.strip() if isinstance(answer, str) else "",
                        error=error if isinstance(error, str) else None,
                        duration_seconds=duration_seconds if isinstance(duration_seconds, (int, float)) else None,
                        completed_at=completed_at if isinstance(completed_at, str) else None,
                        gold_evidence=gold_evidence,
                        retrieved_evidence=retrieved_evidence,
                        question_source_evidence=question_sources,
                        answer_source_evidence=answer_sources,
                        source_documents=tuple(dict.fromkeys(item.strip() for item in source_documents_raw)),
                    )
                )
        if errors:
            raise ProjectError("題目集格式錯誤：\n- " + "\n- ".join(errors))
        return name.strip(), description.strip(), questions

    @staticmethod
    def _parse_source_pages(value: object, location: str) -> tuple[GoldEvidence, ...]:
        if not isinstance(value, list):
            raise ProjectError(f"{location}：必須是文件／頁碼陣列")
        parsed: list[GoldEvidence] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ProjectError(f"{location}[{index}]：必須是 object")
            document_id = item.get("document_id")
            document_name = item.get("document_name", document_id)
            pages = item.get("pages", [])
            chunk_ids = item.get("chunk_ids", [])
            if (
                not isinstance(document_id, str) or not document_id.strip()
                or not isinstance(document_name, str) or not document_name.strip()
                or not isinstance(pages, list)
                or any(not isinstance(page, int) or page < 1 for page in pages)
                or not isinstance(chunk_ids, list)
                or any(not isinstance(chunk, str) for chunk in chunk_ids)
            ):
                raise ProjectError(f"{location}[{index}]：文件、頁碼或 chunk_ids 格式錯誤")
            if not pages and not chunk_ids:
                raise ProjectError(f"{location}[{index}]：至少要有一個頁碼或 chunk_id")
            parsed.append(
                GoldEvidence(
                    document_id.strip(),
                    tuple(dict.fromkeys(pages)),
                    tuple(dict.fromkeys(chunk_ids)),
                    document_name.strip(),
                )
            )
        return tuple(parsed)

    @staticmethod
    def _format_source_pages(evidence: Sequence[GoldEvidence]) -> str:
        return json.dumps(QuestionSetService._sources_payload(evidence), ensure_ascii=False)

    @staticmethod
    def _sources_payload(evidence: Sequence[GoldEvidence]) -> list[dict[str, object]]:
        return [
            {
                "document_id": item.document_id,
                "document_name": item.document_name or item.document_id,
                "pages": list(item.pages),
            }
            for item in evidence
        ]

    @staticmethod
    def _source_documents(question: BatchQuestion) -> tuple[str, ...]:
        question_sources = question.question_source_evidence or question.gold_evidence
        answer_sources = question.answer_source_evidence or question.gold_evidence
        return tuple(
            dict.fromkeys(
                (
                    *(item.document_id for item in (*question_sources, *answer_sources)),
                    *question.source_documents,
                )
            )
        )

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
            document_name = raw.get("document_name", document_id)
            pages = raw.get("pages", [])
            chunk_ids = raw.get("chunk_ids", [])
            if not isinstance(document_id, str) or not document_id.strip():
                errors.append(f"{item_location}.document_id：必須是非空白字串")
            if not isinstance(document_name, str) or not document_name.strip():
                errors.append(f"{item_location}.document_name：必須是非空白字串")
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
                        document_name=document_name.strip() if isinstance(document_name, str) else document_id.strip(),
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
            question_sources_raw = item.get("question_sources", item.get("question_source_pages"))
            answer_sources_raw = item.get("answer_sources", item.get("answer_source_pages"))
            for derived in (
                "correct_answer", "question_source_pages", "answer_source_pages",
                "question_sources", "answer_sources",
            ):
                item.pop(derived, None)
            item["gold_evidence"] = tuple(
                GoldEvidence(
                    document_id=gold["document_id"],
                    pages=tuple(gold.get("pages", [])),
                    chunk_ids=tuple(gold.get("chunk_ids", [])),
                    document_name=gold.get("document_name", gold["document_id"]),
                )
                for gold in item.get("gold_evidence", [])
            )
            item["retrieved_evidence"] = tuple(
                Evidence(**evidence) for evidence in item.get("retrieved_evidence", [])
            )
            for source_key, raw_sources in (
                ("question_source_evidence", question_sources_raw),
                ("answer_source_evidence", answer_sources_raw),
            ):
                if source_key in item:
                    item[source_key] = tuple(
                        GoldEvidence(
                            document_id=source["document_id"],
                            pages=tuple(source.get("pages", [])),
                            chunk_ids=tuple(source.get("chunk_ids", [])),
                            document_name=source.get("document_name", source["document_id"]),
                        )
                        for source in item.get(source_key, [])
                    )
                else:
                    item[source_key] = QuestionSetService._parse_source_pages(
                        raw_sources, source_key
                    ) if raw_sources is not None else ()
            item["source_documents"] = tuple(item.get("source_documents", []))
            questions.append(BatchQuestion(**item))
        return QuestionSet(**value, questions=tuple(questions))

    @staticmethod
    def _to_dict(question_set: QuestionSet) -> dict[str, object]:
        value = asdict(question_set)
        questions = []
        for item in question_set.questions:
            row = asdict(item)
            question_sources = item.question_source_evidence or item.gold_evidence
            answer_sources = item.answer_source_evidence or item.gold_evidence
            row["correct_answer"] = item.reference_answer
            row.pop("question_source_evidence", None)
            row.pop("answer_source_evidence", None)
            row["question_sources"] = QuestionSetService._sources_payload(question_sources)
            row["answer_sources"] = QuestionSetService._sources_payload(answer_sources)
            row["source_documents"] = list(QuestionSetService._source_documents(item))
            questions.append(row)
        value["questions"] = questions
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
