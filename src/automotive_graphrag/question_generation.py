"""Generate grounded candidate questions from sampled source text."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .connections import ALLOWED_CHAT_MODELS, ConnectionSettings
from .projects import ProjectError, ProjectStore
from .question_sets import GoldEvidence
from .source_sampling import SourceSample, SourceSampleBatch, SourceSamplingService


DIFFICULTIES = {"simple", "medium", "cross_section"}
GENERATION_STATUSES = {"pending_review", "approved", "rejected"}
GenerationClient = Callable[[str, str, str, str], object]


@dataclass(frozen=True, slots=True)
class GeneratedQuestion:
    question_id: str
    question: str
    reference_answer: str
    gold_evidence: tuple[GoldEvidence, ...]
    source_sample_ids: tuple[str, ...]
    difficulty: str
    generation_status: str
    question_source_evidence: tuple[GoldEvidence, ...] = ()
    answer_source_evidence: tuple[GoldEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedQuestionBatch:
    generation_batch_id: str
    project_id: str
    sample_batch_id: str
    created_at: str
    updated_at: str
    model: str
    prompt: str
    questions: tuple[GeneratedQuestion, ...]


class QuestionGenerationService:
    def __init__(
        self,
        projects: ProjectStore,
        sampling: SourceSamplingService | None = None,
        connections: ConnectionSettings | None = None,
        client: GenerationClient | None = None,
    ) -> None:
        self.projects = projects
        self.sampling = sampling or SourceSamplingService(projects)
        self.connections = connections or ConnectionSettings(projects.root)
        self.client = client or self._openai_chat

    def generate(
        self,
        project_id: str,
        sample_batch_id: str,
        question_count: int,
        difficulty: str = "simple",
        maximum_source_characters: int = 2500,
    ) -> GeneratedQuestionBatch:
        normalized_difficulty = difficulty.strip().lower()
        if normalized_difficulty not in DIFFICULTIES:
            raise ProjectError(f"不支援的題目難度：{difficulty}")
        if question_count < 1:
            raise ProjectError("生成題數必須是正整數")
        if maximum_source_characters < 200:
            raise ProjectError("每筆來源字數上限不可小於 200")
        sample_batch = self.sampling.get(project_id, sample_batch_id)
        self._validate_source_coverage(sample_batch, question_count, normalized_difficulty)
        prompt = self._build_prompt(
            sample_batch,
            question_count,
            normalized_difficulty,
            maximum_source_characters,
        )
        api_key = self.connections.apply_to_environment(project_id)
        api_base_url = self.connections.get_api_base_url()
        model = self.connections.get_chat_model()
        try:
            response = self.client(api_base_url, api_key, model, prompt)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"題目生成 API 呼叫失敗：{exc}") from exc
        questions = self._parse_response(
            response,
            sample_batch,
            question_count,
            normalized_difficulty,
        )
        now = datetime.now(timezone.utc).isoformat()
        batch = GeneratedQuestionBatch(
            generation_batch_id=uuid.uuid4().hex,
            project_id=project_id,
            sample_batch_id=sample_batch_id,
            created_at=now,
            updated_at=now,
            model=model,
            prompt=prompt,
            questions=tuple(questions),
        )
        self._write(batch)
        return batch

    def generate_for_document(
        self,
        project_id: str,
        document_id: str,
        samples: Sequence[SourceSample],
        question_count: int,
        model: str,
        maximum_source_characters: int = 2500,
        excluded_questions: Sequence[str] = (),
    ) -> tuple[GeneratedQuestion, ...]:
        if question_count < 1:
            raise ProjectError("每份 PDF 生成題數必須是正整數")
        if maximum_source_characters < 200:
            raise ProjectError("每筆來源字數上限不可小於 200")
        if model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的題目生成模型：{model}")
        document_samples = tuple(item for item in samples if item.document_id == document_id)
        if not document_samples:
            raise ProjectError(f"PDF {document_id} 沒有可用的前處理原文")

        sample_batch = SourceSampleBatch(
            sample_batch_id=uuid.uuid4().hex,
            project_id=project_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            section_ids=tuple(sorted({item.section_id for item in document_samples})),
            page_from=None,
            page_to=None,
            content_type="all",
            minimum_characters=1,
            seed=0,
            requested_count=len(document_samples),
            samples=document_samples,
        )
        prompt = self._build_prompt(
            sample_batch,
            question_count,
            "simple",
            maximum_source_characters,
        )
        if excluded_questions:
            prompt += (
                "\n以下既有題目禁止重複或只換同義詞，請生成不同考點的問題："
                + json.dumps(list(excluded_questions), ensure_ascii=False)
            )
        api_key = self.connections.apply_to_environment(project_id)
        try:
            response = self.client(
                self.connections.get_api_base_url(),
                api_key,
                model,
                prompt,
            )
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"{document_id} 題目生成 API 呼叫失敗：{exc}") from exc
        questions = self._parse_response(response, sample_batch, question_count, "simple")
        seen: set[str] = set()
        for item in questions:
            normalized = normalize_question(item.question)
            if normalized in seen:
                raise ProjectError(f"{document_id} 的生成結果有重複題目，請重新執行")
            seen.add(normalized)
        return tuple(questions)

    def get(self, project_id: str, generation_batch_id: str) -> GeneratedQuestionBatch:
        if not generation_batch_id.isalnum():
            raise ProjectError("生成批次 ID 格式不正確")
        path = self._path(project_id, generation_batch_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError(f"找不到題目生成批次：{generation_batch_id}") from exc
        questions = tuple(self._question_from_dict(item) for item in value.pop("questions"))
        return GeneratedQuestionBatch(**value, questions=questions)

    def update_question(
        self,
        project_id: str,
        generation_batch_id: str,
        question_id: str,
        question: str,
        reference_answer: str,
        generation_status: str,
    ) -> GeneratedQuestionBatch:
        status = generation_status.strip().lower()
        if status not in GENERATION_STATUSES:
            raise ProjectError("題目狀態必須是 pending_review、approved 或 rejected")
        if not question.strip() or not reference_answer.strip():
            raise ProjectError("問題與參考答案不可空白")
        batch = self.get(project_id, generation_batch_id)
        found = False
        questions: list[GeneratedQuestion] = []
        for item in batch.questions:
            if item.question_id == question_id:
                found = True
                questions.append(
                    GeneratedQuestion(
                        question_id=item.question_id,
                        question=question.strip(),
                        reference_answer=reference_answer.strip(),
                        gold_evidence=item.gold_evidence,
                        source_sample_ids=item.source_sample_ids,
                        difficulty=item.difficulty,
                        generation_status=status,
                    )
                )
            else:
                questions.append(item)
        if not found:
            raise ProjectError(f"找不到生成題目：{question_id}")
        updated = GeneratedQuestionBatch(
            generation_batch_id=batch.generation_batch_id,
            project_id=batch.project_id,
            sample_batch_id=batch.sample_batch_id,
            created_at=batch.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            model=batch.model,
            prompt=batch.prompt,
            questions=tuple(questions),
        )
        self._write(updated)
        return updated

    def export_question_set(self, project_id: str, generation_batch_id: str) -> Path:
        batch = self.get(project_id, generation_batch_id)
        approved = [item for item in batch.questions if item.generation_status == "approved"]
        if not approved:
            raise ProjectError("沒有已核准的生成題目可匯出")
        payload = {
            "name": f"AUTO-{generation_batch_id[:8]}",
            "description": f"由原文取樣批次 {batch.sample_batch_id} 生成；模型 {batch.model}",
            "questions": [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "correct_answer": item.reference_answer,
                    "reference_answer": item.reference_answer,
                    "gold_evidence": [asdict(gold) for gold in item.gold_evidence],
                    "question_sources": [
                        {
                            "document_id": source.document_id,
                            "document_name": source.document_name or source.document_id,
                            "pages": list(source.pages),
                        }
                        for source in (item.question_source_evidence or item.gold_evidence)
                    ],
                    "answer_sources": [
                        {
                            "document_id": source.document_id,
                            "document_name": source.document_name or source.document_id,
                            "pages": list(source.pages),
                        }
                        for source in (item.answer_source_evidence or item.gold_evidence)
                    ],
                    "source_documents": list(
                        dict.fromkeys(
                            source.document_id
                            for source in (
                                *(item.question_source_evidence or item.gold_evidence),
                                *(item.answer_source_evidence or item.gold_evidence),
                            )
                        )
                    ),
                    "difficulty": item.difficulty,
                    "generation_status": item.generation_status,
                }
                for item in approved
            ],
        }
        path = self.projects.path_for(project_id) / "exports" / f"{generation_batch_id}-question_set.json"
        self._atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return path

    @staticmethod
    def _validate_source_coverage(batch: SourceSampleBatch, count: int, difficulty: str) -> None:
        if len(batch.samples) < count:
            raise ProjectError("取樣原文數量少於生成題數")
        if difficulty == "medium" and len(batch.samples) < 2:
            raise ProjectError("中等題目至少需要兩筆取樣原文")
        if difficulty == "cross_section" and len({item.section_id for item in batch.samples}) < 2:
            raise ProjectError("跨章節題目至少需要兩個不同章節的原文")

    @staticmethod
    def _build_prompt(
        batch: SourceSampleBatch,
        count: int,
        difficulty: str,
        maximum_source_characters: int,
    ) -> str:
        sources = [
            {
                "sample_id": item.sample_id,
                "document_id": item.document_id,
                "page": item.page,
                "section_id": item.section_id,
                "text": item.text[:maximum_source_characters],
            }
            for item in batch.samples
        ]
        return (
            "你是汽車維修手冊題目設計器。只能使用提供的原文，不可加入外部知識。\n"
            "問題與參考答案必須使用繁體中文；零件名稱、縮寫、DTC、單位與原廠術語可保留英文。"
            "不得因翻譯加入原文沒有的資訊。\n"
            f"生成 {count} 題，難度固定為 {difficulty}。問題不可直接暴露答案；參考答案必須可由引用原文完整支持。\n"
            "只輸出 JSON object，格式為 "
            '{"questions":[{"question":"...","reference_answer":"...","question_source_sample_ids":["..."],'
            '"answer_source_sample_ids":["..."]}]}。\n'
            "simple 每題只能引用一筆來源；medium 每題至少兩筆來源；cross_section 每題至少引用兩個不同 section。\n"
            f"來源：{json.dumps(sources, ensure_ascii=False)}"
        )

    def _parse_response(
        self,
        response: object,
        sample_batch: SourceSampleBatch,
        count: int,
        difficulty: str,
    ) -> list[GeneratedQuestion]:
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise ProjectError("題目生成結果不是有效 JSON") from exc
        if not isinstance(response, dict) or not isinstance(response.get("questions"), list):
            raise ProjectError("題目生成結果缺少 questions 陣列")
        raw_questions = response["questions"]
        if len(raw_questions) != count:
            raise ProjectError(f"模型回傳 {len(raw_questions)} 題，預期 {count} 題")
        source_by_id = {item.sample_id: item for item in sample_batch.samples}
        questions: list[GeneratedQuestion] = []
        for index, raw in enumerate(raw_questions, start=1):
            if not isinstance(raw, dict):
                raise ProjectError(f"生成題目第 {index} 筆格式錯誤")
            question = raw.get("question")
            answer = raw.get("reference_answer")
            fallback_source_ids = raw.get("source_sample_ids")
            question_source_ids = raw.get("question_source_sample_ids", fallback_source_ids)
            answer_source_ids = raw.get("answer_source_sample_ids", fallback_source_ids)
            if not isinstance(question, str) or not question.strip():
                raise ProjectError(f"生成題目第 {index} 筆缺少問題")
            if not isinstance(answer, str) or not answer.strip():
                raise ProjectError(f"生成題目第 {index} 筆缺少參考答案")
            if not self._contains_chinese(question) or not self._contains_chinese(answer):
                raise ProjectError(f"生成題目第 {index} 筆未使用繁體中文，請重新生成")
            valid_source_ids = lambda values: (
                isinstance(values, list)
                and bool(values)
                and all(isinstance(source_id, str) and source_id in source_by_id for source_id in values)
            )
            if not valid_source_ids(question_source_ids):
                raise ProjectError(f"生成題目第 {index} 筆引用不存在的 sample_id（question source）")
            if not valid_source_ids(answer_source_ids):
                raise ProjectError(f"生成題目第 {index} 筆引用不存在的 sample_id")
            question_ids = tuple(dict.fromkeys(question_source_ids))
            answer_ids = tuple(dict.fromkeys(answer_source_ids))
            question_sources = [source_by_id[source_id] for source_id in question_ids]
            answer_sources = [source_by_id[source_id] for source_id in answer_ids]
            self._validate_difficulty_sources(answer_sources, difficulty, index)
            selected_sources = list({item.sample_id: item for item in (*question_sources, *answer_sources)}.values())
            unique_ids = tuple(item.sample_id for item in selected_sources)
            sections = sorted({item.section_id for item in answer_sources})
            section_label = sections[0] if len(sections) == 1 else "CROSS"
            questions.append(
                GeneratedQuestion(
                    question_id=f"AUTO-{section_label}-{index:04d}",
                    question=question.strip(),
                    reference_answer=answer.strip(),
                    gold_evidence=self._gold_evidence(answer_sources),
                    source_sample_ids=unique_ids,
                    difficulty=difficulty,
                    generation_status="pending_review",
                    question_source_evidence=self._gold_evidence(question_sources),
                    answer_source_evidence=self._gold_evidence(answer_sources),
                )
            )
        return questions

    @staticmethod
    def _contains_chinese(value: str) -> bool:
        return re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value) is not None

    @staticmethod
    def _validate_difficulty_sources(sources: list[SourceSample], difficulty: str, index: int) -> None:
        if difficulty == "simple" and len(sources) != 1:
            raise ProjectError(f"生成題目第 {index} 筆的 simple 難度必須引用一筆來源")
        if difficulty == "medium" and len(sources) < 2:
            raise ProjectError(f"生成題目第 {index} 筆的 medium 難度至少引用兩筆來源")
        if difficulty == "cross_section" and len({item.section_id for item in sources}) < 2:
            raise ProjectError(f"生成題目第 {index} 筆的 cross_section 難度必須跨章節")

    @staticmethod
    def _gold_evidence(sources: list[SourceSample]) -> tuple[GoldEvidence, ...]:
        grouped: dict[str, dict[str, Any]] = {}
        for source in sources:
            group = grouped.setdefault(
                source.document_id,
                {"pages": [], "chunk_ids": [], "document_name": source.document_id},
            )
            group["pages"].append(source.page)
            group["chunk_ids"].append(source.chunk_id)
        return tuple(
            GoldEvidence(
                document_id=document_id,
                pages=tuple(dict.fromkeys(values["pages"])),
                chunk_ids=tuple(dict.fromkeys(values["chunk_ids"])),
                document_name=values["document_name"],
            )
            for document_id, values in grouped.items()
        )

    @staticmethod
    def _openai_chat(api_base_url: str, api_key: str, model: str, prompt: str) -> object:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{api_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProjectError(f"題目生成 API 回應 HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProjectError("題目生成 API 無法連線或回應格式錯誤") from exc
        try:
            return json.loads(value["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProjectError("題目生成 API 回應缺少有效 JSON 內容") from exc

    def _path(self, project_id: str, generation_batch_id: str) -> Path:
        return self.projects.path_for(project_id) / "runs" / f"generated-questions-{generation_batch_id}.json"

    def _write(self, batch: GeneratedQuestionBatch) -> None:
        self._atomic_text(self._path(batch.project_id, batch.generation_batch_id), json.dumps(asdict(batch), ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _question_from_dict(value: dict[str, Any]) -> GeneratedQuestion:
        value["gold_evidence"] = tuple(
            GoldEvidence(
                document_id=item["document_id"],
                pages=tuple(item.get("pages", [])),
                chunk_ids=tuple(item.get("chunk_ids", [])),
                document_name=item.get("document_name", item["document_id"]),
            )
            for item in value.get("gold_evidence", [])
        )
        value["source_sample_ids"] = tuple(value.get("source_sample_ids", []))
        for field_name in ("question_source_evidence", "answer_source_evidence"):
            value[field_name] = tuple(
                GoldEvidence(
                    document_id=item["document_id"],
                    pages=tuple(item.get("pages", [])),
                    chunk_ids=tuple(item.get("chunk_ids", [])),
                    document_name=item.get("document_name", item["document_id"]),
                )
                for item in value.get(field_name, [])
            )
        return GeneratedQuestion(**value)

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def normalize_question(value: str) -> str:
    """Normalize whitespace and punctuation for cross-PDF duplicate detection."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if character.isalnum()
    )
