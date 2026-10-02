"""Cost-conscious LLM judging for completed grounded answers."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .connections import ConnectionSettings
from .evidence import Evidence
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, GoldEvidence, QuestionSetService
from .reviews import ReviewService


PROMPT_VERSION = "answer-evidence-judge-v1"
CONFIDENCE_LEVELS = {"high", "medium", "low"}
JudgeClient = Callable[[str, str, str, str], object]


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationItem:
    question_id: str
    retrieval_pass: bool
    answer_score: int
    evidence_support_score: int
    judge_reason: str
    judge_confidence: str
    needs_human_review: bool


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationResult:
    project_id: str
    question_set_id: str
    evaluated_at: str
    model: str
    prompt_version: str
    judge_prompt: str
    top_k: int
    question_count: int
    average_answer_score: float
    average_evidence_support_score: float
    human_review_count: int
    items: tuple[AutomaticEvaluationItem, ...]


class AutomaticEvaluationService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService | None = None,
        connections: ConnectionSettings | None = None,
        client: JudgeClient | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)
        self.connections = connections or ConnectionSettings(projects.root)
        self.client = client or self._openai_chat

    def evaluate(
        self,
        project_id: str,
        question_set_id: str,
        top_k: int = 5,
        rerun_answers: bool = False,
        only_previous_failures: bool = False,
        maximum_evidence_characters: int = 1200,
    ) -> AutomaticEvaluationResult:
        if not isinstance(top_k, int) or top_k < 1:
            raise ProjectError("Top-K 必須是正整數")
        if maximum_evidence_characters < 200:
            raise ProjectError("每筆 Evidence 字數上限不可小於 200")
        previous = self.last_result(project_id, question_set_id)
        selected_ids: set[str] | None = None
        if only_previous_failures:
            if previous is None:
                raise ProjectError("尚無自動評測結果，無法只重跑失敗題")
            selected_ids = {item.question_id for item in previous.items if item.needs_human_review}
            if not selected_ids:
                raise ProjectError("前次評測沒有需要重跑的題目")

        question_set = self.question_sets.get(project_id, question_set_id)
        eligible = [
            item
            for item in question_set.questions
            if item.reference_answer and item.gold_evidence and (selected_ids is None or item.question_id in selected_ids)
        ]
        if not eligible:
            raise ProjectError("題目集沒有同時包含 Reference Answer 與 Gold Evidence 的題目")
        if rerun_answers:
            target_ids = [item.question_id for item in eligible]
            question_set = self.question_sets.run(project_id, question_set_id, selected_question_ids=target_ids)
            eligible = [item for item in question_set.questions if item.question_id in set(target_ids)]
        incomplete = [item.question_id for item in eligible if item.status != "COMPLETED" or not item.answer]
        if incomplete:
            raise ProjectError(f"以下題目尚未完成系統回答：{', '.join(incomplete)}")

        prompt = self._build_prompt(eligible, top_k, maximum_evidence_characters)
        api_key = self.connections.apply_to_environment()
        base_url = self.connections.get_api_base_url()
        model = self.connections.get_chat_model()
        try:
            response = self.client(base_url, api_key, model, prompt)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"自動評測 API 呼叫失敗：{exc}") from exc
        judged = self._parse_response(response, eligible, top_k)
        if selected_ids is not None and previous is not None:
            judged_by_id = {item.question_id: item for item in judged}
            items = tuple(judged_by_id.get(item.question_id, item) for item in previous.items)
        else:
            items = tuple(judged)
        result = AutomaticEvaluationResult(
            project_id=project_id,
            question_set_id=question_set_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model=model,
            prompt_version=PROMPT_VERSION,
            judge_prompt=prompt,
            top_k=top_k,
            question_count=len(items),
            average_answer_score=sum(item.answer_score for item in items) / len(items),
            average_evidence_support_score=sum(item.evidence_support_score for item in items) / len(items),
            human_review_count=sum(item.needs_human_review for item in items),
            items=items,
        )
        self._write(result)
        return result

    def last_result(self, project_id: str, question_set_id: str) -> AutomaticEvaluationResult | None:
        try:
            value = json.loads(self._path(project_id, question_set_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ProjectError("自動評測紀錄格式錯誤") from exc
        value["items"] = tuple(AutomaticEvaluationItem(**item) for item in value.get("items", []))
        return AutomaticEvaluationResult(**value)

    def export(self, project_id: str, question_set_id: str) -> tuple[Path, Path]:
        result = self.last_result(project_id, question_set_id)
        if result is None:
            raise ProjectError("尚未執行自動評測")
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / f"{question_set_id}-automatic-evaluation.json"
        csv_path = directory / f"{question_set_id}-automatic-evaluation.csv"
        reviews = ReviewService(self.projects, self.question_sets).records(project_id, question_set_id)
        review_by_id = {item.question_id: item for item in reviews}
        payload = asdict(result)
        payload["human_reviews"] = [asdict(item) for item in reviews if item.human_label is not None]
        self._atomic_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        fieldnames = [
            *AutomaticEvaluationItem.__dataclass_fields__,
            "human_label",
            "reviewer_note",
            "reviewed_at",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.items:
            row = asdict(item)
            review = review_by_id.get(item.question_id)
            row.update(
                {
                    "human_label": review.human_label if review else None,
                    "reviewer_note": review.reviewer_note if review else "",
                    "reviewed_at": review.reviewed_at if review else None,
                }
            )
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    @staticmethod
    def _build_prompt(questions: list[BatchQuestion], top_k: int, maximum_characters: int) -> str:
        cases = [
            {
                "question_id": item.question_id,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "system_answer": item.answer,
                "retrieved_evidence": [
                    {
                        "document_id": evidence.document_id,
                        "page": evidence.page,
                        "chunk_id": evidence.chunk_id,
                        "text": evidence.text[:maximum_characters],
                    }
                    for evidence in item.retrieved_evidence[:top_k]
                ],
            }
            for item in questions
        ]
        return (
            "你是汽車維修問答評測員。只依參考答案與 Retrieved Evidence 評分，不可加入外部知識。\n"
            "answer_score：1-5，評估系統答案相對參考答案的正確性與完整性。\n"
            "evidence_support_score：1-5，評估 Retrieved Evidence 是否直接支持系統答案。\n"
            "confidence 只能是 high、medium、low。reason 必須簡短指出缺漏、矛盾或支持依據。\n"
            "只輸出 JSON object，格式為 "
            '{"items":[{"question_id":"...","answer_score":1,"evidence_support_score":1,'
            '"reason":"...","confidence":"high"}]}。每個輸入題號必須恰好出現一次。\n'
            f"評測案例：{json.dumps(cases, ensure_ascii=False)}"
        )

    @classmethod
    def _parse_response(
        cls, response: object, questions: list[BatchQuestion], top_k: int
    ) -> list[AutomaticEvaluationItem]:
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise ProjectError("自動評測結果不是有效 JSON") from exc
        if not isinstance(response, dict) or not isinstance(response.get("items"), list):
            raise ProjectError("自動評測結果缺少 items 陣列")
        raw_items = response["items"]
        question_by_id = {item.question_id: item for item in questions}
        if len(raw_items) != len(question_by_id):
            raise ProjectError("Judge 回傳題數與預期不符")
        result: list[AutomaticEvaluationItem] = []
        seen: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ProjectError("Judge 回傳項目格式錯誤")
            question_id = raw.get("question_id")
            answer_score = raw.get("answer_score")
            evidence_score = raw.get("evidence_support_score")
            reason = raw.get("reason")
            confidence = raw.get("confidence")
            if not isinstance(question_id, str) or question_id not in question_by_id or question_id in seen:
                raise ProjectError("Judge 回傳未知或重複的 question_id")
            if not isinstance(answer_score, int) or isinstance(answer_score, bool) or not 1 <= answer_score <= 5:
                raise ProjectError(f"{question_id} 的 answer_score 必須是 1 到 5")
            if not isinstance(evidence_score, int) or isinstance(evidence_score, bool) or not 1 <= evidence_score <= 5:
                raise ProjectError(f"{question_id} 的 evidence_support_score 必須是 1 到 5")
            if not isinstance(reason, str) or not reason.strip():
                raise ProjectError(f"{question_id} 缺少 Judge reason")
            if confidence not in CONFIDENCE_LEVELS:
                raise ProjectError(f"{question_id} 的 confidence 格式錯誤")
            question = question_by_id[question_id]
            retrieval_pass = any(
                cls._is_relevant(evidence, question.gold_evidence)
                for evidence in question.retrieved_evidence[:top_k]
            )
            needs_review = not retrieval_pass or answer_score < 4 or evidence_score < 4 or confidence != "high"
            result.append(
                AutomaticEvaluationItem(
                    question_id=question_id,
                    retrieval_pass=retrieval_pass,
                    answer_score=answer_score,
                    evidence_support_score=evidence_score,
                    judge_reason=reason.strip(),
                    judge_confidence=confidence,
                    needs_human_review=needs_review,
                )
            )
            seen.add(question_id)
        return result

    @staticmethod
    def _is_relevant(evidence: Evidence, gold_evidence: tuple[GoldEvidence, ...]) -> bool:
        return any(
            evidence.document_id == gold.document_id
            and (
                (bool(gold.chunk_ids) and evidence.chunk_id in gold.chunk_ids)
                or (bool(gold.pages) and evidence.page in gold.pages)
            )
            for gold in gold_evidence
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
            raise ProjectError(f"自動評測 API 回應 HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProjectError("自動評測 API 無法連線或回應格式錯誤") from exc
        try:
            return json.loads(value["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProjectError("自動評測 API 回應缺少有效 JSON 內容") from exc

    def _path(self, project_id: str, question_set_id: str) -> Path:
        self.question_sets.get(project_id, question_set_id)
        return self.projects.path_for(project_id) / "runs" / f"automatic-evaluation-{question_set_id}.json"

    def _write(self, result: AutomaticEvaluationResult) -> None:
        self._atomic_text(
            self._path(result.project_id, result.question_set_id),
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
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
