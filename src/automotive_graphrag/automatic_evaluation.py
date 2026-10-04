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

from .connections import ALLOWED_CHAT_MODELS, ConnectionSettings
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, QuestionSetService
from .reviews import ReviewService


PROMPT_VERSION = "answer-match-judge-v2"
JUDGMENT_RESULTS = {"correct", "incorrect"}
JudgeClient = Callable[[str, str, str, str], object]


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationItem:
    question_id: str
    is_correct: bool
    judge_reason: str


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationResult:
    project_id: str
    question_set_id: str
    evaluated_at: str
    model: str
    prompt_version: str
    judge_prompt: str
    question_count: int
    correct_count: int
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
        rerun_answers: bool = False,
        only_previous_failures: bool = False,
        model: str | None = None,
    ) -> AutomaticEvaluationResult:
        previous = self.last_result(project_id, question_set_id)
        selected_ids: set[str] | None = None
        if only_previous_failures:
            if previous is None:
                raise ProjectError("尚無自動評測結果，無法只重跑失敗題")
            selected_ids = {item.question_id for item in previous.items if not item.is_correct}
            if not selected_ids:
                raise ProjectError("前次評測沒有答錯的題目")

        question_set = self.question_sets.get(project_id, question_set_id)
        eligible = [
            item
            for item in question_set.questions
            if item.reference_answer
            and (selected_ids is None or item.question_id in selected_ids)
        ]
        if not eligible:
            raise ProjectError("題目集沒有包含可供評判的正確答案")
        if rerun_answers:
            target_ids = [item.question_id for item in eligible]
            question_set = self.question_sets.run(project_id, question_set_id, selected_question_ids=target_ids)
            eligible = [item for item in question_set.questions if item.question_id in set(target_ids)]
        eligible = [item for item in eligible if item.status == "COMPLETED" and item.answer]
        if not eligible:
            raise ProjectError("沒有已完成回答且包含正確答案的題目可供評判")

        prompt = self._build_prompt(eligible)
        api_key = self.connections.apply_to_environment(project_id)
        base_url = self.connections.get_api_base_url()
        model = model or self.connections.get_chat_model()
        if model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的評判模型：{model}")
        try:
            response = self.client(base_url, api_key, model, prompt)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"自動評測 API 呼叫失敗：{exc}") from exc
        judged = self._parse_response(response, eligible)
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
            question_count=len(items),
            correct_count=sum(item.is_correct for item in items),
            items=items,
        )
        self._write(result)
        return result

    def evaluate_single_answer(
        self,
        project_id: str,
        question_id: str,
        question: str,
        correct_answer: str,
        system_answer: str,
        model: str,
    ) -> AutomaticEvaluationItem:
        """Judge one answer without retrieval evidence or shared question-set mutation."""
        if model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的評判模型：{model}")
        case = BatchQuestion(
            question_id=question_id,
            question=question,
            reference_answer=correct_answer,
            answer=system_answer,
        )
        prompt = self._build_prompt([case])
        api_key = self.connections.apply_to_environment(project_id)
        try:
            response = self.client(
                self.connections.get_api_base_url(), api_key, model, prompt
            )
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"答案評判 API 呼叫失敗：{exc}") from exc
        return self._parse_response(response, [case])[0]

    def last_result(self, project_id: str, question_set_id: str) -> AutomaticEvaluationResult | None:
        try:
            value = json.loads(self._path(project_id, question_set_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ProjectError("自動評測紀錄格式錯誤") from exc
        # Migrate records written by the former scored/evidence-based judge.
        raw_items = value.pop("items", [])
        migrated_items = []
        for item in raw_items:
            if "is_correct" not in item:
                item = {
                    "question_id": item["question_id"],
                    "is_correct": item.get("answer_score", 0) >= 4,
                    "judge_reason": item.get("judge_reason", "舊版評測結果；請重新評判"),
                }
            migrated_items.append(AutomaticEvaluationItem(**item))
        allowed_fields = set(AutomaticEvaluationResult.__dataclass_fields__) - {"items"}
        value = {key: field for key, field in value.items() if key in allowed_fields}
        value.setdefault("correct_count", sum(item.is_correct for item in migrated_items))
        value["items"] = tuple(migrated_items)
        return AutomaticEvaluationResult(**value)

    def update_manual_results(
        self, project_id: str, question_set_id: str, decisions: dict[str, str]
    ) -> AutomaticEvaluationResult:
        """Save manual correctness edits and clear stale judge reasons for changed decisions."""
        result = self.last_result(project_id, question_set_id)
        if result is None:
            raise ProjectError("尚無評測結果可供修改")
        unknown = set(decisions) - {item.question_id for item in result.items}
        if unknown:
            raise ProjectError("評測結果找不到題號：" + ", ".join(sorted(unknown)))
        items = []
        for item in result.items:
            decision = decisions.get(item.question_id)
            if decision is None:
                items.append(item)
                continue
            if decision not in {"正確", "錯誤"}:
                raise ProjectError(f"{item.question_id} 的判斷只能是「正確」或「錯誤」")
            is_correct = decision == "正確"
            items.append(AutomaticEvaluationItem(
                question_id=item.question_id,
                is_correct=is_correct,
                judge_reason="" if is_correct != item.is_correct else item.judge_reason,
            ))
        updated = AutomaticEvaluationResult(
            project_id=result.project_id,
            question_set_id=result.question_set_id,
            evaluated_at=result.evaluated_at,
            model=result.model,
            prompt_version=result.prompt_version,
            judge_prompt=result.judge_prompt,
            question_count=len(items),
            correct_count=sum(item.is_correct for item in items),
            items=tuple(items),
        )
        self._write(updated)
        return updated

    def clear_result(self, project_id: str, question_set_id: str) -> None:
        path = self._path(project_id, question_set_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ProjectError("無法清除舊評測結果") from exc

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
    def _build_prompt(questions: list[BatchQuestion]) -> str:
        cases = [
            {
                "question_id": item.question_id,
                "question": item.question,
                "correct_answer": item.reference_answer,
                "system_answer": item.answer,
            }
            for item in questions
        ]
        return (
            "你是答案比對員。只根據每筆資料中的題目、正確答案和系統回答判斷，不得使用外部知識或任何檢索證據。\n"
            "若系統回答與正確答案表達的內容一致，且完整包含所有必要步驟、條件、順序與要求，判為 correct。\n"
            "只要缺少必要內容／步驟、步驟順序錯誤、與正確答案矛盾，或加入正確答案未支持的額外步驟／實質資訊，判為 incorrect。"
            "不要求字面完全相同；同義改寫可接受，但不能因此省略或新增實質內容。非流程型答案也不得缺漏或添加實質主張。\n"
            "只輸出 JSON object，格式為 "
            '{"items":[{"question_id":"...","result":"correct或incorrect","reason":"..."}]}。'
            "每個輸入題號必須恰好出現一次。reason 簡短指出符合之處，或具體缺漏、錯誤、額外內容。\n"
            f"評測案例：{json.dumps(cases, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_response(
        response: object, questions: list[BatchQuestion]
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
            judgment = raw.get("result")
            reason = raw.get("reason")
            if not isinstance(question_id, str) or question_id not in question_by_id or question_id in seen:
                raise ProjectError("Judge 回傳未知或重複的 question_id")
            if not isinstance(judgment, str) or judgment not in JUDGMENT_RESULTS:
                raise ProjectError(f"{question_id} 的 result 必須是 correct 或 incorrect")
            if not isinstance(reason, str) or not reason.strip():
                raise ProjectError(f"{question_id} 缺少 Judge reason")
            result.append(
                AutomaticEvaluationItem(
                    question_id=question_id,
                    is_correct=judgment == "correct",
                    judge_reason=reason.strip(),
                )
            )
            seen.add(question_id)
        return result

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
