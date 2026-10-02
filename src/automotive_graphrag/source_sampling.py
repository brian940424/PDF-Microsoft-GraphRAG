"""Filter and reproducibly sample original processed PDF text."""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .projects import ProjectError, ProjectStore


CONTENT_TYPES = {"all", "diagnostic", "procedure", "specification", "general"}


@dataclass(frozen=True, slots=True)
class SourceSample:
    sample_id: str
    chunk_id: str
    project_id: str
    section_id: str
    section_name: str
    document_id: str
    page: int
    block_id: str
    content_type: str
    character_count: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceSampleBatch:
    sample_batch_id: str
    project_id: str
    created_at: str
    section_ids: tuple[str, ...]
    page_from: int | None
    page_to: int | None
    content_type: str
    minimum_characters: int
    seed: int
    requested_count: int
    samples: tuple[SourceSample, ...]


class SourceSamplingService:
    def __init__(self, projects: ProjectStore) -> None:
        self.projects = projects

    def available_sections(self, project_id: str) -> list[tuple[str, str]]:
        records = self._records(project_id)
        sections = {(str(item["section_id"]), str(item["section_name"])) for item in records}
        return sorted(sections, key=lambda item: item[0].casefold())

    def scan(
        self,
        project_id: str,
        section_ids: Iterable[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        content_type: str = "all",
        minimum_characters: int = 200,
    ) -> list[SourceSample]:
        selected_sections = set(section_ids or [])
        normalized_type = content_type.strip().lower()
        if normalized_type not in CONTENT_TYPES:
            raise ProjectError(f"不支援的內容類型：{content_type}")
        if minimum_characters < 1:
            raise ProjectError("最小字數必須是正整數")
        if page_from is not None and page_from < 1:
            raise ProjectError("起始頁碼必須大於 0")
        if page_to is not None and page_to < 1:
            raise ProjectError("結束頁碼必須大於 0")
        if page_from is not None and page_to is not None and page_from > page_to:
            raise ProjectError("起始頁碼不可大於結束頁碼")

        samples: list[SourceSample] = []
        for record in self._records(project_id):
            text = str(record["text"]).strip()
            page = int(record["page"])
            record_type = self._classify(text)
            if selected_sections and str(record["section_id"]) not in selected_sections:
                continue
            if page_from is not None and page < page_from:
                continue
            if page_to is not None and page > page_to:
                continue
            if len(text) < minimum_characters or self._is_table_of_contents(text):
                continue
            if normalized_type != "all" and record_type != normalized_type:
                continue
            samples.append(
                SourceSample(
                    sample_id=str(record["chunk_id"]),
                    chunk_id=str(record["chunk_id"]),
                    project_id=str(record["project_id"]),
                    section_id=str(record["section_id"]),
                    section_name=str(record["section_name"]),
                    document_id=str(record["document_id"]),
                    page=page,
                    block_id=str(record["block_id"]),
                    content_type=record_type,
                    character_count=len(text),
                    text=text,
                )
            )
        return sorted(samples, key=lambda item: (item.document_id.casefold(), item.page, item.block_id))

    def sample(
        self,
        project_id: str,
        count: int,
        section_ids: Iterable[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        content_type: str = "all",
        minimum_characters: int = 200,
        seed: int = 42,
    ) -> SourceSampleBatch:
        if count < 1:
            raise ProjectError("取樣數量必須是正整數")
        candidates = self.scan(
            project_id,
            section_ids,
            page_from,
            page_to,
            content_type,
            minimum_characters,
        )
        if len(candidates) < count:
            raise ProjectError(f"符合條件的原文只有 {len(candidates)} 筆，少於要求的 {count} 筆")
        selected = random.Random(seed).sample(candidates, count)
        selected.sort(key=lambda item: (item.document_id.casefold(), item.page, item.block_id))
        now = datetime.now(timezone.utc).isoformat()
        batch = SourceSampleBatch(
            sample_batch_id=uuid.uuid4().hex,
            project_id=project_id,
            created_at=now,
            section_ids=tuple(sorted(set(section_ids or []))),
            page_from=page_from,
            page_to=page_to,
            content_type=content_type,
            minimum_characters=minimum_characters,
            seed=seed,
            requested_count=count,
            samples=tuple(selected),
        )
        self._write_batch(batch)
        return batch

    def get(self, project_id: str, sample_batch_id: str) -> SourceSampleBatch:
        if not sample_batch_id.isalnum():
            raise ProjectError("取樣批次 ID 格式不正確")
        path = self.projects.path_for(project_id) / "runs" / f"source-samples-{sample_batch_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError(f"找不到取樣批次：{sample_batch_id}") from exc
        samples = tuple(SourceSample(**item) for item in value.pop("samples"))
        value["section_ids"] = tuple(value.get("section_ids", []))
        return SourceSampleBatch(**value, samples=samples)

    def export(self, project_id: str, sample_batch_id: str) -> tuple[Path, Path]:
        batch = self.get(project_id, sample_batch_id)
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / f"source-samples-{sample_batch_id}.json"
        csv_path = directory / f"source-samples-{sample_batch_id}.csv"
        self._atomic_text(json_path, json.dumps(asdict(batch), ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(SourceSample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in batch.samples)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    def _records(self, project_id: str) -> list[dict[str, Any]]:
        path = self.projects.path_for(project_id) / "processed" / "input.jsonl"
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except FileNotFoundError as exc:
            raise ProjectError("找不到前處理原文，請先執行 PDF 前處理") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError("前處理原文格式錯誤") from exc
        required = {
            "chunk_id",
            "project_id",
            "section_id",
            "section_name",
            "document_id",
            "page",
            "block_id",
            "text",
        }
        for index, record in enumerate(records):
            missing = required - set(record)
            if missing:
                raise ProjectError(
                    f"前處理原文第 {index + 1} 筆缺少 Metadata：{', '.join(sorted(missing))}；請重新前處理"
                )
        return records

    @staticmethod
    def _classify(text: str) -> str:
        value = text.casefold()
        keywords = {
            "diagnostic": ("diagnos", "symptom", "dtc", "trouble", "inspection", "check", "故障", "診斷", "檢查"),
            "procedure": ("removal", "installation", "procedure", "step", "caution", "warning", "拆卸", "安裝", "步驟"),
            "specification": ("specification", "service data", "torque", "tolerance", "標準", "扭力", "規格"),
        }
        scores = {kind: sum(value.count(keyword) for keyword in values) for kind, values in keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    @staticmethod
    def _is_table_of_contents(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return True
        dotted_lines = sum(bool(re.search(r"\.{4,}\s*\d*\s*$", line)) for line in lines)
        contents_heading = any(line.casefold() in {"contents", "table of contents", "目錄"} for line in lines[:5])
        return contents_heading and dotted_lines / len(lines) >= 0.2

    def _write_batch(self, batch: SourceSampleBatch) -> None:
        path = self.projects.path_for(batch.project_id) / "runs" / f"source-samples-{batch.sample_batch_id}.json"
        self._atomic_text(path, json.dumps(asdict(batch), ensure_ascii=False, indent=2) + "\n")

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
