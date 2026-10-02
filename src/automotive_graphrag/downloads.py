"""Stage generated exports in a Gradio-safe temporary download directory."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from .projects import ProjectError


def stage_downloads(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    raw_sources = [Path(path) for path in paths]
    sources = [source.resolve() for source in raw_sources]
    if not sources:
        raise ProjectError("沒有可下載的匯出檔案")
    for raw_source, source in zip(raw_sources, sources, strict=True):
        if raw_source.is_symlink() or not source.is_file():
            raise ProjectError(f"找不到可下載的匯出檔案：{source.name}")
    directory = Path(tempfile.mkdtemp(prefix="automotive-graphrag-download-"))
    staged: list[Path] = []
    try:
        for source in sources:
            destination = directory / source.name
            shutil.copy2(source, destination)
            staged.append(destination)
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise ProjectError("無法準備下載檔案") from exc
    return tuple(staged)
