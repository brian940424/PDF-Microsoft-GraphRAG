"""Export a whole project workspace (including built graph files) to a zip and import it back."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .projects import (
    INDEXING_LOCK_FILE,
    PROJECT_DIRECTORIES,
    PROJECT_ID_PATTERN,
    Project,
    ProjectError,
    ProjectStore,
    project_indexing_locked,
)

ARCHIVE_FORMAT = "automotive-graphrag-project"
ARCHIVE_VERSION = 1
MANIFEST_NAME = "graphrag-project-export.json"
ARCHIVE_SUFFIX = ".graphrag-project.zip"
BUSY_STATUSES = {"INDEXING", "PROCESSING"}
# Per-project secrets and transient lock files never leave the machine.
EXCLUDED_NAMES = {".env", INDEXING_LOCK_FILE}
CACHE_DIRECTORY = PurePosixPath("graphrag", "cache")
CACHE_DIRECTORIES = (CACHE_DIRECTORY, PurePosixPath("cache"))
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
# Files whose `project_id` fields are rewritten when a project is imported under a new ID.
REWRITTEN_SUFFIXES = {".json", ".jsonl"}


def export_project(store: ProjectStore, project_id: str, *, include_cache: bool = False) -> Path:
    """Write `project_id` to a zip in a fresh temporary directory and return its path."""
    project = store.get(project_id)
    project_path = store.path_for(project_id)
    if project.status in BUSY_STATUSES or project_indexing_locked(project_path):
        raise ProjectError("專案正在前處理或建圖，請完成後再匯出")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = Path(tempfile.mkdtemp(prefix="automotive-graphrag-project-export-"))
    archive_path = directory / f"{project.project_id}-{timestamp}{ARCHIVE_SUFFIX}"
    manifest = {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "project_id": project.project_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "include_cache": include_cache,
    }
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
            for path in sorted(project_path.rglob("*")):
                relative = PurePosixPath(path.relative_to(project_path).as_posix())
                if path.is_symlink() or not path.is_file() or path.name in EXCLUDED_NAMES:
                    continue
                if not include_cache and any(relative.parts[: len(item.parts)] == item.parts for item in CACHE_DIRECTORIES):
                    continue
                archive.write(path, f"{project.project_id}/{relative}")
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise ProjectError(f"無法匯出專案：{exc}") from exc
    return archive_path


def _read_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    except KeyError as exc:
        raise ProjectError("不是本系統匯出的專案檔（缺少匯出資訊）") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError("專案匯出資訊無法解析") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != ARCHIVE_FORMAT:
        raise ProjectError("不是本系統匯出的專案檔")
    if manifest.get("version") != ARCHIVE_VERSION:
        raise ProjectError(f"不支援的專案匯出版本：{manifest.get('version')}")
    return manifest


def _validated_members(archive: zipfile.ZipFile, project_id: str) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    total = 0
    for info in archive.infolist():
        if info.filename == MANIFEST_NAME:
            continue
        name = PurePosixPath(info.filename)
        if (
            "\\" in info.filename
            or name.is_absolute()
            or ".." in name.parts
            or not name.parts
            or name.parts[0] != project_id
        ):
            raise ProjectError(f"專案檔包含不安全的路徑：{info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ProjectError(f"專案檔不可包含符號連結：{info.filename}")
        if name.name in EXCLUDED_NAMES:
            continue
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ProjectError("專案檔解壓後過大")
        members.append(info)
    return members


def _rewrite_project_id_fields(project_path: Path, old_id: str, new_id: str) -> None:
    """Point every `"project_id": "<old_id>"` field in the project's JSON/JSONL files at `new_id`.

    Lines are rewritten in place so file formatting is preserved. Chunk and document IDs that merely
    start with the old ID are left alone: they are opaque keys shared with the built graph files.
    """
    pattern = re.compile(rb'("project_id"\s*:\s*)"' + re.escape(old_id.encode()) + rb'"')
    replacement = rb'\g<1>"' + new_id.encode() + rb'"'
    for path in project_path.rglob("*"):
        if path.suffix not in REWRITTEN_SUFFIXES or path.is_symlink() or not path.is_file():
            continue
        rewritten = path.with_name(path.name + ".rewrite")
        changed = False
        with path.open("rb") as source, rewritten.open("wb") as output:
            for line in source:
                new_line = pattern.sub(replacement, line)
                changed = changed or new_line != line
                output.write(new_line)
        if changed:
            os.replace(rewritten, path)
        else:
            rewritten.unlink()


def import_project(store: ProjectStore, archive_path: str | Path, *, project_id: str | None = None) -> Project:
    """Restore a project exported by `export_project`; refuses to overwrite an existing project.

    `project_id` imports the project under a new Project ID instead of the one stored in the archive.
    """
    new_id = (project_id or "").strip()
    if new_id and not PROJECT_ID_PATTERN.fullmatch(new_id):
        raise ProjectError("新的 Project ID 只能包含英文、數字、底線與連字號")
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectError("無法讀取專案檔，請確認是本系統匯出的 .zip 檔") from exc
    with archive:
        manifest = _read_manifest(archive)
        project_id = str(manifest.get("project_id") or "")
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ProjectError("專案檔中的 Project ID 格式不正確")
        target_id = new_id or project_id
        destination = store.root / target_id
        if destination.exists():
            if new_id:
                raise ProjectError(f"Project ID 已存在：{target_id}，請換一個新的 Project ID")
            raise ProjectError(f"Project ID 已存在：{target_id}，請在「新的 Project ID」輸入其他名稱後再匯入")
        members = _validated_members(archive, project_id)
        try:
            metadata = json.loads(archive.read(f"{project_id}/project.json").decode("utf-8"))
        except KeyError as exc:
            raise ProjectError("專案檔缺少 project.json") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectError("專案檔中的 project.json 無法解析") from exc
        project = Project.from_dict(metadata)
        if project.project_id != project_id:
            raise ProjectError("專案檔中的 Project ID 與 project.json 不一致")
        if project.status in BUSY_STATUSES:
            raise ProjectError("專案檔是在前處理或建圖途中匯出的，無法匯入")

        store.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".import-", dir=store.root))
        try:
            for info in members:
                target = staging / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            staged_project = staging / project_id
            for directory in PROJECT_DIRECTORIES:
                (staged_project / directory).mkdir(parents=True, exist_ok=True)
            if target_id != project_id:
                _rewrite_project_id_fields(staged_project, project_id, target_id)
            try:
                os.rename(staged_project, destination)
            except OSError as exc:
                if destination.exists():
                    raise ProjectError(f"Project ID 已存在：{target_id}") from exc
                raise
        except OSError as exc:
            raise ProjectError(f"無法匯入專案：{exc}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return store.get(target_id)
