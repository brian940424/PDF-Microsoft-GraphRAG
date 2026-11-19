"""Export a whole project workspace (including built graph files) to a zip and import it back."""

from __future__ import annotations

import json
import os
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
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3


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
                if not include_cache and relative.parts[:2] == CACHE_DIRECTORY.parts:
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


def import_project(store: ProjectStore, archive_path: str | Path) -> Project:
    """Restore a project exported by `export_project`; refuses to overwrite an existing project."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectError("無法讀取專案檔，請確認是本系統匯出的 .zip 檔") from exc
    with archive:
        manifest = _read_manifest(archive)
        project_id = str(manifest.get("project_id") or "")
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ProjectError("專案檔中的 Project ID 格式不正確")
        destination = store.root / project_id
        if destination.exists():
            raise ProjectError(f"Project ID 已存在：{project_id}，請先刪除或改名既有專案後再匯入")
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
            try:
                os.rename(staged_project, destination)
            except OSError as exc:
                if destination.exists():
                    raise ProjectError(f"Project ID 已存在：{project_id}") from exc
                raise
        except OSError as exc:
            raise ProjectError(f"無法匯入專案：{exc}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return store.get(project_id)
