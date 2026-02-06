from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.settings import settings
from backend import schemas


PROJECT_META_PATH = Path(__file__).parent.parent / "data" / "project.json"
EXPORT_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_project_meta() -> dict:
    if PROJECT_META_PATH.exists():
        try:
            meta = json.loads(PROJECT_META_PATH.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("project meta must be an object")
            return _hydrate_project_meta(meta)
        except Exception:
            pass
    meta = {"version": EXPORT_VERSION, "name": "default", "created_at": _now()}
    PROJECT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_META_PATH.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return _hydrate_project_meta(meta)


def _hydrate_project_meta(meta: dict, *, strict: bool = False) -> dict:
    """Return meta dict with safe defaults for known fields.

    - Preserves unknown keys (forward compatibility)
    - Adds reviewer/graph fields for legacy project.json files
    - Normalizes reviewer identity values
    """
    meta = dict(meta or {})
    meta.setdefault("version", EXPORT_VERSION)
    meta["name"] = (meta.get("name") or "default").strip() or "default"
    meta.setdefault("created_at", _now())

    # Defaults for Phase 09 fields (do not overwrite existing).
    meta.setdefault("reviewers", [])
    meta.setdefault("active_reviewer_uid", None)
    meta.setdefault("compare_reviewer_a", None)
    meta.setdefault("compare_reviewer_b", None)
    meta.setdefault("graph_settings", {})

    try:
        # Normalize known fields using the schema validators, without dropping
        # unknown keys.
        normalized = schemas.ProjectMeta(
            name=meta.get("name") or "default",
            version=int(meta.get("version") or EXPORT_VERSION),
            created_at=meta.get("created_at"),
            updated_at=meta.get("updated_at"),
            reviewers=meta.get("reviewers"),
            active_reviewer_uid=meta.get("active_reviewer_uid"),
            compare_reviewer_a=meta.get("compare_reviewer_a"),
            compare_reviewer_b=meta.get("compare_reviewer_b"),
            graph_settings=meta.get("graph_settings"),
        ).model_dump(mode="json")
        for key in (
            "version",
            "name",
            "created_at",
            "updated_at",
            "reviewers",
            "active_reviewer_uid",
            "compare_reviewer_a",
            "compare_reviewer_b",
            "graph_settings",
        ):
            meta[key] = normalized.get(key)
    except Exception:
        if strict:
            raise

        def _norm(value: Any) -> str:
            text = str(value or "").strip()
            return " ".join(text.split())

        reviewers = meta.get("reviewers")
        if not isinstance(reviewers, list):
            reviewers = []
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in reviewers:
            text = _norm(raw)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        meta["reviewers"] = cleaned

        for key in ("active_reviewer_uid", "compare_reviewer_a", "compare_reviewer_b"):
            value = meta.get(key)
            if value is None:
                meta[key] = None
                continue
            text = _norm(value)
            meta[key] = text or None

        if not isinstance(meta.get("graph_settings"), dict):
            meta["graph_settings"] = {}

    _ensure_reviewer_membership(meta)
    return meta


def _ensure_reviewer_membership(meta: dict) -> None:
    reviewers = meta.get("reviewers")
    if not isinstance(reviewers, list):
        reviewers = []
    seen = {str(v).casefold() for v in reviewers if isinstance(v, str)}
    for key in ("active_reviewer_uid", "compare_reviewer_a", "compare_reviewer_b"):
        uid = meta.get(key)
        if not isinstance(uid, str) or not uid.strip():
            continue
        k = uid.casefold()
        if k in seen:
            continue
        reviewers.append(uid)
        seen.add(k)
    meta["reviewers"] = reviewers


def merge_project_meta(existing: dict, patch: dict) -> dict:
    """Merge a partial project meta patch without clobbering unknown keys."""
    merged = dict(existing or {})
    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")

    # Shallow merge, with special handling for graph_settings to preserve
    # unknown settings keys.
    for key, value in patch.items():
        if (
            key == "graph_settings"
            and isinstance(value, dict)
            and isinstance(merged.get("graph_settings"), dict)
        ):
            merged["graph_settings"] = {**(merged.get("graph_settings") or {}), **value}
        else:
            merged[key] = value

    merged.setdefault("version", EXPORT_VERSION)
    merged["name"] = (merged.get("name") or "default").strip() or "default"
    merged.setdefault("created_at", _now())
    merged["updated_at"] = _now()

    return _hydrate_project_meta(merged, strict=True)


def write_project_meta_update(patch: dict) -> dict:
    meta = read_project_meta()
    meta = merge_project_meta(meta, patch)
    PROJECT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_META_PATH.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def write_project_meta(*, name: str) -> dict:
    """Backward-compatible helper: update project name only."""
    return write_project_meta_update({"name": name})


def _paths_to_export() -> list[Path]:
    root = Path(__file__).parent.parent / "data"
    # Prefer explicit settings paths so future project scoping can reuse this.
    candidates = [
        PROJECT_META_PATH,
        Path(settings.GRAPH_DB_PATH),
        Path(settings.CLAIM_DB_PATH),
        Path(settings.INGESTION_DIR),
        Path(settings.ATTACHMENT_DIR),
        Path(settings.EVIDENCE_STORE_DIR),
        root / "judgments",
    ]
    return candidates


def export_project_zip() -> bytes:
    meta = read_project_meta()
    root = Path(__file__).parent.parent
    data_root = root / "data"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "project.json",
            json.dumps(
                {**meta, "exported_at": _now(), "export_version": EXPORT_VERSION},
                indent=2,
                sort_keys=True,
            ),
        )
        for path in _paths_to_export():
            if not path.exists():
                continue
            if path.is_file():
                arc = (
                    str(path.relative_to(data_root))
                    if data_root in path.parents
                    else str(path.name)
                )
                zf.write(path, arcname=arc)
                continue
            if path.is_dir():
                for child in path.rglob("*"):
                    if not child.is_file():
                        continue
                    try:
                        arc = str(child.relative_to(data_root))
                    except ValueError:
                        arc = str(child.relative_to(path.parent))
                    zf.write(child, arcname=arc)
    return buf.getvalue()


def _safe_extract(zf: zipfile.ZipFile, *, target_root: Path) -> None:
    target_root = target_root.resolve()
    for member in zf.infolist():
        name = member.filename
        if not name or name.endswith("/"):
            continue
        dest = (target_root / name).resolve()
        if target_root not in dest.parents and dest != target_root:
            raise RuntimeError(f"Unsafe zip path: {name}")
    zf.extractall(path=target_root)


def import_project_zip(
    zip_bytes: bytes,
    *,
    overwrite: bool,
    backup_dir: Optional[Path] = None,
) -> dict:
    if not overwrite:
        raise RuntimeError("overwrite must be true")

    data_root = (Path(__file__).parent.parent / "data").resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    # Backup current state.
    backup_dir = backup_dir or (data_root / "backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = (
        backup_dir / f"project-backup-{_now().replace(':', '').replace('-', '')}.zip"
    )
    backup_path.write_bytes(export_project_zip())

    # Clear existing stores (keep backups).
    for path in [
        Path(settings.GRAPH_DB_PATH),
        Path(settings.CLAIM_DB_PATH),
    ]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    for folder in [
        Path(settings.INGESTION_DIR),
        Path(settings.ATTACHMENT_DIR),
        Path(settings.EVIDENCE_STORE_DIR),
        data_root / "judgments",
    ]:
        try:
            if folder.exists():
                shutil.rmtree(folder)
        except Exception:
            pass
    # Recreate roots.
    Path(settings.INGESTION_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.ATTACHMENT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.EVIDENCE_STORE_DIR).mkdir(parents=True, exist_ok=True)
    (data_root / "judgments").mkdir(parents=True, exist_ok=True)

    # Extract.
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        _safe_extract(zf, target_root=data_root)

    imported_meta = read_project_meta()
    return {"ok": True, "backup_zip": str(backup_path), "project": imported_meta}
