"""Local filesystem browser used by the agent import flow.

Limited to a single user-configured root (defaults to ``$HOME``) to avoid the
endpoint becoming a general filesystem peek. The frontend exposes this as a
file picker dialog scoped to checkpoint files.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from rl_trading_playground.api.settings import get_settings


router = APIRouter(prefix="/files", tags=["files"])


class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None
    modified: datetime | None = None


class BrowseResponse(BaseModel):
    root: str
    path: str
    parent: str | None
    entries: list[FileEntry]


def _allowed_root() -> Path:
    settings = get_settings()
    # repo root + user home are both reasonable. Settings.store_root is under repo.
    return settings.store_root.parent.parent.resolve()


def _safe_resolve(raw: str | None) -> Path:
    root = _allowed_root()
    home = Path.home().resolve()
    candidate = Path(raw).expanduser().resolve() if raw else root
    for allowed in (root, home):
        try:
            candidate.relative_to(allowed)
            return candidate
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"path must live under {root} or {home}")


@router.get("/browse", response_model=BrowseResponse)
def browse(
    path: str | None = Query(None, description="Directory to list; defaults to repo root."),
    ext: str | None = Query(
        None,
        description="File extension filter; comma-separated for multiple (e.g. '.ptm,.pt,.pth').",
    ),
) -> BrowseResponse:
    exts = {e.strip() for e in ext.split(",") if e.strip()} if ext else None
    target = _safe_resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="path does not exist")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")

    entries: list[FileEntry] = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        is_dir = child.is_dir()
        if not is_dir and exts and child.suffix not in exts:
            continue
        try:
            stat = child.stat()
            size = None if is_dir else stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except OSError:
            size, modified = None, None
        entries.append(
            FileEntry(
                name=child.name,
                path=str(child),
                is_dir=is_dir,
                size=size,
                modified=modified,
            )
        )

    parent = str(target.parent) if target.parent != target else None
    return BrowseResponse(
        root=str(_allowed_root()),
        path=str(target),
        parent=parent,
        entries=entries,
    )
