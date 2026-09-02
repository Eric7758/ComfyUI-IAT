from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import folder_paths
from aiohttp import web
from server import PromptServer


IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def _resolve_output_path(relative_path: str) -> tuple[Path, Path]:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    requested = (output_root / relative_path).resolve()
    if requested != output_root and output_root not in requested.parents:
        raise ValueError("Path is outside the ComfyUI output directory")
    return output_root, requested


def _relative_posix(path: Path, output_root: Path) -> str:
    relative = path.relative_to(output_root)
    return "" if relative == Path(".") else relative.as_posix()


def _view_url(path: Path, output_root: Path) -> str:
    subfolder = _relative_posix(path.parent, output_root)
    return (
        f"/view?filename={quote(path.name)}"
        f"&subfolder={quote(subfolder)}&type=output"
    )


def _directory_item(path: Path, output_root: Path) -> dict:
    stat = path.stat()
    item_count = 0
    try:
        item_count = sum(1 for _ in path.iterdir())
    except OSError:
        pass
    return {
        "name": path.name,
        "path": _relative_posix(path, output_root),
        "item_count": item_count,
        "created": int(getattr(stat, "st_birthtime", stat.st_ctime) * 1000),
        "modified": int(stat.st_mtime * 1000),
    }


def _file_item(path: Path, output_root: Path) -> dict:
    stat = path.stat()
    extension = path.suffix.lower()
    kind = "other"
    if extension in IMAGE_EXTENSIONS:
        kind = "image"
    elif extension in VIDEO_EXTENSIONS:
        kind = "video"
    elif extension in AUDIO_EXTENSIONS:
        kind = "audio"
    return {
        "name": path.name,
        "path": _relative_posix(path, output_root),
        "kind": kind,
        "size": stat.st_size,
        "created": int(getattr(stat, "st_birthtime", stat.st_ctime) * 1000),
        "modified": int(stat.st_mtime * 1000),
        "url": _view_url(path, output_root),
    }


def _search_output(output_root: Path, query: str) -> tuple[list[dict], list[dict]]:
    directories = []
    files = []
    folded_query = query.casefold()
    for current_root, directory_names, file_names in os.walk(
        output_root, topdown=True, followlinks=False, onerror=lambda _: None
    ):
        current_directory = Path(current_root)
        safe_directories = []
        for name in directory_names:
            candidate = current_directory / name
            try:
                resolved = candidate.resolve()
                if resolved != output_root and output_root not in resolved.parents:
                    continue
                if candidate.is_symlink() or not resolved.is_dir():
                    continue
                safe_directories.append(name)
                if folded_query in name.casefold():
                    directories.append(_directory_item(resolved, output_root))
            except (OSError, ValueError):
                continue
        directory_names[:] = safe_directories
        for name in file_names:
            candidate = current_directory / name
            try:
                if folded_query not in name.casefold():
                    continue
                resolved = candidate.resolve()
                if resolved != output_root and output_root not in resolved.parents:
                    continue
                if resolved.is_file():
                    files.append(_file_item(resolved, output_root))
            except (OSError, ValueError):
                continue
    return directories, files


@PromptServer.instance.routes.get("/iat/output-browser")
async def browse_output(request: web.Request) -> web.Response:
    relative_path = request.query.get("path", "")
    search_query = request.query.get("search", "").strip()
    try:
        output_root, directory = _resolve_output_path(relative_path)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)

    if not directory.is_dir():
        return web.json_response({"error": "Output folder was not found"}, status=404)

    directories = []
    files = []
    if search_query:
        directories, files = _search_output(output_root, search_query)
        return web.json_response(
            {
                "path": _relative_posix(directory, output_root),
                "parent": "",
                "directories": directories,
                "files": files,
                "search": search_query,
            }
        )

    try:
        children = list(directory.iterdir())
        for child in children:
            try:
                resolved = child.resolve()
                if resolved != output_root and output_root not in resolved.parents:
                    continue
                if resolved.is_dir():
                    directories.append(_directory_item(resolved, output_root))
                    continue

                if not resolved.is_file():
                    continue
                files.append(_file_item(resolved, output_root))
            except (OSError, ValueError):
                continue
    except OSError as error:
        return web.json_response({"error": str(error)}, status=500)

    directories.sort(key=lambda item: item["name"].casefold())
    files.sort(key=lambda item: (-item["created"], item["name"].casefold()))
    current_path = _relative_posix(directory, output_root)
    parent = ""
    if current_path:
        parent = _relative_posix(directory.parent, output_root)

    return web.json_response(
        {
            "path": current_path,
            "parent": parent,
            "directories": directories,
            "files": files,
        }
    )
