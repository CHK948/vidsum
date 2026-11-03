# -*- coding: utf-8 -*-
"""通用工具：时间格式化、SRT 生成、模型路径解析、媒体文件探测。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

MEDIA_EXTS = {
    ".mp4", ".mkv", ".ts", ".flv", ".avi", ".mov", ".wmv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".rmvb", ".rm", ".vob", ".m2ts", ".mts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}

VIDEO_EXTS = {e for e in MEDIA_EXTS if e not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}}
AUDIO_EXTS = MEDIA_EXTS - VIDEO_EXTS


def fmt_ts(seconds: float, comma: bool = False) -> str:
    """秒 -> HH:MM:SS,mmm （SRT 用逗号，正文用句点）"""
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_clock(seconds: float) -> str:
    """秒 -> MM:SS 或 HH:MM:SS（转写稿时间戳用）"""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}小时{m:02d}分" if h else f"{m}分{s:02d}秒"


def build_srt(segments: Iterable[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{fmt_ts(seg['start'], True)} --> {fmt_ts(seg['end'], True)}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def build_txt(segments: Iterable[dict]) -> str:
    lines = []
    for seg in segments:
        t = seg["text"].strip()
        if t:
            lines.append(f"[{fmt_clock(seg['start'])}] {t}")
    return "\n".join(lines) + "\n"


def resolve_model_path(name_or_path: str) -> str:
    """把模型名解析为本地路径；本地缓存不存在则抛错（默认完全离线）。"""
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return str(p)
    # 在常见 HF 缓存目录中查找 models--*--<name>
    candidates = []
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    candidates.append(Path(hf_home) / "hub")
    candidates.append(Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") else None)
    needle = name_or_path.replace("/", "--").lower()
    seen = set()
    for base in candidates:
        if not base or not base.is_dir() or str(base) in seen:
            continue
        seen.add(str(base))
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.lower().startswith("models--") and needle in d.name.lower():
                snaps = sorted((d / "snapshots").glob("*"))
                snaps = [s for s in snaps if s.is_dir()]
                if snaps:
                    return str(snaps[-1])
    raise FileNotFoundError(
        f"本地未找到模型 {name_or_path!r}（已搜索 {hf_home}/hub）。"
        f"可在设置中填入模型的本地目录绝对路径。"
    )


def find_local_models() -> list[str]:
    """扫描 HF 缓存，列出本地已有的 whisper(ct2) 模型名。"""
    found = []
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = Path(hf_home) / "hub"
    if hub.is_dir():
        for d in sorted(hub.iterdir()):
            n = d.name
            if d.is_dir() and n.startswith("models--") and "whisper" in n.lower():
                if list((d / "snapshots").glob("*")):
                    m = re.match(r"models--[^-]+--(.+)", n)
                    found.append(m.group(1) if m else n)
    return found


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}PB"


def is_media_file(name: str) -> bool:
    return Path(name).suffix.lower() in MEDIA_EXTS
