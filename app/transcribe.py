# -*- coding: utf-8 -*-
"""音频抽取（ffmpeg）+ faster-whisper 本地转写（GPU 优先，自动回退 CPU）。"""
from __future__ import annotations

import subprocess
import threading
import warnings
from pathlib import Path
from typing import Callable

from .config import ROOT
from .utils import resolve_model_path

# ---------------------------------------------------------------- ffmpeg ---

def probe_duration(path: str) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return None


def extract_audio(media: str, out_wav: Path, on_log=lambda m: None) -> None:
    """任意媒体 -> 16kHz 单声道 wav。"""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", "-loglevel", "error", str(out_wav)]
    on_log("ffmpeg 抽取音频: " + " ".join(cmd[:8]) + " …")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError("ffmpeg 抽取音频失败:\n" + "\n".join(tail))
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError("ffmpeg 未产出音频文件（可能没有音轨）")
    on_log(f"音频抽取完成: {out_wav.name} ({out_wav.stat().st_size/1e6:.1f} MB)")


# ------------------------------------------------------------- whisper ---

_model_cache: dict[tuple, object] = {}
_model_lock = threading.Lock()


def _resolve_device_compute(asr_cfg: dict, on_log) -> tuple[str, str]:
    device = (asr_cfg.get("device") or "auto").lower()
    compute = (asr_cfg.get("compute_type") or "auto").lower()
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    on_log(f"ASR 设备: {device} ({compute})")
    return device, compute


def load_model(asr_cfg: dict, on_log=lambda m: None):
    """加载（或复用）faster-whisper 模型。返回 (model, device)。"""
    model_ref = asr_cfg.get("model") or "faster-whisper-large-v3"
    device, compute = _resolve_device_compute(asr_cfg, on_log)
    path = resolve_model_path(model_ref)
    key = (path, device, compute)
    with _model_lock:
        if key in _model_cache:
            on_log(f"复用已加载模型: {Path(path).name} [{device}]")
            return _model_cache[key], device
        warnings.filterwarnings("ignore")
        from faster_whisper import WhisperModel
        on_log(f"加载模型 {Path(path).name} → {device}({compute})，首次需十几秒 …")
        import time
        t0 = time.time()
        m = WhisperModel(path, device=device, compute_type=compute)
        on_log(f"模型加载完成（{time.time()-t0:.1f}s）")
        _model_cache[key] = m
        return m, device


class CancelledError(RuntimeError):
    pass


def transcribe(
    wav_path: Path,
    asr_cfg: dict,
    language: str,
    cancel: threading.Event | None = None,
    on_log: Callable[[str], None] = lambda m: None,
    on_live: Callable[[dict], None] | None = None,
) -> dict:
    """转写 wav -> {"language", "duration", "segments":[{"start","end","text"}]}
    language: "auto" / "zh" / "en" ...
    on_live: 每约 2% 进度回调一次，参数 {"pct","speed","remain","text"} 供界面实时展示。"""
    model, device = load_model(asr_cfg, on_log)
    kwargs = dict(
        beam_size=int(asr_cfg.get("beam_size", 5)),
        vad_filter=bool(asr_cfg.get("vad_filter", True)),
        vad_parameters={"min_silence_duration_ms": 700},
    )
    if language and language != "auto":
        kwargs["language"] = language
    prompt = (asr_cfg.get("initial_prompt") or "").strip()
    if prompt:
        kwargs["initial_prompt"] = prompt

    on_log(f"开始转写（vad={kwargs['vad_filter']}, beam={kwargs['beam_size']}, "
           f"lang={kwargs.get('language', '自动检测')}）…")
    segs_iter, info = model.transcribe(str(wav_path), **kwargs)

    segments = []
    total = info.duration or 0.0
    last_pct = -1
    import time
    t0 = time.time()
    for seg in segs_iter:
        if cancel is not None and cancel.is_set():
            raise CancelledError("已取消")
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
        if total > 0:
            pct = int(seg.end / total * 100)
            if pct != last_pct and pct % 2 == 0:
                last_pct = pct
                elapsed = max(time.time() - t0, 1e-6)
                speed = seg.end / elapsed
                remain = (total - seg.end) / max(speed, 1e-6)
                on_log(f"转写进度 {pct}%  ({int(seg.end)}s/{int(total)}s, {speed:.1f}x 实时)")
                if on_live:
                    on_live({"pct": pct, "speed": round(speed, 1),
                             "remain": int(remain), "text": text})
    if not segments:
        raise RuntimeError("未识别到任何语音内容（音轨可能是静音或纯音乐）")
    on_log(f"转写完成：{len(segments)} 段，语言 {info.language}，"
           f"耗时 {time.time()-t0:.0f}s（音频 {total:.0f}s）")
    return {"language": info.language, "duration": info.duration, "segments": segments}
