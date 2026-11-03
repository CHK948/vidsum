# -*- coding: utf-8 -*-
"""配置读写：config.json + 环境变量覆盖。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    "server": {"host": "127.0.0.1", "port": 8322},
    "asr": {
        "model": "faster-whisper-large-v3",
        "device": "auto",          # auto | cuda | cpu
        "compute_type": "auto",    # auto | float16 | int8 ...
        "vad_filter": True,
        "beam_size": 5,
        "initial_prompt": "",
    },
    "llm": {
        "api_key": "",             # 也可用环境变量 ZHIPUAI_API_KEY / GLM_API_KEY
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5.3-flash",
        "temperature": 0.6,
        "top_p": 0.95,
        "timeout": 600,
        "max_retries": 3,
        "chunk_chars": 6000,
        "max_workers": 2,
    },
    "outputs_dir": "outputs",
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg = _merge(cfg, json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[config] 读取 config.json 失败，使用默认配置: {e}")
    # 环境变量覆盖 API key
    for var in ("ZHIPUAI_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY"):
        v = os.environ.get(var, "").strip()
        if v:
            cfg["llm"]["api_key"] = v
            break
    return cfg


def save(cfg: dict) -> None:
    """仅保存与默认值可 JSON 化的部分；保证目录存在。"""
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def outputs_dir(cfg: dict) -> Path:
    p = Path(cfg.get("outputs_dir") or "outputs")
    return p if p.is_absolute() else (ROOT / p)


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "****" + key[-4:]
