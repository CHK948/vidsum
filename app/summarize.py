# -*- coding: utf-8 -*-
"""GLM 大模型总结：分块 map + 汇总 reduce，支持断点续传与取消。"""
from __future__ import annotations

import concurrent.futures as futures
import time
from pathlib import Path
from typing import Callable

import requests

from .utils import fmt_dur

CANCELLED = object()


def _fmt(tpl: str, **kw) -> str:
    """安全的模板填充：显式替换 {key}，模板中的其它花括号（如 LaTeX 示例）不受影响。"""
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


class LLMAuthError(Exception):
    pass


def _chat(cfg_llm: dict, messages: list[dict], cancel: "futures.Event | None" = None) -> str:
    """调用 GLM Chat Completion，带重试；返回正文文本。"""
    api_key = (cfg_llm.get("api_key") or "").strip()
    if not api_key:
        raise LLMAuthError("未配置 GLM API Key：请打开页面右上角“设置”填入智谱 API Key，"
                           "或设置环境变量 ZHIPUAI_API_KEY。")
    url = cfg_llm["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg_llm.get("model") or "glm-5.3-flash",
        "messages": messages,
        "temperature": cfg_llm.get("temperature", 0.6),
        "top_p": cfg_llm.get("top_p", 0.95),
        "stream": False,
    }
    timeout = int(cfg_llm.get("timeout", 600))
    retries = int(cfg_llm.get("max_retries", 3))
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("已取消")
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 5 * attempt
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                raise LLMAuthError(f"API Key 无效 (401)：{r.text[:300]}")
            if r.status_code != 200:
                raise RuntimeError(f"GLM API HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            if not text:
                raise RuntimeError(f"GLM 返回空内容: {str(data)[:400]}")
            return text
        except LLMAuthError:
            raise
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"GLM API 调用失败（重试 {retries} 次）: {last_err}")


# ---------------------------------------------------------------- prompt ---

STYLE_DESC = {
    "course": "课程笔记（面向复习：突出知识点、概念定义、论证逻辑与例子，条理清晰）",
    "meeting": "会议纪要（突出议题、讨论要点、结论、待办事项）",
    "general": "内容总结（突出主题脉络、核心观点与结论）",
    "lecture_tech": "技术讲座总结（突出技术原理、方案对比、实践建议）",
}

MAP_PROMPT = """你是一名专业的中文内容整理专家。下面是视频《{title}》第 {i}/{n} 段的语音转写稿（每行以 [时间] 开头）。
请输出这一段的摘要，要求：
1. 使用简体中文，分条列点；
2. 提取本段讲述的主题、论点、关键概念、重要例证与数据；
3. 保留专有名词、人名、作品名、术语的原文写法；
4. 在要点后标注对应时间，如（12:03-15:40）；
5. 只依据转写稿内容，不要编造；口语重复、寒暄请剔除；
6. 数学公式与符号表达式的写法（重要）：行内公式一律用 $...$ 包裹（如 $f(x)=x^2$、$\\{x\\in S:P(x)\\}$），独立成行的公式用 $$...$$ 包裹；禁止使用 \\( \\) 或 \\[ \\] 定界；零散的单个符号（如 ∈、∀、∅）可保留 Unicode 写法；公式内不得出现中文汉字——需要中文说明时写在公式后面的括号里，且不要用 \\text 包裹中文；
7. 直接输出摘要，不要开场白。{extra_note}

转写稿：
{chunk}
"""

REDUCE_PROMPT = """下面是视频《{title}》（时长 {duration}，共 {n} 段）各段语音的摘要，按时间顺序排列。
请将其整合成一份完整的{style}，Markdown 格式，要求：
1. 使用简体中文；
2. 结构固定为：
   ## 一句话总结
   ## 内容脉络（按主题分节，每节给出要点，尽量保留时间戳引用）
   ## 关键要点
   ## 值得记住的话（摘录原话中的金句/重要结论，没有可省略此节）
3. 允许为连贯性做合理衔接，但不得虚构具体事实、数据、人名；
4. 数学公式与符号表达式的写法（重要）：行内公式一律用 $...$ 包裹（如 $f(x)=x^2$、$A\\cup B$、$2^E$），独立成行的公式用 $$...$$ 包裹；禁止使用 \\( \\) 或 \\[ \\] 定界；零散的单个符号（如 ∈、∀、∅、ℵ₀）可保留 Unicode 写法；公式内不得出现中文汉字——需要中文说明时写在公式后面的括号里，且不要用 \\text 包裹中文；
5. 直接输出 Markdown 正文，不要额外解释、不要用代码块包裹全文。{extra}

各段摘要：
{joined}
"""

DIRECT_PROMPT = """下面是视频《{title}》（时长 {duration}）的完整语音转写稿（每行以 [时间] 开头）。
请将其整理成{style}，Markdown 格式，要求：
1. 使用简体中文；
2. 结构固定为：
   ## 一句话总结
   ## 内容脉络（按主题分节，每节给出要点，尽量保留时间戳引用）
   ## 关键要点
   ## 值得记住的话（摘录原话中的金句/重要结论，没有可省略此节）
3. 只依据转写稿内容，不得虚构事实、数据、人名；剔除口语冗余；
4. 数学公式与符号表达式的写法（重要）：行内公式一律用 $...$ 包裹（如 $f(x)=x^2$、$A\\cup B$、$2^E$），独立成行的公式用 $$...$$ 包裹；禁止使用 \\( \\) 或 \\[ \\] 定界；零散的单个符号（如 ∈、∀、∅、ℵ₀）可保留 Unicode 写法；公式内不得出现中文汉字——需要中文说明时写在公式后面的括号里，且不要用 \\text 包裹中文；
5. 直接输出 Markdown 正文，不要额外解释、不要用代码块包裹全文。{extra}

转写稿：
{chunk}
"""


def _extra_block(extra: str) -> str:
    extra = (extra or "").strip()
    if not extra:
        return ""
    return ("\n6. 用户对这份总结提出了额外要求，请务必优先满足：\n   " +
            "\n   ".join(extra.splitlines()))


def _extra_note_map(extra: str) -> str:
    extra = (extra or "").strip()
    if not extra:
        return ""
    return (f"\n8. 此外，用户特别关注以下方面，请在摘要中优先、完整保留相关内容：{extra}")


# ------------------------------------------------------------- chunking ---

def make_chunks(segments: list[dict], chunk_chars: int = 6000) -> list[list[dict]]:
    """把 segment 列表聚合成若干块（确定性切分，保证续传一致）。"""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for seg in segments:
        cur.append(seg)
        size += len(seg["text"])
        if size >= chunk_chars:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_text(title: str, chunk: list[dict]) -> str:
    from .utils import build_txt
    return build_txt(chunk)


# ------------------------------------------------------------- pipeline ---

def summarize_transcript(
    segments: list[dict],
    title: str,
    duration: float,
    style: str,
    llm_cfg: dict,
    blocks_dir: Path,
    cancel=None,
    on_log: Callable[[str], None] = lambda m: None,
    on_progress: Callable[[float], None] = lambda p: None,
    extra_request: str = "",
) -> tuple[str, list[str]]:
    """转写稿 -> 总结 markdown。返回 (summary_md, 分块摘要列表)。
    blocks_dir 下已有 blk_XXX.md 的块会直接复用（断点续传）。"""
    blocks_dir.mkdir(parents=True, exist_ok=True)
    style = STYLE_DESC.get(style, STYLE_DESC["general"])
    extra_note_map = _extra_note_map(extra_request)  # 分块阶段：提醒优先保留关注点
    extra_block = _extra_block(extra_request)        # 汇总阶段：完整注入要求
    chunk_chars = int(llm_cfg.get("chunk_chars", 6000))
    chunks = make_chunks(segments, chunk_chars)
    on_log(f"转写稿共 {len(segments)} 段 / {sum(len(s['text']) for s in segments)} 字，"
           f"按 {chunk_chars} 字切分为 {len(chunks)} 块")

    def do_block(i: int, chunk: list[dict]) -> str:
        cache = blocks_dir / f"blk_{i:03d}.md"
        if cache.exists() and cache.stat().st_size > 0:
            text = cache.read_text(encoding="utf-8")
            on_log(f"块 {i + 1}/{len(chunks)}：使用已有缓存 {cache.name}")
            return text
        t0, t1 = chunk[0]["start"], chunk[-1]["end"]
        prompt = _fmt(MAP_PROMPT, title=title, i=i + 1, n=len(chunks),
                                   chunk=_chunk_text(title, chunk),
                                   extra_note=extra_note_map)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("已取消")
        on_log(f"块 {i + 1}/{len(chunks)}（{fmt_dur(t0)}-{fmt_dur(t1)}，"
               f"{sum(len(s['text']) for s in chunk)} 字）→ 调用 GLM …")
        text = _chat(llm_cfg, [{"role": "user", "content": prompt}], cancel)
        cache.write_text(text, encoding="utf-8")
        on_log(f"块 {i + 1}/{len(chunks)} 完成（{len(text)} 字）")
        return text

    if len(chunks) <= 1:
        # 短转写稿：直接整篇总结
        on_progress(0.5)
        prompt = _fmt(DIRECT_PROMPT, title=title, duration=fmt_dur(duration),
                                      style=style, chunk=_chunk_text(title, segments),
                                      extra=extra_block)
        summary = _chat(llm_cfg, [{"role": "user", "content": prompt}], cancel)
        on_progress(1.0)
        return summary, []

    # map：并发分块摘要
    summaries: list[str | None] = [None] * len(chunks)
    done = 0
    workers = max(1, int(llm_cfg.get("max_workers", 2)))
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_block, i, c): i for i, c in enumerate(chunks)}
        for fut in futures.as_completed(futs):
            i = futs[fut]
            summaries[i] = fut.result()  # 异常向上抛
            done += 1
            on_progress(done / len(chunks) * 0.85)
            if cancel is not None and cancel.is_set():
                raise RuntimeError("已取消")

    # reduce：汇总
    on_log(f"全部分块完成，汇总为最终{('总结')}…")
    joined = "\n\n".join(
        f"### 第 {i + 1} 段（{fmt_dur(chunks[i][0]['start'])}-{fmt_dur(chunks[i][-1]['end'])}）\n{s}"
        for i, s in enumerate(summaries) if s
    )
    prompt = _fmt(REDUCE_PROMPT, title=title, duration=fmt_dur(duration),
                                  n=len(chunks), style=style, joined=joined,
                                  extra=extra_block)
    final = _chat(llm_cfg, [{"role": "user", "content": prompt}], cancel)
    on_progress(1.0)
    return final, [s or "" for s in summaries]
