# -*- coding: utf-8 -*-
"""Markdown -> HTML（与前端同规则），数学公式经本地 LaTeX 渲染为内联 SVG。

处理顺序（每一层都先"摘走"再交给下一层，避免 $ 计数错位）：
1. ``` 代码围栏 -> 占位符（内容原样 <pre> 展示，不做公式提取）
2. $$...$$ 块公式（可跨行）-> 占位符
3. 跨行的行内公式修复（GLM 偶尔把长列表项硬换行，$...$ 断成两行 -> 拼回一行）
4. $...$ 行内公式 -> 占位符
5. markdown 转 HTML（标题/列表/粗体/代码/时间戳/引用/表格）
6. 占位符回填 SVG（LaTeX 失败的回退为代码样式）
"""
from __future__ import annotations

import html as _html
import re

from .mathrender import render_formulas

# 占位符：控制字符包裹，不会被 markdown 转义/正则破坏
_STX, _ETX = "\x02", "\x03"

_EQ_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
_EQ_INLINE_RE = re.compile(r"\$([^$\n]+?)\$")
_FENCE_RE = re.compile(r"```[^\n]*\n[\s\S]*?```|```[^\n]*```")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _extract_fences(text: str, store: list[str]) -> str:
    def repl(m: re.Match) -> str:
        store.append(m.group(0))
        return f"{_STX}CODE{len(store) - 1}{_ETX}"
    return _FENCE_RE.sub(repl, text)


def _extract_formulas(text: str, store: list[tuple[str, str]]) -> str:
    """块公式(跨行)与行内公式 -> 占位符。"""
    def repl_block(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex:
            return m.group(0)
        store.append(("block", tex))
        return f"{_STX}EQ{len(store) - 1}{_ETX}"

    def repl_inline(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex:
            return m.group(0)
        store.append(("inline", tex))
        return f"{_STX}EQ{len(store) - 1}{_ETX}"

    text = _EQ_BLOCK_RE.sub(repl_block, text)
    text = _EQ_INLINE_RE.sub(repl_inline, text)
    return text


def _join_broken_inline_math(text: str, max_join: int = 4) -> str:
    """把被硬换行拆开的行内公式拼回一行（$ 计数不平衡的行向后合并）。
    只在相邻少量行内解决，超限放弃，避免把整个文档吞成一行。"""
    lines = text.split("\n")
    out: list[str] = []
    buf: str | None = None
    joined = 0
    for ln in lines:
        if buf is not None:
            buf += ln
            joined += 1
            if buf.count("$") % 2 == 0:
                out.append(buf)
                buf, joined = None, 0
            elif joined >= max_join:  # 放弃拼接，原样断开
                out.append(buf)
                buf, joined = None, 0
        elif ln.count("$") % 2 == 1:
            buf, joined = ln, 0
        else:
            out.append(ln)
    if buf is not None:
        out.append(buf)
    return "\n".join(out)


def _inline(s: str) -> str:
    s = _html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)(?:[-–~](\d{1,2}:\d{2}(?::\d{2})?))?\]",
               lambda m: f'<span class="ts">[{m.group(1)}{"-" + m.group(2) if m.group(2) else ""}]</span>', s)
    return s


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}[-:\s|]*\|?\s*$")


def _try_table(lines: list[str], i: int, out: list[str]) -> int:
    """lines[i] 起是否为 markdown 表格；是则输出 <table> 并返回结束下标，否则返回 i。"""
    if i + 1 >= len(lines) or "|" not in lines[i] or not _TABLE_SEP_RE.match(lines[i + 1]):
        return i
    def cells(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]
    head = cells(lines[i])
    j = i + 2
    body: list[list[str]] = []
    while j < len(lines) and "|" in lines[j] and lines[j].strip():
        body.append(cells(lines[j]))
        j += 1
    thead = "".join(f"<th>{_inline(c)}</th>" for c in head)
    rows = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
    out.append(f'<table class="mdtable"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>')
    return j


def md_to_html(src: str) -> str:
    lines = src.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    in_ul = False
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if "|" in t and i + 1 < len(lines):
            j = _try_table(lines, i, out)
            if j != i:
                i = j
                continue
        if re.match(r"^[-*·]\s+", t):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append("<li>" + _inline(re.sub(r"^[-*·]\s+", "", t)) + "</li>")
            i += 1
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if t.startswith("### "):
            out.append("<h3>" + _inline(t[4:]) + "</h3>")
        elif t.startswith("## "):
            out.append("<h2>" + _inline(t[3:]) + "</h2>")
        elif t.startswith("# "):
            out.append("<h1>" + _inline(t[2:]) + "</h1>")
        elif t.startswith(">"):
            out.append("<blockquote>" + _inline(t.lstrip("> ")) + "</blockquote>")
        elif t == "":
            out.append('<div style="height:6px"></div>')
        else:
            out.append("<p>" + _inline(t) + "</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def render_md_to_html(src: str) -> str:
    """完整管线：围栏/公式提取 -> 修复跨行公式 -> 转换 -> 回填 SVG。"""
    fences: list[str] = []
    text = _extract_fences(src, fences)

    formulas: list[tuple[str, str]] = []
    text = _extract_formulas(text, formulas)     # 先摘 $$ 块（跨行）
    text = _join_broken_inline_math(text)        # 修复被换行拆开的 $...$
    text = _extract_formulas(text, formulas)     # 再摘行内 $...$（含 fence 占位行，不受影响）

    html = md_to_html(text)

    # 回填代码围栏（原样 <pre>）
    for i, code in enumerate(fences):
        ph = f"{_STX}CODE{i}{_ETX}"
        html = html.replace(ph, f"<pre class='codeblock'>{_html.escape(code, quote=False)}</pre>")

    # 回填公式 SVG
    if formulas:
        svgs = render_formulas(formulas)
        for i, (mode, tex) in enumerate(formulas):
            ph = f"{_STX}EQ{i}{_ETX}"
            svg = svgs[i]
            if svg:
                rep = f'<span class="mathwrap {mode}">{svg}</span>'
            else:  # LaTeX 编译失败：回退为代码样式显示原公式
                s2 = ("$" + tex + "$") if mode == "inline" else ("$$" + tex + "$$")
                rep = f'<code class="math-fallback">{_html.escape(s2, quote=False)}</code>'
            html = html.replace(ph, rep)

    # 兜底：仍残留的未配对 $ 原样显示为字符（不参与公式）
    html = html.replace(_STX, "").replace(_ETX, "")
    return html
