# -*- coding: utf-8 -*-
"""用本地 LaTeX 把数学公式渲染成 SVG（不依赖外部服务）。

管线: $公式$ -> 批量 .tex(preview+tightpage, 每公式一页) -> latex -> dvisvgm --no-fonts
      -> 每页一个 SVG(字形转路径, 浏览器零字体依赖) -> 全局磁盘缓存
失败公式回退 None（调用方以代码文本展示）。
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~/.cache/vidsum_math"))
_lock = threading.Lock()

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 普通公式：latex -> dvi（快）
TEX_HEAD_LATEX = r"""\documentclass[12pt]{article}
\usepackage[active,tightpage]{preview}
\usepackage{amsmath,amssymb,amsfonts}
\pagestyle{empty}
\begin{document}
"""
# 含中文的公式：xelatex -> xdv（DVI 引擎不支持中文）
TEX_HEAD_XELATEX = r"""\documentclass[12pt]{article}
\usepackage[active,tightpage]{preview}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{xeCJK}
\setCJKmainfont{AR PL UKai CN}
\pagestyle{empty}
\begin{document}
"""
TEX_TAIL = r"\end{document}" + "\n"

_EXT = {"latex": "dvi", "xelatex": "xdv"}


def _formula_key(mode: str, tex: str) -> str:
    return hashlib.sha1((mode + "\x00" + tex).encode("utf-8")).hexdigest()[:24]


def _run(cmd: list[str], cwd: str, timeout: int = 60) -> bool:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0
    except Exception:
        return False


def _build_batch(jobs: list[tuple[int, str, str]], workdir: str, engine: str = "latex") -> bool:
    """jobs: [(idx, mode, tex)]；一个 tex 文件多页编译。
    engine: latex(默认,快) 或 xelatex(支持公式内中文, -no-pdf 输出 xdv)。
    nonstopmode 下个别坏公式只影响其所在页，其余页正常产出，
    因此只要 dvi/xdv 存在即视为成功，缺失页由调用方逐个兜底。"""
    head = TEX_HEAD_XELATEX if engine == "xelatex" else TEX_HEAD_LATEX
    body = head
    for _, mode, tex in jobs:
        style = "displaystyle" if mode == "block" else "textstyle"
        body += "\\begin{preview}$\\" + style + "{" + tex + "}$\\end{preview}\n"
    body += TEX_TAIL
    (Path(workdir) / "batch.tex").write_text(body, encoding="utf-8")
    if engine == "xelatex":
        _run(["xelatex", "-no-pdf", "-interaction=nonstopmode", "batch.tex"],
             workdir, timeout=300)
    else:
        _run(["latex", "-interaction=nonstopmode", "batch.tex"], workdir, timeout=300)
    dvi = Path(workdir) / ("batch." + _EXT[engine])
    if not dvi.exists():
        return False

    import concurrent.futures as futures

    def one_page(page: int, idx: int) -> None:
        out = Path(workdir) / f"eq_{idx}.svg"
        _run(["dvisvgm", "--no-fonts", "--exact", "--precision=4",
              f"--page={page}", "-o", out.name, dvi.name],
             workdir, timeout=120)

    workers = min(8, max(1, len(jobs)))
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_page, page, idx)
                for page, (idx, _, _) in enumerate(jobs, 1)]
        for f in futs:
            f.result()
    return True


def _build_single(idx: int, mode: str, tex: str, workdir: str, engine: str = "latex") -> bool:
    head = TEX_HEAD_XELATEX if engine == "xelatex" else TEX_HEAD_LATEX
    body = head + "\\begin{preview}$\\" + (
        "displaystyle" if mode == "block" else "textstyle"
    ) + "{" + tex + "}$\\end{preview}\n" + TEX_TAIL
    (Path(workdir) / "one.tex").write_text(body, encoding="utf-8")
    if engine == "xelatex":
        ok = _run(["xelatex", "-no-pdf", "-interaction=nonstopmode", "one.tex"],
                  workdir, timeout=120)
    else:
        ok = _run(["latex", "-interaction=nonstopmode", "one.tex"], workdir, timeout=60)
    if not ok:
        return False
    dvi = Path(workdir) / ("one." + _EXT[engine])
    if not dvi.exists():
        return False
    out = Path(workdir) / f"eq_{idx}.svg"
    return (_run(["dvisvgm", "--no-fonts", "--exact", "--precision=4",
                  "--page=1", "-o", out.name, dvi.name],
                 workdir, timeout=60) and out.exists() and out.stat().st_size > 0)


def render_formulas(formulas: list[tuple[str, str]]) -> list[str | None]:
    """输入 [(mode, tex)]，mode: 'inline'|'block'。返回与输入等长的 SVG 列表（失败为 None）。"""
    if not formulas:
        return []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[str | None] = [None] * len(formulas)
    todo: list[tuple[int, str, str]] = []
    for i, (mode, tex) in enumerate(formulas):
        cache = CACHE_DIR / (_formula_key(mode, tex) + ".svg")
        if cache.exists() and cache.stat().st_size > 0:
            try:
                results[i] = cache.read_text(encoding="utf-8")
            except Exception:
                todo.append((i, mode, tex))
        else:
            todo.append((i, mode, tex))
    if not todo:
        return results

    with _lock:  # latex 编译全局串行（已读缓存的直接返回）
        # 双重检查：等待期间其他线程可能已写入
        still = []
        for (i, mode, tex) in todo:
            cache = CACHE_DIR / (_formula_key(mode, tex) + ".svg")
            if cache.exists() and cache.stat().st_size > 0:
                try:
                    results[i] = cache.read_text(encoding="utf-8")
                    continue
                except Exception:
                    pass
            still.append((i, mode, tex))
        todo = still
        if not todo:
            return results

        with tempfile.TemporaryDirectory(prefix="vidsum_tex_") as wd:
            # 按引擎分组：含中文 -> xelatex，普通 -> latex（各自批量编译 + 逐个兜底）
            groups: dict[str, list[tuple[int, str, str]]] = {"latex": [], "xelatex": []}
            for i, mode, tex in todo:
                groups["xelatex" if _CJK_RE.search(tex) else "latex"].append((i, mode, tex))
            for engine, group in groups.items():
                if not group:
                    continue
                if _build_batch(group, wd, engine):
                    for i, mode, tex in group:
                        f = Path(wd) / f"eq_{i}.svg"
                        if f.exists() and f.stat().st_size > 0:
                            results[i] = _finalize(f, mode)
                # 批量中缺失的页（坏公式所在页）逐个尝试，保证能渲染的尽量渲染
                for i, mode, tex in group:
                    if results[i] is not None:
                        continue
                    if _build_single(i, mode, tex, wd, engine):
                        f = Path(wd) / f"eq_{i}.svg"
                        if f.exists() and f.stat().st_size > 0:
                            results[i] = _finalize(f, mode)

        # 写缓存
        for i, mode, tex in todo:
            if results[i] is not None:
                try:
                    (CACHE_DIR / (_formula_key(mode, tex) + ".svg")).write_text(
                        results[i], encoding="utf-8")
                except Exception:
                    pass
    return results


def _finalize(svg_file: Path, mode: str) -> str:
    """SVG -> base64 data-URI <img>。

    关键：dvisvgm 的根元素 width/height 是 pt 单位，作为 HTML 内联 <svg> 时
    浏览器不接受该属性值，尺寸失效导致公式撑满容器；转成 <img data-uri> 后
    SVG 按独立文档解析（pt 合法），且内部 id 与页面完全隔离。"""
    import base64
    svg = svg_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<svg[\s\S]*</svg>", svg)
    if m:
        svg = m.group(0)
    wm = re.search(r"width='([\d.]+)pt'", svg)
    hm = re.search(r"height='([\d.]+)pt'", svg)
    wpx = round(float(wm.group(1)) * 4 / 3, 1) if wm else None
    hpx = round(float(hm.group(1)) * 4 / 3, 1) if hm else None
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    cls = "eq-block" if mode == "block" else "eq-inline"
    dim = (f' width="{wpx}" height="{hpx}"' if wpx and hpx else "")
    return (f'<img class="{cls}" src="data:image/svg+xml;base64,{b64}" '
            f'alt="公式"{dim}>')
