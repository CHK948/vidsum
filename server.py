# -*- coding: utf-8 -*-
"""VidSum 本地服务：静态页面 + REST API。仅监听 127.0.0.1。

用法: python server.py [--port 8322] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import config as cfgmod
from app import jobs as jobsmod
from app.summarize import LLMAuthError, _chat
from app.utils import is_media_file, find_local_models

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

MGR: jobsmod.JobManager | None = None


# ------------------------------------------------------------------ api ---
def api_system(cfg: dict) -> dict:
    gpu = ""
    try:
        import ctranslate2
        gpu = f"CUDA x{ctranslate2.get_cuda_device_count()}" if ctranslate2.get_cuda_device_count() else "CPU"
    except Exception as e:
        gpu = f"未知({e})"
    import sys
    return {
        "python": sys.version.split()[0],
        "device": gpu,
        "models": find_local_models(),
        "asr_model": cfg["asr"].get("model"),
        "llm_model": cfg["llm"].get("model"),
        "llm_base_url": cfg["llm"].get("base_url"),
        "has_key": bool(cfgmod.load()["llm"].get("api_key")),
        "key_masked": cfgmod.mask_key(cfgmod.load()["llm"].get("api_key", "")),
    }


def _user_dirs() -> list[dict]:
    """读取 XDG 用户目录，生成侧栏"快速位置"。"""
    import os
    home = Path.home()
    places = [("🏠 主目录", str(home))]
    mapping = {"DESKTOP": "🖥 桌面", "DOWNLOAD": "⬇ 下载", "DOCUMENTS": "📄 文档",
               "VIDEOS": "🎬 视频", "MUSIC": "🎵 音乐", "PICTURES": "🖼 图片"}
    parsed: dict[str, str] = {}
    cfg = Path.home() / ".config" / "user-dirs.dirs"
    if cfg.exists():
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r'\s*XDG_(\w+)_DIR="?([^"]*)"?', line)
                if m and m.group(2):
                    v = m.group(2).replace("$HOME", str(home))
                    parsed[m.group(1)] = v
        except Exception:
            pass
    for key, label in mapping.items():
        p = parsed.get(key)
        if p and Path(p).is_dir() and str(p) != str(home):
            places.append((label, p))
    places.append(("💽 文件系统", "/"))
    seen, out = set(), []
    for label, p in places:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            out.append({"label": label, "path": rp})
    return out


def api_browse(cfg: dict, raw_path: str) -> dict:
    path = Path(raw_path or Path.home()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if path.is_file():
        path = path.parent
    path = path.resolve()
    dirs, files = [], []
    try:
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        raise PermissionError(f"无权限读取目录: {path}")
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            st = e.stat()
        except OSError:
            continue
        if e.is_dir():
            dirs.append({"name": e.name, "mtime": st.st_mtime})
        else:
            files.append({"name": e.name, "size": st.st_size,
                          "mtime": st.st_mtime, "media": is_media_file(e.name)})
    return {"path": str(path), "parent": str(path.parent) if path.parent != path else "",
            "home": str(Path.home()), "dirs": dirs, "files": files,
            "special": _user_dirs()}


def api_test_llm(cfg: dict) -> dict:
    llm = cfgmod.load()["llm"]
    try:
        out = _chat(llm, [{"role": "user",
                           "content": "回复两个字：正常"}], None)
        return {"ok": True, "reply": out[:50], "model": llm.get("model")}
    except LLMAuthError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"{e}"}


SAFE_RESULT_FILES = {"summary.md", "transcript.txt", "transcript.srt", "transcript.json"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VidSum/1.0"

    # ---------------------------------------------------------- plumbing --
    def log_message(self, fmt, *args):  # 静默默认访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download_name: str | None = None):
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            q = urllib.parse.quote(download_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{q}")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def _route(self, method: str):
        try:
            self._dispatch(method)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._json({"error": f"{e}"}, 500)
            except Exception:
                pass

    # ----------------------------------------------------------- routing --
    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _dispatch(self, method: str):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        path = u.path
        cfg = cfgmod.load()

        if method == "GET" and path in ("/", "/index.html"):
            return self._file(WEB_DIR / "index.html")

        # 临时诊断：绕过前端 JS，直接看服务端渲染产物是否正常
        if method == "GET" and path == "/diag":
            from pathlib import Path as _P
            from app.mdrender import render_md_to_html
            done = sorted(_P("outputs").glob("*/summary.md"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if not done:
                return self._json({"error": "暂无已完成的总结"}, 404)
            src = done[0].read_text(encoding="utf-8")
            body = render_md_to_html(src)
            css = ("body{background:#fff;color:#111;font:14px/1.6 -apple-system,'PingFang SC',sans-serif;"
                   "max-width:900px;margin:30px auto;padding:0 20px}"
                   "h1,h2,h3,h4,h5,h6{margin:14px 0 8px}h2{font-size:16px}h3{font-size:14px;color:#0b7ec2}"
                   "h4,h5,h6{font-size:13.5px;color:#5a6472}"
                   "p{margin:6px 0}ul{margin:6px 0 6px 20px}li{margin:3px 0}"
                   ".ts{color:#0ea36b;font-family:monospace;font-size:12px;font-weight:600}"
                   ".mathwrap.inline{display:inline-block;vertical-align:middle;margin:0 1px}"
                   ".mathwrap.inline .eq-inline{vertical-align:middle}"
                   ".mathwrap.block{display:block;text-align:center;margin:10px 0;overflow-x:auto}"
                   ".mathwrap.block .eq-block{max-width:100%;height:auto}"
                   ".mdtable{border-collapse:collapse;margin:10px 0;font-size:13px;width:100%}"
                   ".mdtable th,.mdtable td{border:1px solid #ddd;padding:6px 10px;text-align:left}"
                   ".codeblock{background:#f8fafb;border:1px solid #ddd;border-radius:8px;padding:10px;"
                   "font-family:monospace;font-size:12px;white-space:pre-wrap;color:#41505f}")
            html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>诊断对照页</title>"
                    f"<style>{css}</style></head><body>"
                    f"<p style='background:#e7f6ef;padding:8px;border-radius:8px'>诊断对照页：此页面由服务端直接渲染，"
                    f"不经过任何前端 JS。文件：{done[0]}</p><div class='md'>{body}</div></body></html>")
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/api/"):
            if method == "GET" and path == "/api/system":
                return self._json(api_system(cfg))
            if method == "GET" and path == "/api/config":
                safe = json.loads(json.dumps(cfg, ensure_ascii=False))
                safe["llm"]["api_key"] = cfgmod.mask_key(safe["llm"].get("api_key", ""))
                return self._json(safe)
            if method == "POST" and path == "/api/config":
                body = self._body()
                cur = cfgmod.load()
                for section in ("server", "asr", "llm"):
                    if section in body and isinstance(body[section], dict):
                        for k, v in body[section].items():
                            if section == "llm" and k == "api_key":
                                # 空串/掩码值表示不修改
                                if v and "****" not in v:
                                    cur[section][k] = v.strip()
                                continue
                            cur[section][k] = v
                cfgmod.save(cur)
                return self._json({"ok": True, "key_masked": cfgmod.mask_key(cur["llm"].get("api_key", ""))})
            if method == "POST" and path == "/api/test-llm":
                return self._json(api_test_llm(cfg))
            if method == "GET" and path == "/api/browse":
                try:
                    return self._json(api_browse(cfg, q.get("path", "")))
                except Exception as e:
                    return self._json({"error": str(e)}, 400)
            if path == "/api/jobs":
                if method == "GET":
                    return self._json({"jobs": MGR.list()})
                if method == "POST":
                    body = self._body()
                    media = (body.get("path") or "").strip()
                    if not media:
                        return self._json({"error": "缺少视频路径"}, 400)
                    media = str(Path(media).expanduser())
                    if not Path(media).exists():
                        return self._json({"error": f"文件不存在: {media}"}, 400)
                    if not is_media_file(media):
                        return self._json({"error": "不是支持的音视频格式"}, 400)
                    job = MGR.create(media, {
                        "language": body.get("language") or "auto",
                        "style": body.get("style") or "course",
                        "title": body.get("title") or Path(media).stem,
                        "extra_request": (body.get("extra_request") or "").strip(),
                    })
                    return self._json({"job": job.to_dict()})
            m = _match_job(path)
            if m:
                jid, action = m
                job = MGR.get(jid)
                if not job:
                    return self._json({"error": "任务不存在"}, 404)
                if action == "" and method == "GET":
                    after = int(q.get("after", "0") or 0)
                    evs, nxt = job.events_after(after)
                    d = job.to_dict()
                    d["events"] = evs
                    d["next"] = nxt
                    return self._json(d)
                if action == "resume" and method == "POST":
                    body = self._body()
                    patch = {k: body[k] for k in ("language", "style", "extra_request") if k in body}
                    ok = MGR.resume(jid, patch or None)
                    return self._json({"ok": ok})
                if method == "POST" and action in ("cancel", "resume", "delete"):
                    ok = getattr(MGR, action)(jid)
                    return self._json({"ok": ok})
                if action == "meta" and method == "POST":
                    body = self._body()
                    j2 = MGR.set_meta(jid, body.get("title"), body.get("category"))
                    if not j2:
                        return self._json({"error": "任务不存在"}, 404)
                    return self._json({"job": j2.to_dict(brief=True)})
                if action == "result" and method == "GET":
                    name = q.get("name", "summary.md")
                    if name not in SAFE_RESULT_FILES:
                        return self._json({"error": "非法文件名"}, 400)
                    f = job.out_dir / name
                    if not f.exists():
                        return self._json({"error": "文件尚未生成"}, 404)
                    return self._json({"name": name, "content": f.read_text(encoding="utf-8", errors="replace")})
                if action == "result-html" and method == "GET":
                    name = q.get("name", "summary.md")
                    if name not in SAFE_RESULT_FILES:
                        return self._json({"error": "非法文件名"}, 400)
                    f = job.out_dir / name
                    if not f.exists():
                        return self._json({"error": "文件尚未生成"}, 404)
                    from app.mdrender import render_md_to_html
                    html = render_md_to_html(f.read_text(encoding="utf-8", errors="replace"))
                    return self._json({"name": name, "html": html})
                if action == "file" and method == "GET":
                    name = q.get("name", "")
                    if name not in SAFE_RESULT_FILES:
                        return self._json({"error": "非法文件名"}, 400)
                    f = job.out_dir / name
                    if not f.exists():
                        return self._json({"error": "文件尚未生成"}, 404)
                    return self._file(f, download_name=f"{job.title}_{name}")
            return self._json({"error": "未知接口"}, 404)

        # 静态资源（无）
        return self._json({"error": "not found"}, 404)


def _match_job(path: str):
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "jobs":
        jid = parts[2]
        action = parts[3] if len(parts) > 3 else ""
        return jid, action
    return None


def main():
    global MGR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = cfgmod.load()
    host = args.host or cfg["server"].get("host", "127.0.0.1")
    port = args.port or int(cfg["server"].get("port", 8322))

    MGR = jobsmod.JobManager(cfg)
    cfgmod.outputs_dir(cfg).mkdir(parents=True, exist_ok=True)

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"""
  ██╗   ██╗██╗██████╗ ███████╗██╗   ██╗███╗   ███╗
  ╚██╗ ██╔╝██║██╔══██╗██╔════╝██║   ██║████╗ ████║   本地视频语音转写 + GLM 总结
   ╚████╔╝ ██║██║  ██║███████╗██║   ██║██╔████╔██║
    ╚██╔╝  ██║██║  ██║╚════██║██║   ██║██║╚██╔╝██║   ASR: {cfg['asr'].get('model')}
     ██║   ██║██████╔╝███████║╚██████╔╝██║ ╚═╝ ██║   LLM: {cfg['llm'].get('model')}
     ╚═╝   ╚═╝╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝
  服务已启动: http://{host}:{port}   (Ctrl+C 退出)
""", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出 …")
        srv.shutdown()


if __name__ == "__main__":
    main()
