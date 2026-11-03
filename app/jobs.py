# -*- coding: utf-8 -*-
"""任务管理：后台线程执行 转写→总结 管线，支持取消与断点续传。"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config as cfgmod
from . import summarize, transcribe
from .utils import build_srt, build_txt, fmt_dur

STAGES = ["queued", "extracting", "transcribing", "summarizing", "done"]
STAGE_LABEL = {
    "queued": "排队中", "extracting": "抽取音频", "transcribing": "本地转写",
    "summarizing": "GLM 总结", "done": "完成", "error": "出错", "cancelled": "已取消",
}

# 断点续传判断依赖的中间产物
_F = ["audio.wav", "transcript.json", "transcript.srt", "transcript.txt", "summary.md"]


class Job:
    def __init__(self, jid: str, media_path: str, params: dict, out_dir: Path):
        self.id = jid
        self.media_path = media_path
        self.title = params.get("title") or Path(media_path).stem
        self.params = params
        self.out_dir = out_dir
        self.status = "queued"          # queued/running/done/error/cancelled
        self.stage = "queued"
        self.progress = 0.0             # 0..1（当前阶段内）
        self.eta = 0.0                  # 预计剩余秒数（转写/总结阶段动态更新）
        self.error = ""
        self.events: list[dict] = []
        self.ev_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.created = time.time()
        self.started = 0.0
        self.finished = 0.0
        self.thread: threading.Thread | None = None

    def to_dict(self, brief=False):
        d = {
            "id": self.id, "title": self.title, "media_path": self.media_path,
            "status": self.status, "stage": self.stage, "stage_label": STAGE_LABEL.get(self.stage, self.stage),
            "progress": round(self.progress, 4),
            "eta": round(self.eta) if self.status == "running" and self.eta > 0 else 0, "error": self.error,
            "created": self.created, "started": self.started, "finished": self.finished,
            "params": {k: v for k, v in self.params.items() if k != "title"},
            "out_dir": str(self.out_dir),
        }
        if not brief:
            d["files"] = {f: (self.out_dir / f).exists() for f in _F}
        return d

    def log(self, msg: str, type_: str = "log"):
        line = {"t": time.time(), "type": type_, "msg": msg}
        with self.ev_lock:
            self.events.append(line)
            if len(self.events) > 800:
                del self.events[:200]
        print(f"[job {self.id[:6]}] {msg}", flush=True)

    def live(self, data: dict):
        """ASR 实时状态事件（不写 stdout 日志，仅推送前端）。"""
        if data.get("remain") is not None:
            try:
                self.eta = max(0.0, float(data["remain"]))
            except (TypeError, ValueError):
                pass
        line = {"t": time.time(), "type": "live", **data}
        with self.ev_lock:
            self.events.append(line)
            if len(self.events) > 800:
                del self.events[:200]

    def events_after(self, idx: int):
        with self.ev_lock:
            return self.events[idx:], len(self.events)


class JobManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()
        self._state_path = cfgmod.outputs_dir(cfg) / "tasks.json"
        self._load_state()

    # ------------------------------------------------ 持久化（仅记录列表）--
    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            items = json.loads(self._state_path.read_text(encoding="utf-8"))
            for it in items:
                j = Job(it["id"], it["media_path"], it.get("params", {}), Path(it["out_dir"]))
                j.created = it.get("created", time.time())
                if it.get("status") in ("running", "queued"):
                    j.status = "error"
                    j.stage = "error"
                    j.error = "服务重启导致任务中断，可点击“继续”从断点恢复"
                elif it.get("status"):
                    j.status = it["status"]
                    j.stage = it.get("stage", "done")
                    j.error = it.get("error", "")
                j.finished = it.get("finished", 0)
                self.jobs[j.id] = j
                self.order.append(j.id)
        except Exception as e:
            print(f"[jobs] 读取 tasks.json 失败: {e}")

    def persist(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            items = []
            for jid in reversed(self.order):
                j = self.jobs[jid]
                items.append({
                    "id": j.id, "media_path": j.media_path, "params": j.params,
                    "out_dir": str(j.out_dir), "status": j.status, "stage": j.stage,
                    "error": j.error, "created": j.created, "finished": j.finished,
                })
            self._state_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
        except Exception as e:
            print(f"[jobs] 写 tasks.json 失败: {e}")

    # -------------------------------------------------------- 创建/查询 --
    def create(self, media_path: str, params: dict) -> Job:
        jid = time.strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]
        out_dir = cfgmod.outputs_dir(self.cfg) / jid
        params = dict(params)
        params.setdefault("title", Path(media_path).stem)
        j = Job(jid, media_path, params, out_dir)
        with self._lock:
            self.jobs[jid] = j
            self.order.append(jid)
        j.thread = threading.Thread(target=self._run, args=(j,), daemon=True,
                                    name=f"job-{jid}")
        j.thread.start()
        self.persist()
        return j

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def list(self) -> list[dict]:
        return [self.jobs[j].to_dict(brief=True) for j in reversed(self.order) if j in self.jobs]

    # ---------------------------------------------------- 取消/继续/删除 --
    def cancel(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j and j.status in ("queued", "running"):
            j.cancel_event.set()
            j.log("收到取消请求 …")
            return True
        return False

    def resume(self, jid: str) -> bool:
        """从断点继续（重新跑管线，已有中间产物自动跳过）。
        done 的任务也可重新总结：转写/分块摘要走缓存，仅重新生成最终总结。"""
        j = self.jobs.get(jid)
        if j and j.status in ("error", "cancelled", "done"):
            j.status = "queued"
            j.stage = "queued"
            j.error = ""
            j.finished = 0.0
            j.cancel_event = threading.Event()
            j.thread = threading.Thread(target=self._run, args=(j,), daemon=True)
            j.thread.start()
            self.persist()
            return True
        return False

    def delete(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if not j:
            return False
        if j.status in ("queued", "running"):
            j.cancel_event.set()
        import shutil
        try:
            if j.out_dir.exists() and j.out_dir.name == jid:
                shutil.rmtree(j.out_dir)
        except Exception as e:
            j.log(f"清理目录失败: {e}")
        with self._lock:
            self.jobs.pop(jid, None)
            if jid in self.order:
                self.order.remove(jid)
        self.persist()
        return True

    # ------------------------------------------------------------ 管线 --
    def _run(self, j: Job):
        cfg = cfgmod.load()  # 每任务开始时取最新配置（key 可在运行中被更新）
        asr_cfg = cfg["asr"]
        llm_cfg = dict(cfg["llm"])
        lang = j.params.get("language") or "auto"
        style = j.params.get("style") or "course"
        extra = (j.params.get("extra_request") or "").strip()
        if extra:
            j.log(f"用户附加要求：{extra[:80]}{'…' if len(extra) > 80 else ''}")

        def set_stage(stage: str, progress=0.0):
            j.stage = stage
            j.progress = progress

        try:
            j.status = "running"
            j.started = time.time()
            j.log(f"任务开始：{j.media_path}")
            if not Path(j.media_path).exists():
                raise FileNotFoundError(f"文件不存在: {j.media_path}")

            # 1) 抽音频 ------------------------------------------------------
            wav = j.out_dir / "audio.wav"
            if wav.exists() and wav.stat().st_size > 44:
                j.log("audio.wav 已存在，跳过抽取（断点续传）")
            else:
                set_stage("extracting")
                transcribe.extract_audio(j.media_path, wav, on_log=j.log)

            # 2) 本地转写 ----------------------------------------------------
            tj = j.out_dir / "transcript.json"
            if tj.exists() and tj.stat().st_size > 2:
                j.log("transcript.json 已存在，跳过转写（断点续传）")
                transcript = json.loads(tj.read_text(encoding="utf-8"))
            else:
                set_stage("transcribing", 0.0)

                def on_prog(p):
                    j.progress = max(0, min(p, 1.0)) * 0.95

                try:
                    transcript = transcribe.transcribe(
                        wav, asr_cfg, lang, cancel=j.cancel_event,
                        on_log=j.log, on_live=j.live)
                except transcribe.CancelledError:
                    raise
                except Exception as e:
                    if (asr_cfg.get("device", "auto") != "cpu"
                            and ("cuda" in str(e).lower() or "cublas" in str(e).lower()
                                 or "cudnn" in str(e).lower())):
                        j.log(f"GPU 转写失败（{e}），自动回退 CPU …")
                        cpu_cfg = dict(asr_cfg, device="cpu", compute_type="int8")
                        transcript = transcribe.transcribe(
                            wav, cpu_cfg, lang, cancel=j.cancel_event,
                            on_log=j.log, on_live=j.live)
                    else:
                        raise
                tj.write_text(json.dumps(transcript, ensure_ascii=False, indent=1), "utf-8")

            # 副产物：srt / txt
            (j.out_dir / "transcript.srt").write_text(build_srt(transcript["segments"]), "utf-8")
            (j.out_dir / "transcript.txt").write_text(build_txt(transcript["segments"]), "utf-8")
            j.log(f"已生成 transcript.srt / transcript.txt（语言 {transcript.get('language')}，"
                  f"时长 {fmt_dur(transcript.get('duration') or 0)}）")

            if j.cancel_event.is_set():
                raise transcribe.CancelledError("已取消")

            # 3) GLM 总结 ----------------------------------------------------
            j.eta = 0.0  # 转写结束，切到总结阶段的估算
            set_stage("summarizing", 0.0)
            blocks_dir = j.out_dir / "blocks"

            prog_state = {"last_p": 0.0, "last_t": time.time()}

            def on_prog(p):  # 0.85 map + 0.15 reduce
                p = max(0.0, min(p, 1.0))
                j.progress = p
                # 按进度增量速率估算剩余时间
                now = time.time()
                dp = p - prog_state["last_p"]
                if dp >= 0.02 and p < 1.0:
                    rate = dp / max(now - prog_state["last_t"], 1e-6)
                    j.eta = max(0.0, (1.0 - p) / max(rate, 1e-9))
                    prog_state["last_p"], prog_state["last_t"] = p, now

            summary, _blocks = summarize.summarize_transcript(
                transcript["segments"], j.title, transcript.get("duration") or 0.0,
                style, llm_cfg, blocks_dir, cancel=j.cancel_event,
                on_log=j.log, on_progress=on_prog, extra_request=extra)
            (j.out_dir / "summary.md").write_text(summary, "utf-8")
            j.log("总结已写入 summary.md")

            # 4) 完成 --------------------------------------------------------
            j.status, j.stage, j.progress = "done", "done", 1.0
            j.eta = 0.0
            j.finished = time.time()
            j.log(f"任务完成，总耗时 {fmt_dur(j.finished - j.started)}", "done")
        except transcribe.CancelledError:
            j.status, j.stage, j.error = "cancelled", "cancelled", "已取消"
            j.finished = time.time()
            j.eta = 0.0
            j.log("任务已取消", "warn")
        except Exception as e:
            j.status, j.stage = "error", "error"
            j.error = str(e) or e.__class__.__name__
            j.finished = time.time()
            j.eta = 0.0
            j.log(f"出错: {j.error}", "error")
            j.log(traceback.format_exc(limit=6), "error")
        finally:
            self.persist()
