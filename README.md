# VidSum —— 本地视频语音转写 + GLM 智能总结

一个纯本地运行的工具：用本地已缓存的 **faster-whisper** 模型识别视频/音频中的语音（GPU 加速），
再调用智谱 **GLM-5.3-Flash** 生成分块总结，提供本地网页界面管理任务。

## 功能

- 🎬 支持常见音视频格式（mp4 / mkv / ts / flv / mp3 …），自动用 ffmpeg 抽取 16kHz 单声道音频
- 🎧 本地 ASR：faster-whisper（默认 large-v3，float16，GPU 优先；CUDA 环境异常时自动回退 CPU）
- 📝 产出：`transcript.srt` 字幕、`transcript.txt` 带时间戳全文、`transcript.json`
- 🤖 GLM-5.3-Flash 总结：长转写稿自动分块（map）→ 汇总（reduce），支持并发、失败重试
- ⏸ 断点续传：转写/分块摘要落盘缓存，失败或重启后可"从断点继续"，不重复烧 API
- 🖥 网页界面：目录浏览选文件、实时进度与日志、结果在线预览（Markdown 渲染）与下载

## 快速开始

```bash
./run.sh                # 默认 http://127.0.0.1:8322
./run.sh --port 9000    # 换端口
```

首次打开页面 → 右上角 **⚙ 设置** → 粘贴智谱 API Key（open.bigmodel.cn）→ **测试 GLM 连接** → 保存。

然后：选择音视频文件 → 语言/风格 → **开始转写并总结**。

## 配置

所有配置保存在 `config.json`（也可用环境变量 `ZHIPUAI_API_KEY` 覆盖 key）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `asr.model` | faster-whisper-large-v3 | 本地 HF 缓存中的模型名或目录绝对路径 |
| `asr.device` | auto | auto / cuda / cpu |
| `asr.vad_filter` | true | VAD 静音过滤（长视频明显提速） |
| `asr.initial_prompt` | "" | 热词提示，帮助识别专有名词 |
| `llm.model` | glm-5.3-flash | 智谱模型 code |
| `llm.chunk_chars` | 6000 | 长转写稿分块阈值（字符） |
| `llm.max_workers` | 2 | 分块摘要并发数 |
| `server.port` | 8322 | 网页端口 |

## 目录结构

```
vidsum/
├── run.sh              启动脚本（自动设置 CUDA 库路径）
├── server.py           HTTP 服务 + API
├── app/
│   ├── transcribe.py   ffmpeg 抽音频 + faster-whisper 转写
│   ├── summarize.py    GLM 调用 + 分块 map-reduce 总结
│   ├── jobs.py         任务线程、进度事件、断点续传
│   ├── config.py       配置读写
│   └── utils.py        SRT/文本生成、模型路径解析
├── web/index.html      前端单页
└── outputs/<任务id>/   audio.wav / transcript.* / summary.md / blocks/
```

## 常见问题

- **未找到模型**：模型需在 `~/.cache/huggingface/hub` 中（如 `models--Systran--faster-whisper-large-v3`），
  或在设置里填模型目录的绝对路径。
- **GPU 转写失败自动回退 CPU**：`run.sh` 已自动注入 pip `nvidia-*` 包的库路径；
  若仍失败（如驱动问题），任务会继续用 CPU 完成但速度慢很多。
- **GLM 401**：API Key 错误；在设置里重新粘贴并测试。
