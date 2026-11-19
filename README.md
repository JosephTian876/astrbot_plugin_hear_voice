# 我听得见语音 (`astrbot_plugin_hear_voice`)

> 当前版本 **0.1.0** ｜ 适配 **AstrBot >= 4.24.1**（基于符号存在性论证：依赖 `TextPart.mark_as_temp()` 的 `_no_save` 机制；已在 AstrBot `4.29.1` 源码契约上完成离线验证）

一个面向 AstrBot 的**非抢答、双阶段、多引擎主备**语音理解插件，原生支持 **Gemini 多模态**（原生 `generateContent` / OpenAI 兼容 `input_audio`）、**DeepSeek 官方 ASR**（`POST /v1/tools/asr`）与 **OpenAI Whisper** 自由主/备组合，默认静默放行失败策略（`on_stt_fail="pass"`），原生兼容「我会永远陪着你」（`astrbot_plugin_private_companion`）与「我会牢牢记住你」（`astrbot_plugin_memory_companion`）。

---

## ✨ 核心特性

- **多引擎自由主备 + 无缝故障切换（`hear_voice/engine.py`）**：
  - **支持三种引擎**：`gemini`（多模态 6 项声学分析或纯转写）、`deepseek_asr`（DeepSeek 官方 `/v1/tools/asr` 语音识别）、`openai_whisper`（`/v1/audio/transcriptions`）；
  - **任意互为主备**：例如主用 `Gemini`（享受 6 项情绪/语气/环境音分析）+ 备用 `DeepSeek ASR`（如 `deepseek/deepseek-flash`）；也可主用 `DeepSeek ASR` + 备用 `Gemini`；
  - **智能自动识别（`auto`）**：根据 `provider_id`、`api_base` 或 `provider_type` 自动判定引擎——命中 `deepseek.com`、`/tools/asr` 或 `deepseek` Provider 自动走 DeepSeek 官方 ASR，Whisper 类型走 Whisper，其余默认走 Gemini；
  - **精准故障切换**：不可重试 `4xx`（400/401/403）或重试耗尽的 `429`/`5xx`/网络异常/HTTP 200 错误载荷会立即记录脱敏告警并自动切换至回退通道。
- **DeepSeek 官方 ASR 深度适配（`hear_voice/deepseek_client.py` & `hear_voice/audio_resolver.py`）**：
  - 严格对齐官方 `/v1/tools/asr` `multipart/form-data`（`file` 字段 + `Authorization: Bearer`）契约；
  - 支持将 `https://api.deepseek.com`、`https://api.deepseek.com/v1` 或留空自动规范化为 `https://api.deepseek.com/v1/tools/asr`，并完整保留自定义反代 `/tools/asr` 路径；
  - **智能音频容器与规格适配**：原生 16-bit PCM WAV（`8k/16k/22.05k/24k/32k/44.1k/48k` Hz，含 QQ SILK 经 `pysilk` 解码直出的 24kHz WAV）及 Ogg Opus 在 `<= 8 MiB` 且 `<= 120s` 时零转码直传；其他格式自动异步调用 `ffmpeg` 转码为 `16kHz` 单声道 `pcm_s16le` WAV（若 WAV 体积超过 `8 MiB` 自动回退 `libopus` Ogg），严格执行 `<= 120s` 与 `<= 8 MiB` 官方硬限制并输出脱敏告警。
- **两阶段非抢答双轨隔离（`hear_voice/hook_pipeline.py`）**：
  - **阶段 1（`priority=300000` 消息事件钩子）**：先于陪伴与记忆插件完成语音解析，**绝不调用 `request_llm()`**、**保留原 `Record` 位置不变**，仅将纯净原话 `[语音消息]: ...` 追加写入 `Plain` 与 `message_str`，同时导出 `_hear_voice_*`、`_gemini_stt_*` 兼容别名与 `astrbot_stt_transcripts`；
  - **阶段 2（`on_llm_request`, `priority=300000`）**：Gemini `rich` 模式下将 6 项声学分析以 `TextPart(...).mark_as_temp()` 临时入模而不污染记忆库；若最终命中 DeepSeek ASR / Whisper 等纯转写引擎，绝不编造 6 项声学模板，也不注入空临时块；统一清理冗余 `req.audio_urls` 与 `[Audio Attachment: ...]` 占位。
- **默认静默放行失败策略（`on_stt_fail="pass"`）**：
  - 默认策略为 `"pass"`：即使主备全挂，也仅在后台日志输出含状态码与脱敏摘要的 `logger.warning("[HearVoice] ...")`，**绝不向 QQ 群或私聊发送失败提示刷屏**，原事件静默放行由下游决定如何响应；仍保留 `notify_pass` / `block` / `notify` 供进阶调试选用。
- **安全防线与可选 STT Provider 注册（`hear_voice/stt_provider.py`）**：
  - 禁用/校验重定向、DNS 解析校验 + `StaticResolver` 防重绑定、拦截私网/回环/IPv4-mapped IPv6、本地路径 `realpath` + `commonpath` 白名单、全链路密钥脱敏；
  - 配置 `register_stt_provider=true` 可挂载标准 `hear_voice_stt` `STTProvider`（底层复用同一套主备引擎，带热重载幂等护栏）。

---

## 📦 目录结构

```text
astrbot_plugin_hear_voice/
├── main.py                      # AstrBot 插件入口（HearVoicePlugin，阶段1/2钩子与生命周期）
├── metadata.yaml                # 插件元数据（astrbot_plugin_hear_voice / 我听得见语音）
├── _conf_schema.json            # AstrBot WebUI 配置表单（主/备双槽配置、secret 标记、默认 pass）
├── requirements.txt             # 运行依赖（aiohttp, silk-python）
├── docs/
│   └── API.md                   # 对外集成契约文档（_hear_voice_* / _gemini_stt_* / astrbot_stt_transcripts）
├── hear_voice/
│   ├── __init__.py              # 包导出
│   ├── config.py                # 主/备双槽配置校验、别名兼容、NapCat 路径发现与统一脱敏函数
│   ├── normalize.py             # 模型名清洗、rich 6项提示词与解析器、错误/幻觉过滤
│   ├── audio_resolver.py        # 语音提取、OneBot 兜底、SSRF/路径防护、SILK 解码与 DeepSeek WAV/Ogg 适配
│   ├── gemini_client.py         # Gemini 原生（generateContent）与 OpenAI 兼容（input_audio）协议适配器
│   ├── deepseek_client.py       # DeepSeek 官方 ASR（POST /v1/tools/asr）multipart 客户端与 URL 规范化
│   ├── engine.py                # 统一 SttEndpointSpec、引擎自动判定、共享 HTTP 重试传输层、Whisper 与主备切换引擎
│   ├── hook_pipeline.py         # 两阶段非抢答核心管线、双前缀契约导出、纯转写零空块护栏与音频清理
│   └── stt_provider.py          # 可选 hear_voice_stt ProviderType.SPEECH_TO_TEXT 适配器（委托统一主备引擎）
└── tests/                       # 全量离线单元测试与契约验证套件
```

---

## ⚙️ 推荐配置方案

1. **保持 AstrBot 全局原生 STT 关闭**：
   请在 AstrBot 设置中保持 `provider_stt_settings.enable = false`，以免核心 `PreProcessStage` 提前用 `Plain` 覆盖 `Record` 组件。
2. **Gemini 主用 + DeepSeek 官方 ASR 备用（推荐组合）**：
   - `provider_engine`: `auto`，`provider_id`: 选择已配置的 Gemini 提供商（如 `AlmoonsGemini-1/...`）；
   - `enable_fallback`: `true`，`fallback_engine`: `auto`，`fallback_provider_id`: 选择 DeepSeek 提供商（如 `deepseek/deepseek-flash`）；
   - `on_stt_fail`: `pass`（默认，故障仅记后台脱敏日志，不打扰群聊）。
3. **DeepSeek 官方 ASR 主用 + Gemini 备用**：
   - 将主通道 `provider_id` 选为 `deepseek/deepseek-flash`（或填写独立 `api_key`），`fallback_provider_id` 选为 Gemini 提供商即可。

---

## 🧪 本地运行测试

本仓库全部单元测试均可完全离线运行：

```bash
pytest
```
