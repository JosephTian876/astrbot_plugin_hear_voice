# 更新日志

## 0.1.0

### 新增
- 正式重命名并升级为独立插件仓库 `astrbot_plugin_hear_voice`（**我听得见语音**），全面重构包结构为 `hear_voice/`，日志前缀统一为 `[HearVoice]`，与旧版 `astrbot_plugin_gemini_stt` 零命名冲突。
- 新增 **DeepSeek 官方 ASR（`POST /v1/tools/asr`）** 完整支持（`hear_voice/deepseek_client.py` & `hear_voice/audio_resolver.py`）：
  - 支持 `normalize_deepseek_asr_url` 将空字符串、`https://api.deepseek.com`、`https://api.deepseek.com/v1` 自动规范化为 `https://api.deepseek.com/v1/tools/asr`，并完整保留自定义代理 `/tools/asr` 路径。
  - 支持自动识别 16-bit PCM WAV（`8k/16k/22.05k/24k/32k/44.1k/48k` Hz，含 `pysilk` 解码直出 WAV）与 Ogg Opus 直传；其他格式自动异步调用 `ffmpeg` 转码为 `16kHz` 单声道 `pcm_s16le` WAV，若 WAV 体积超出 `8 MiB` 自动回退 `libopus` Ogg，并严格校验 `<= 120s` 与 `<= 8 MiB` 硬限制。
- 新增 **主/备双槽多引擎自由组合与故障自动切换引擎**（`hear_voice/engine.py`）：
  - 统一抽象 `SttEndpointSpec`、`resolve_engine_kind` 与共享 HTTP 重试/脱敏传输层 `SttHttpTransport`，同时支持 `gemini`、`deepseek_asr` 与 `openai_whisper` 三种引擎。
  - 主通道与回退通道均支持通过 `_special: select_provider` 复用任意 AstrBot Provider（如 `AlmoonsGemini-1/...` 或 `deepseek/deepseek-flash`）或填写独立接口配置；主识别失败（不可重试 `4xx`、重试耗尽 `429`/`5xx`/超时或 HTTP 200 错误载荷）时记录脱敏告警并立即切换回退通道。
  - 当最终由 `deepseek_asr` / `openai_whisper` 等纯转写引擎成功产出时，即使 `output_mode="rich"` 也仅输出纯净 `[语音消息]: <原话>`，绝不编造 6 项声学模板，也不注入空的临时分析 `TextPart`。
- 调整默认失败策略 **`on_stt_fail="pass"`**：主备均失败时仅记录脱敏警告日志并静默放行下游插件，不向 QQ 群/私聊发送报错提示；保留 `notify_pass` / `block` / `notify` 为显式可选模式。
- 导出双前缀事件 `extra` 契约：同时写入 `_hear_voice_*`（含 `_hear_voice_engine_used`、`_hear_voice_fallback_used`）与兼容别名 `_gemini_stt_*` 及 `astrbot_stt_transcripts`，无缝兼容 `astrbot_plugin_private_companion` 与 `astrbot_plugin_memory_companion`。
- 更新可选标准 STT Provider 适配器 `hear_voice_stt`（`hear_voice/stt_provider.py`），统一委托主/备切换引擎执行转写。
