# `astrbot_plugin_hear_voice`（我听得见语音）对外集成契约文档 (`docs/API.md`)

本文档面向需要与前置非抢答多引擎语音转写能力集成的 AstrBot 插件开发者（包括但不限于「我会永远陪着你」`astrbot_plugin_private_companion`、「我会牢牢记住你」`astrbot_plugin_memory_companion`、`contextaware` 等）。

---

## 1. 两阶段非抢答管线时序

本插件在架构上严格遵守**只做前置语音/声学理解、绝不抢答**原则：

1. **阶段 1 — 前置消息事件钩子**：
   `@filter.event_message_type(EventMessageType.ALL, priority=300000)`
   - 早于常见伴随/记忆插件执行：
     - `private_companion.observe_companion_recall` (`priority=240000`)
     - `private_companion.observe_companion_group_early` (`priority=235000`)
     - `memory_companion` (`priority=1000`)
     - `private_companion.receive_companion_message` (`priority=-230000`)
   - **绝不调用 `event.request_llm(...)`**。
   - **主/备双槽自动切换**：先调用主识别引擎（Gemini 多模态 / DeepSeek 官方 ASR `/v1/tools/asr` / OpenAI Whisper）；若主通道不可重试失败（`4xx`）、重试耗尽（`429` / `5xx` / 超时）或返回错误载荷，则立即自动切换回退通道；主备互为对称可配。
   - 识别成功后**不停止事件传播**，交由原有回复链路与群聊接话引擎统一决策。
   - **绝不删除或移动原 `Record` 组件**。

2. **阶段 2 — LLM 请求前注入与清理钩子**：
   `@filter.on_llm_request(priority=300000)`
   - 早于 `private_companion` (`250000` / `-1000000`) 与 `memory_companion` (`-20`) 执行。
   - 仅当阶段 1 对该事件成功完成转写（`_hear_voice_done=True` / `_gemini_stt_done=True` 且状态为 `"ok"`）时激活：
     - **Gemini `rich` 声学分析路径**：将 `rich` 模式的 6 项声学分析（原话、语气/情绪、说话人/音色、语速/停顿、环境音、大意总结）、语音来源标记（直发 vs 引用）和说话人元信息打包为 `TextPart(text=...).mark_as_temp()`（即 `_no_save=True`）追加到 `req.extra_user_content_parts`（若最小调用桩无 `extra_user_content_parts` 属性，则追加带 `_no_save=True` 的临时消息至 `req.contexts`）。
     - **纯转写引擎路径（`deepseek_asr` / `openai_whisper`）**：即使全局配置 `output_mode="rich"`，只要最终由纯转写引擎成功完成，本插件绝不编造 6 项声学模板，也不注入任何空的 `[语音转写分析]` 临时 `TextPart`，仅以干净纯原话 `[语音消息]: <text>` 进入消息链与 `req.prompt`。
     - **冗余附件清理**：无论由哪个引擎转写成功，都会自动清理 `req.audio_urls`、以及 `req.extra_user_content_parts` / `req.contexts` 中的 `[Audio Attachment: ...]`、`[Audio Attachment in quoted message: ...]`、`[Voice unavailable]` 与残留 `audio_url` / `input_audio` 块，避免不支持音频模态的对话模型（如 `sta1n`）报错或被双倍计费。

---

## 2. 消息链与 `message_str` 写入规范

阶段 1 转写成功后，按以下不变量写入纯原话，确保记忆持久化不被 `rich` 模板污染，且不破坏原消息媒体组件顺序：

- **保留原 `Record` 组件及其索引位置**：
  新的 `Plain` 组件始终以 **`append`（追加到消息链末尾）** 的方式写入 `event.message_obj.message`，绝不 `insert(0, ...)`。因此原有 `Record`、`Video`、`Image`、`Reply` 组件的数组下标完全保持不变，与 `astrbot_plugin_private_companion` 中基于 `enumerate(raw_message["message"])` 的位置校验保持兼容。
- **格式规范**：
  - 单条直发语音：`[语音消息]: <纯原话>`（多条时依次为 `[语音消息 1]: ...`、`[语音消息 2]: ...`）。
  - 单条引用语音：`[引用语音消息]: <纯原话>`（多条时依次为 `[引用语音消息 1]: ...`）。
  - 同步更新 `event.message_str` 与 `event.message_obj.message_str`：若原消息仅含语音则直接置为上述文本；若同时附带用户文字（如 `@机器人 听一下`），则在原文字后以 `\n` 追加，保留原有唤醒前缀。

---

## 3. 事件 `extra` 公开字段契约（双前缀兼容导出）

为实现与旧版 `gemini_stt` 及下游 `astrbot_plugin_private_companion` 的零改造无缝兼容，本插件**同时写入规范键 `_hear_voice_*` 与兼容别名 `_gemini_stt_*`**，外加公共列表键 `astrbot_stt_transcripts`。

### 3.1 结构化与主状态契约

| `_hear_voice_*` 规范键 | `_gemini_stt_*` 兼容别名 | 类型 | 含义 |
|---|---|---|---|
| `_hear_voice_done` | `_gemini_stt_done` | `bool` | 标记本轮前置 STT 已执行完毕（用于幂等去重与跳过重复转写）。 |
| `_hear_voice_status` | `_gemini_stt_status` | `str` | `"ok"` \| `"failed"`。 |
| `_hear_voice_non_hijacking` | `_gemini_stt_non_hijacking` | `bool` | 阶段 1 干净完成转写或按 `pass` / `notify_pass` 失败放行时固定写入 `True`；明确声明本插件**未调用 `event.request_llm(...)`、未接管回复生成**。 |
| `_hear_voice_hijacked` | `_gemini_stt_hijacked` | `bool` | 阶段 1 干净完成转写或按 `pass` / `notify_pass` 失败放行时固定写入 `False`；明确声明本轮回复**未被 STT 插件抢答**。 |
| `_hear_voice_notify_only_sent` | `_gemini_stt_notify_only_sent` | `bool` | 当显式配置 `on_stt_fail="notify_pass"` 触发并成功发出失败提醒时写入 `True`（默认 `on_stt_fail="pass"` 静默记日志不发消息，不置此字段）。 |
| `_hear_voice_has_quoted` | `_gemini_stt_has_quoted` | `bool` | 本轮转写是否包含 `Reply.chain` 中的引用语音。 |
| `_hear_voice_audio_cleaned` | `_gemini_stt_audio_cleaned` | `bool` | 阶段 2 是否已清理 `req` 中的原生音频附件占位。 |
| `_hear_voice_engine_used` | — | `str` | 本轮实际产出的识别引擎（`"gemini"` \| `"deepseek_asr"` \| `"openai_whisper"`，多条混用时逗号拼接）。 |
| `_hear_voice_fallback_used` | — | `bool` | 本轮是否触发并命中了主到备的回退通道。 |
| `astrbot_stt_transcripts` | — | `list[dict]` | 结构化逐条语音转写列表，供宿主 `capture_text_only`、`group_mentions` 与 `core_reply_perception` 直接采纳。 |

`astrbot_stt_transcripts` 单条元素结构：

```python
{
    "index": 0,                    # 语音序号（按 direct -> quoted 扫描顺序）
    "source": "direct",            # "direct"（直发）| "quoted"（引用）
    "transcript": "明天记得带伞",   # 严格为提取后的纯原话文本
    "rich_text": "1) 原话转写：...", # Gemini rich 模式下为 6 项完整分析；simple 或 DeepSeek ASR/Whisper 下为空串
    "emotion": "轻快、略带提醒",     # 2) 语气/情绪（纯转写引擎为空串）
    "speaker": "单人女声，普通话",   # 3) 说话人/音色（纯转写引擎为空串）
    "speed": "语速适中，句尾上扬",   # 4) 语速/停顿（纯转写引擎为空串）
    "ambient": "室内安静",          # 5) 环境音（纯转写引擎为空串）
    "summary": "提醒明天出门带伞",   # 6) 大意总结（纯转写引擎为空串）
    "mode": "rich",                # "rich" | "simple"
    "model": "gemini-3.8-flash",   # 实际使用的模型 ID（DeepSeek ASR 为 "deepseek-asr"）
    "engine": "gemini",            # "gemini" | "deepseek_asr" | "openai_whisper"
    "fallback_used": False,        # 本条是否经过主->备切换完成
}
```

### 3.2 转写内容与门控状态契约（受 `group_voice_export_context=true` 控制）

| `_hear_voice_*` 规范键 | `_gemini_stt_*` 兼容别名 | 类型 | 含义 |
|---|---|---|---|
| `_hear_voice_transcript` | `_gemini_stt_transcript` | `str` | **严格仅放提取后的纯原话文本**（多条语音以 `\n` 拼接；无论 `simple` 还是 `rich` 模式均不包含 6 项模板前缀）。 |
| `_hear_voice_raw_text` | `_gemini_stt_raw_text` | `str` | 模型返回的原始完整结果（Gemini `rich` 模式下为 6 项声学分析；纯转写引擎下等于纯原话）。 |
| `_hear_voice_forward_text` | `_gemini_stt_forward_text` | `str` | 阶段 2 构造的临时注入上下文文本（纯转写引擎时为空串 `""`）。 |
| `_hear_voice_is_group` | `_gemini_stt_is_group` | `bool` | 是否为群聊消息。 |
| `_hear_voice_should_reply` | `_gemini_stt_should_reply` | `bool` | 仅表示下游是否**允许**由语音触发回复（私聊固定 `True`；群聊依据 `group_voice_reply_probability` 门控判定），**绝不表示本插件已调用 `request_llm`**。 |
| `_hear_voice_reply_reason` | `_gemini_stt_reply_reason` | `str` | `"private_voice"` \| `"group_passthrough"` \| `"group_probability_hit"` \| `"group_probability_miss"`。 |
| `_hear_voice_cache_only` | `_gemini_stt_cache_only` | `bool` | `not _hear_voice_should_reply`。 |

> **默认静默放行与非抢答契约（必读）**：
> - 本插件默认 **`on_stt_fail = "pass"`**：当主、备通道全部失败时，仅在后台日志输出脱敏 `logger.warning("[HearVoice] ...")`，**不向 QQ 群/私聊发送任何报错消息**，并写入 `_hear_voice_status="failed"` / `_gemini_stt_status="failed"`、`_hear_voice_non_hijacking=True` / `_gemini_stt_non_hijacking=True`、`_hear_voice_hijacked=False` / `_gemini_stt_hijacked=False`，静默放行后续插件处理。
> - `block` / `notify` 为显式调试/强管控选项，会调用 `event.stop_event()` 中止传播；`notify_pass` 仅在显式开启时发提醒并放行。

推荐读取方式：

```python
structured = event.get_extra("astrbot_stt_transcripts") or []
transcript = (
    event.get_extra("_hear_voice_transcript", "")
    or event.get_extra("_gemini_stt_transcript", "")
    or event.get_extra("_hear_voice_raw_text", "")
    or event.get_extra("_gemini_stt_raw_text", "")
)
```

---

## 4. 宿主侧防双处理与 `provider_stt_settings` 约定

1. **保持 AstrBot 全局 `provider_stt_settings.enable: false`**：
   否则 AstrBot `PreProcessStage` 会在插件阶段之前直接将 `Record` 替换为 `Plain`，绕过本插件的主备切换、群白名单、失败策略与 `rich` 声学分析，并破坏原生 `Record` 契约。
2. **与「我会永远陪着你」协作**：
   双前缀写入确保 `astrbot_plugin_private_companion` 无需额外改动即可直接识别 `_gemini_stt_*` 别名与 `astrbot_stt_transcripts`。
3. **与「我会牢牢记住你」协作**：
   `memory_companion`（`priority=1000`）运行在阶段 1 之后，从 `Plain` 与 `message_str` 中读到的仅是 `[语音消息]: <纯原话>`，长期记忆库始终纯净无污染。
