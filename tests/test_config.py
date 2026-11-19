"""Unit tests for configuration validation, coercion, dual slots, aliases, and redaction."""

from __future__ import annotations

import os

from hear_voice.config import HearVoiceConfig, redact_secret, redact_url, sanitize_error_message


def test_defaults_are_safe_and_pass_is_default_failure_policy():
    cfg, notices = HearVoiceConfig.from_mapping_detailed({})
    assert notices == []
    assert cfg.enable_voice is True
    assert cfg.provider_engine == "auto"
    assert cfg.api_format == "auto"
    assert cfg.enable_fallback is True
    assert cfg.fallback_engine == "auto"
    assert cfg.fallback_api_format == "auto"
    assert cfg.on_stt_fail == "pass"
    assert cfg.enable_private is True
    assert cfg.enable_group_voice is True
    assert cfg.output_mode == "rich"
    assert cfg.enable_model_normalize is False
    assert cfg.enable_text_normalize is False
    assert cfg.block_private_network is True
    assert cfg.strict_local_path_check is True
    assert cfg.allow_remote_audio_url is False
    assert cfg.register_stt_provider is False
    assert cfg.stt_provider_id == "hear_voice_stt"
    assert cfg.stt_max_concurrency == 3


def test_numeric_and_enum_coercion_with_notices():
    cfg, notices = HearVoiceConfig.from_mapping_detailed(
        {
            "stt_max_concurrency": 99,
            "max_audio_mb": 0,
            "group_voice_reply_probability": -0.5,
            "provider_engine": "bad_engine",
            "fallback_engine": "unknown_fb",
            "api_format": "invalid_mode",
            "auth_mode": "unknown",
            "output_mode": "bad",
            "on_stt_fail": "explode",
            "enable_private": "false",
            "enable_model_normalize": "true",
        }
    )
    assert cfg.stt_max_concurrency == 10
    assert cfg.max_audio_mb == 1
    assert cfg.group_voice_reply_probability == 0.0
    assert cfg.provider_engine == "auto"
    assert cfg.fallback_engine == "auto"
    assert cfg.api_format == "auto"
    assert cfg.auth_mode == "auto"
    assert cfg.output_mode == "rich"
    assert cfg.on_stt_fail == "pass"
    assert cfg.enable_private is False
    assert cfg.enable_model_normalize is True
    assert len(notices) == 9


def test_legacy_aliases_and_slot_extraction():
    cfg = HearVoiceConfig.from_mapping(
        {
            "group_voice_whitelist": [123456, "789012"],
            "enable_private_voice": False,
            "user_whitelist": [10001, "10002"],
            "api_mode": "openai_compat",
            "api_url": "https://relay.example.com/v1",
            "proxy": "http://127.0.0.1:7890",
            "fallback_engine": "deepseek_asr",
            "fallback_provider_id": "deepseek/deepseek-flash",
            "fallback_api_key": "sk-fb-9999",
        }
    )
    assert cfg.group_whitelist == ["123456", "789012"]
    assert cfg.user_whitelist == ["10001", "10002"]
    assert cfg.enable_private is False
    assert cfg.api_format == "openai_compat"
    assert cfg.api_mode == "openai_compat"
    assert cfg.api_base == "https://relay.example.com/v1"
    assert cfg.api_url == "https://relay.example.com/v1"

    primary = cfg.get_slot_raw("primary")
    fallback = cfg.get_slot_raw("fallback")
    assert primary.slot == "primary"
    assert primary.api_format == "openai_compat"
    assert fallback.slot == "fallback"
    assert fallback.engine == "deepseek_asr"
    assert fallback.provider_id == "deepseek/deepseek-flash"
    assert fallback.proxy == "http://127.0.0.1:7890"


def test_redact_secret_url_and_error_message():
    assert redact_secret("") == ""
    assert redact_secret("short") == "***"
    assert redact_secret("sk-very-secret-token-9876") == "***9876"
    assert "secret" not in redact_secret("sk-very-secret-token-9876")

    url = "https://user:pass@cdn.example.com:8443/audio/1.amr?sig=abc123&token=xyz#frag"
    cleaned = redact_url(url)
    assert cleaned == "https://cdn.example.com:8443/audio/1.amr"
    assert "sig=" not in cleaned
    assert "pass" not in cleaned

    err = sanitize_error_message(
        '{"error": {"code": 401, "message": "Bad Bearer my-tok-999 and ?key=AIzaAbcDefGhiJklMnO"}}',
        secrets=("my-tok-999",),
    )
    assert "my-tok-999" not in err
    assert "AIzaAbcDefGhiJklMnO" not in err
    assert "401" in err


def test_allowed_dirs_includes_user_dir(tmp_path):
    custom_dir = tmp_path / "qq_audio"
    custom_dir.mkdir()
    cfg = HearVoiceConfig.from_mapping({"local_audio_allowed_dirs": [str(custom_dir)]})
    assert os.path.realpath(str(custom_dir)) in cfg.local_audio_allowed_dirs
