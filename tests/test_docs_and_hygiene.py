"""Documentation, schema integrity, and repository hygiene checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_metadata_yaml_and_versions_stay_in_sync():
    meta = yaml.safe_load((REPO / "metadata.yaml").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")

    for req_key in ("name", "display_name", "desc", "version", "author", "repo", "astrbot_version"):
        assert meta.get(req_key), f"Missing metadata field: {req_key}"

    assert meta["name"] == "astrbot_plugin_hear_voice"
    assert meta["display_name"] == "我听得见语音"
    assert meta["repo"] == "https://github.com/JosephTian876/astrbot_plugin_hear_voice"
    assert meta["version"] == "0.1.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(meta["version"]))
    assert f"当前版本 **{meta['version']}**" in readme
    assert f"## {meta['version']}" in changelog


def test_conf_schema_valid_types_labels_defaults_and_secret_flags():
    schema = json.loads((REPO / "_conf_schema.json").read_text(encoding="utf-8"))
    valid_types = {"string", "text", "bool", "int", "float", "list", "dict", "object", "template_list"}

    for key, spec in schema.items():
        assert spec.get("type") in valid_types, key
        assert isinstance(spec.get("description"), str) and spec["description"].strip(), key
        assert "hint" in spec, key
        if "options" in spec:
            assert isinstance(spec.get("labels"), list), f"{key} missing bilingual labels"
            assert len(spec["labels"]) == len(spec["options"]), key
            for label in spec["labels"]:
                assert "/" in label, f"{key} label should be bilingual: {label}"

    assert schema["on_stt_fail"]["default"] == "pass"
    assert schema["api_key"].get("secret") is True
    assert schema["fallback_api_key"].get("secret") is True
    assert schema["provider_id"].get("_special") == "select_provider"
    assert schema["fallback_provider_id"].get("_special") == "select_provider"
    assert schema["enable_fallback"]["default"] is True
    assert schema["provider_engine"]["default"] == "auto"
    assert schema["fallback_engine"]["default"] == "auto"
    assert schema["stt_provider_id"]["default"] == "hear_voice_stt"


def test_docs_api_and_readme_tree_cover_all_modules_and_extras():
    api_doc = (REPO / "docs" / "API.md").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    for extra_name in (
        "_hear_voice_transcript",
        "_hear_voice_raw_text",
        "_hear_voice_forward_text",
        "_hear_voice_is_group",
        "_hear_voice_should_reply",
        "_hear_voice_reply_reason",
        "_hear_voice_cache_only",
        "_hear_voice_non_hijacking",
        "_hear_voice_hijacked",
        "_hear_voice_notify_only_sent",
        "_hear_voice_done",
        "_hear_voice_status",
        "_hear_voice_has_quoted",
        "_hear_voice_audio_cleaned",
        "_hear_voice_engine_used",
        "_hear_voice_fallback_used",
        "_gemini_stt_transcript",
        "_gemini_stt_raw_text",
        "_gemini_stt_forward_text",
        "_gemini_stt_is_group",
        "_gemini_stt_should_reply",
        "_gemini_stt_reply_reason",
        "_gemini_stt_cache_only",
        "_gemini_stt_non_hijacking",
        "_gemini_stt_hijacked",
        "_gemini_stt_notify_only_sent",
        "_gemini_stt_done",
        "_gemini_stt_status",
        "_gemini_stt_has_quoted",
        "_gemini_stt_audio_cleaned",
        "astrbot_stt_transcripts",
    ):
        assert extra_name in api_doc, extra_name

    actual_modules = {p.name for p in (REPO / "hear_voice").glob("*.py")}
    expected_modules = {
        "__init__.py",
        "config.py",
        "normalize.py",
        "audio_resolver.py",
        "gemini_client.py",
        "deepseek_client.py",
        "engine.py",
        "hook_pipeline.py",
        "stt_provider.py",
    }
    assert actual_modules == expected_modules
    for mod in actual_modules:
        assert mod in readme, f"README missing module {mod}"


def test_repo_contains_no_hardcoded_secrets_or_local_paths():
    forbidden_patterns = (
        re.compile(r"sk-[A-Za-z0-9]{16,}"),
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"C:\\\\?Users\\\\?[A-Za-z0-9_]+"),
        re.compile(r"D:\\\\?AstrBot"),
    )
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".md", ".json", ".yaml", ".yml", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pat in forbidden_patterns:
            match = pat.search(content)
            assert match is None, f"Forbidden pattern {pat.pattern} found in {path}: {match.group(0)}"
