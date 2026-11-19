from __future__ import annotations

from .entities import ProviderMetaData, ProviderType

provider_registry: list[ProviderMetaData] = []
provider_cls_map: dict[str, ProviderMetaData] = {}


def register_provider_adapter(
    provider_type_name: str,
    desc: str,
    provider_type: ProviderType = ProviderType.CHAT_COMPLETION,
    default_config_tmpl: dict | None = None,
    provider_display_name: str | None = None,
):
    def decorator(cls):
        if provider_type_name in provider_cls_map:
            raise ValueError(f"Duplicate provider adapter: {provider_type_name}")
        pm = ProviderMetaData(
            id="default",
            model=None,
            type=provider_type_name,
            desc=desc,
            provider_type=provider_type,
            cls_type=cls,
            default_config_tmpl=default_config_tmpl,
            provider_display_name=provider_display_name,
        )
        provider_registry.append(pm)
        provider_cls_map[provider_type_name] = pm
        return cls

    return decorator
