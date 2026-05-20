from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONFIG_KEYS = [
    "ADMIN_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_EMBEDDING_MODEL",
    "OPENROUTER_SITE_URL",
    "OPENROUTER_APP_NAME",
]

PLACEHOLDER_MARKERS = {
    "",
    "change-me",
    "your-secret-key",
    "your-service-role-key",
    "your-openai-api-key",
    "your-openrouter-api-key",
    "paste_supabase_secret_key_here",
    "paste_service_role_key_here",
    "paste_openai_api_key_here",
    "paste_openrouter_api_key_here",
}


@dataclass(frozen=True)
class ConfigStatus:
    admin_password: bool
    supabase: bool
    openai: bool
    openrouter: bool
    model_access: bool
    semantic_search: bool


def load_dotenv_file(root: Path) -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    env_path = root / ".env"
    if not env_path.exists():
        return False
    load_dotenv(env_path, override=False)
    return True


def sync_mapping_to_environ(values: Mapping[str, object]) -> None:
    for key in CONFIG_KEYS:
        if key not in os.environ and key in values:
            os.environ[key] = str(values[key])


def is_real_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in PLACEHOLDER_MARKERS and not normalized.startswith("your-")


def get_config_status() -> ConfigStatus:
    supabase_backend_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase = bool(
        is_real_value(os.getenv("SUPABASE_URL"))
        and is_real_value(supabase_backend_key)
        and not str(supabase_backend_key).strip().lower().startswith("sb_publishable_")
    )
    openai = is_real_value(os.getenv("OPENAI_API_KEY"))
    openrouter = is_real_value(os.getenv("OPENROUTER_API_KEY"))
    model_access = openai or openrouter
    return ConfigStatus(
        admin_password=is_real_value(os.getenv("ADMIN_PASSWORD")),
        supabase=supabase,
        openai=openai,
        openrouter=openrouter,
        model_access=model_access,
        semantic_search=supabase and model_access,
    )
