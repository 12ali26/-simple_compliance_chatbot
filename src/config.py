from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONFIG_KEYS = [
    "ADMIN_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
]


@dataclass(frozen=True)
class ConfigStatus:
    admin_password: bool
    supabase: bool
    openai: bool
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


def get_config_status() -> ConfigStatus:
    supabase = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    openai = bool(os.getenv("OPENAI_API_KEY"))
    return ConfigStatus(
        admin_password=bool(os.getenv("ADMIN_PASSWORD")),
        supabase=supabase,
        openai=openai,
        semantic_search=supabase and openai,
    )
