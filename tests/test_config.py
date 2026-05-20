import os
from pathlib import Path

from src.config import get_config_status, load_dotenv_file, sync_mapping_to_environ


CONFIG_ENV_KEYS = [
    "ADMIN_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
]


def clear_config(monkeypatch):
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_dotenv_file_sets_missing_values(tmp_path: Path, monkeypatch):
    clear_config(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ADMIN_PASSWORD=secret",
                "SUPABASE_URL=https://example.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY=service-role",
                "OPENAI_API_KEY=openai-key",
                "OPENAI_CHAT_MODEL=gpt-5.4-mini",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_dotenv_file(tmp_path)
    status = get_config_status()

    assert loaded is True
    assert os.environ["ADMIN_PASSWORD"] == "secret"
    assert status.admin_password is True
    assert status.supabase is True
    assert status.openai is True
    assert status.semantic_search is True


def test_existing_environment_values_are_not_overridden(tmp_path: Path, monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("ADMIN_PASSWORD", "from-env")
    (tmp_path / ".env").write_text("ADMIN_PASSWORD=from-dotenv\n", encoding="utf-8")

    load_dotenv_file(tmp_path)

    assert os.environ["ADMIN_PASSWORD"] == "from-env"


def test_streamlit_secret_style_mapping_fills_missing_values(monkeypatch):
    clear_config(monkeypatch)

    sync_mapping_to_environ(
        {
            "ADMIN_PASSWORD": "from-secrets",
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "role",
        }
    )

    status = get_config_status()
    assert os.environ["ADMIN_PASSWORD"] == "from-secrets"
    assert status.admin_password is True
    assert status.supabase is True
    assert status.openai is False
    assert status.semantic_search is False
