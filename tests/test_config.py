import os
from pathlib import Path

from src.config import get_config_status, load_dotenv_file, sync_mapping_to_environ
from src.openai_service import get_openai_settings
from src.supabase_store import get_supabase_settings


CONFIG_ENV_KEYS = [
    "ADMIN_PASSWORD",
    "STAFF_PASSWORD",
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


def clear_config(monkeypatch):
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_dotenv_file_sets_missing_values(tmp_path: Path, monkeypatch):
    clear_config(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ADMIN_PASSWORD=secret",
                "STAFF_PASSWORD=staff-secret",
                "SUPABASE_URL=https://example.supabase.co",
                "SUPABASE_SECRET_KEY=sb_secret_real",
                "OPENAI_API_KEY=openai-key",
                "OPENAI_CHAT_MODEL=gpt-5.4-mini",
                "OPENROUTER_API_KEY=sk-or-real",
                "OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_dotenv_file(tmp_path)
    status = get_config_status()

    assert loaded is True
    assert os.environ["ADMIN_PASSWORD"] == "secret"
    assert os.environ["STAFF_PASSWORD"] == "staff-secret"
    assert status.admin_password is True
    assert status.staff_password is True
    assert status.supabase is True
    assert status.openai is True
    assert status.openrouter is True
    assert status.model_access is True
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
            "STAFF_PASSWORD": "staff-from-secrets",
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_from_secrets",
        }
    )

    status = get_config_status()
    assert os.environ["ADMIN_PASSWORD"] == "from-secrets"
    assert os.environ["STAFF_PASSWORD"] == "staff-from-secrets"
    assert status.admin_password is True
    assert status.staff_password is True
    assert status.supabase is True
    assert status.openai is False
    assert status.openrouter is False
    assert status.model_access is False
    assert status.semantic_search is False


def test_placeholders_do_not_count_as_configured(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me")
    monkeypatch.setenv("STAFF_PASSWORD", "your-staff-password")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "PASTE_SUPABASE_SECRET_KEY_HERE")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "PASTE_SERVICE_ROLE_KEY_HERE")
    monkeypatch.setenv("OPENAI_API_KEY", "PASTE_OPENAI_API_KEY_HERE")
    monkeypatch.setenv("OPENROUTER_API_KEY", "PASTE_OPENROUTER_API_KEY_HERE")

    status = get_config_status()

    assert status.admin_password is False
    assert status.staff_password is False
    assert status.supabase is False
    assert status.openai is False
    assert status.openrouter is False
    assert status.model_access is False
    assert status.semantic_search is False
    assert get_supabase_settings() is None
    assert get_openai_settings() is None


def test_supabase_secret_key_configures_supabase(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_real")

    status = get_config_status()
    settings = get_supabase_settings()

    assert status.supabase is True
    assert settings is not None
    assert settings.backend_key == "sb_secret_real"


def test_legacy_service_role_key_still_configures_supabase(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "legacy-service-role")

    status = get_config_status()
    settings = get_supabase_settings()

    assert status.supabase is True
    assert settings is not None
    assert settings.backend_key == "legacy-service-role"


def test_supabase_secret_key_wins_over_legacy_service_role(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_preferred")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "legacy-service-role")

    settings = get_supabase_settings()

    assert settings is not None
    assert settings.backend_key == "sb_secret_preferred"


def test_publishable_key_is_not_accepted_as_backend_key(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_publishable_not_backend")

    status = get_config_status()

    assert status.supabase is False
    assert get_supabase_settings() is None


def test_openrouter_key_enables_model_access_without_semantic_search(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

    status = get_config_status()

    assert status.openrouter is True
    assert status.model_access is True
    assert status.openai is False
    assert status.semantic_search is False


def test_openrouter_and_supabase_enable_semantic_search(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

    status = get_config_status()

    assert status.openrouter is True
    assert status.openai is False
    assert status.model_access is True
    assert status.semantic_search is True


def test_openrouter_and_openai_enable_semantic_model_access(monkeypatch):
    clear_config(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_real")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

    status = get_config_status()

    assert status.openrouter is True
    assert status.openai is True
    assert status.model_access is True
    assert status.semantic_search is True
