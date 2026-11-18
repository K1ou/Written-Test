"""Choose and build a provider from the CLI options and the settings."""

from __future__ import annotations

from mini_agent.config import ConfigError, Settings
from mini_agent.providers.base import Provider
from mini_agent.providers.gemini import GeminiProvider
from mini_agent.providers.qwen import DEFAULT_BASE_URL, QwenProvider

PROVIDERS = ("qwen", "gemini")


def build_provider(name: str | None, model: str | None, settings: Settings) -> Provider:
    name = (name or settings.mini_agent_provider).lower()
    if name == "qwen":
        key = settings.dashscope_api_key
        if key is None:
            raise ConfigError(
                "DASHSCOPE_API_KEY is not set. Put it in the environment or in a .env file "
                "(see .env.example) to use the Qwen provider."
            )
        return QwenProvider(key, model or settings.qwen_model, settings.qwen_base_url or DEFAULT_BASE_URL)
    if name == "gemini":
        key = settings.gemini_api_key
        if key is None:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Put it in the environment or in a .env file "
                "(see .env.example) to use the Gemini provider."
            )
        chosen = model or settings.gemini_model
        if not chosen:
            raise ConfigError("No Gemini model chosen. Pass --model or set GEMINI_MODEL.")
        return GeminiProvider(key, chosen)
    raise ConfigError(f"Unknown provider {name!r}. Choose one of: {', '.join(PROVIDERS)}.")
