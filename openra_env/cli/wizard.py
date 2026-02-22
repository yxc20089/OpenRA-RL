"""Interactive first-run setup wizard."""

from pathlib import Path
from typing import Optional

import yaml

from openra_env.cli.console import dim, error, header, info, success, warn

CONFIG_DIR = Path.home() / ".openra-rl"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

# Provider presets
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "needs_key": True,
        "key_help": "Get one at https://openrouter.ai/keys",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "models": [
            ("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4 (recommended)"),
            ("qwen/qwen3-coder-next", "Qwen3 Coder (budget)"),
        ],
    },
    "ollama": {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "needs_key": False,
        "default_model": "qwen3:32b",
        "models": [
            ("qwen3:32b", "Qwen3 32B (recommended)"),
            ("qwen3:4b", "Qwen3 4B (lightweight)"),
        ],
    },
    "lmstudio": {
        "name": "LM Studio",
        "base_url": "http://localhost:1234/v1/chat/completions",
        "needs_key": False,
        "default_model": "",
        "models": [],
    },
}


def _prompt(question: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    if default:
        raw = input(f"  {question} [{default}]: ").strip()
        return raw or default
    else:
        while True:
            raw = input(f"  {question}: ").strip()
            if raw:
                return raw
            error("Please enter a value.")


def _choose(question: str, options: list[tuple[str, str]], allow_custom: bool = False) -> str:
    """Present numbered options and get user choice."""
    print(f"\n  {question}")
    for i, (value, label) in enumerate(options, 1):
        print(f"    [{i}] {label}")
    if allow_custom:
        print(f"    [{len(options) + 1}] Enter custom value")

    max_choice = len(options) + (1 if allow_custom else 0)
    while True:
        raw = input("  > ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
            if allow_custom and idx == max_choice:
                return _prompt("Enter value")
        except ValueError:
            # Allow typing the value directly
            if raw:
                return raw
        error(f"Please enter a number 1-{max_choice}.")


def has_saved_config() -> bool:
    """Check if a saved config exists."""
    return CONFIG_PATH.exists()


def load_saved_config() -> Optional[dict]:
    """Load saved config if it exists."""
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def save_config(config: dict) -> None:
    """Save config to ~/.openra-rl/config.yaml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    success(f"Config saved to {CONFIG_PATH}")


def run_wizard() -> dict:
    """Run the interactive setup wizard. Returns a config dict."""
    header("Welcome to OpenRA-RL!")
    info("Let's set up your LLM provider.\n")

    # Choose provider
    provider_key = _choose(
        "Choose provider:",
        [
            ("openrouter", "OpenRouter (cloud — Claude, GPT, Qwen, Mistral, etc.)"),
            ("ollama", "Ollama (local, free)"),
            ("lmstudio", "LM Studio (local, free)"),
        ],
    )

    provider = PROVIDERS.get(provider_key, PROVIDERS["openrouter"])
    config: dict = {"provider": provider_key, "llm": {"base_url": provider["base_url"]}}

    # API key (if needed)
    if provider.get("needs_key"):
        print()
        api_key = _prompt(f"Enter your {provider['name']} API key ({provider.get('key_help', '')})")
        config["llm"]["api_key"] = api_key

    # Model selection
    if provider.get("models"):
        model = _choose(
            "Choose a model:",
            [(m, label) for m, label in provider["models"]],
            allow_custom=True,
        )
    else:
        model = _prompt("Enter model ID", default=provider.get("default_model", ""))

    config["llm"]["model"] = model

    # Ollama: warn about context window
    if provider_key == "ollama":
        print()
        warn("Tip: If you see truncation errors, increase the context window:")
        dim(f"  ollama create {model}-32k --from {model} --parameter num_ctx 32768")

    print()
    save_config(config)
    dim("Run `openra-rl config` to change these settings later.\n")

    return config


def merge_cli_into_config(
    config: dict,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Apply CLI flag overrides onto a config dict."""
    if provider and provider in PROVIDERS:
        p = PROVIDERS[provider]
        config.setdefault("llm", {})["base_url"] = p["base_url"]
        config["provider"] = provider

    if model:
        config.setdefault("llm", {})["model"] = model

    if api_key:
        config.setdefault("llm", {})["api_key"] = api_key

    return config
