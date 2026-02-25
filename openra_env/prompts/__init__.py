"""Default prompts and prompt loading for OpenRA-RL agents."""

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).parent


def load_default_prompt() -> str:
    """Load the default system prompt shipped with the package."""
    return (_PROMPTS_DIR / "default.txt").read_text(encoding="utf-8").strip()


def load_default_prompts_yaml() -> dict:
    """Load the default prompts YAML shipped with the package."""
    path = _PROMPTS_DIR / "default_prompts.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompts_file(prompts_file: str) -> dict:
    """Load a custom prompts YAML file.

    Returns a dict suitable for merging into PromptsConfig fields.
    Raises FileNotFoundError if the file doesn't exist.
    """
    p = Path(prompts_file).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"prompts_file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
