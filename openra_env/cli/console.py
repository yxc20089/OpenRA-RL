"""ANSI colored console output helpers (no external deps)."""

import sys

# ANSI codes — disabled when not a TTY
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_RESET = "\033[0m" if _IS_TTY else ""
_BOLD = "\033[1m" if _IS_TTY else ""
_GREEN = "\033[32m" if _IS_TTY else ""
_YELLOW = "\033[33m" if _IS_TTY else ""
_RED = "\033[31m" if _IS_TTY else ""
_CYAN = "\033[36m" if _IS_TTY else ""
_DIM = "\033[2m" if _IS_TTY else ""


def info(msg: str) -> None:
    print(f"  {msg}")


def success(msg: str) -> None:
    print(f"  {_GREEN}{msg}{_RESET}")


def error(msg: str) -> None:
    print(f"  {_RED}{msg}{_RESET}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  {_YELLOW}{msg}{_RESET}")


def step(msg: str) -> None:
    """Print a progress step (e.g. 'Pulling image...')."""
    print(f"  {_CYAN}{msg}{_RESET}")


def header(msg: str) -> None:
    print(f"\n  {_BOLD}{msg}{_RESET}")


def dim(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")
