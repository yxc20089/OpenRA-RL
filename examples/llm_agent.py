#!/usr/bin/env python3
"""LLM agent that plays Red Alert using any OpenAI-compatible model.

Supports OpenRouter, Ollama, LM Studio, or any local/remote endpoint
that implements the OpenAI Chat Completions API with tool calling.

Usage:
    # With OpenRouter (cloud)
    export OPENROUTER_API_KEY=sk-or-...
    python examples/llm_agent.py --verbose

    # With a YAML config file
    python examples/llm_agent.py --config examples/config-ollama.yaml

    # With LM Studio (local, no API key needed)
    python examples/llm_agent.py --base-url http://localhost:1234/v1/chat/completions --model my-model
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from openra_env.config import load_config
from openra_env.agent import run_agent

# Re-export for backwards compatibility
from openra_env.agent import (  # noqa: F401
    SYSTEM_PROMPT,
    compose_pregame_briefing,
    format_state_briefing,
    mcp_tools_to_openai,
    _sanitize_messages,
    chat_completion,
    compress_history,
)

# Line-buffered stdout so output is observable in real time
sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="LLM agent that plays Red Alert via any OpenAI-compatible model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --config examples/config-ollama.yaml --verbose\n"
            "  %(prog)s --api-key sk-or-... --verbose\n"
            "  %(prog)s --base-url http://localhost:1234/v1/chat/completions --model my-model\n"
        ),
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: auto-discover config.yaml)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="OpenRA-RL server URL (overrides config agent.server_url)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API endpoint URL (overrides config llm.base_url)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (overrides config llm.model)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for LLM endpoint (overrides config llm.api_key)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum LLM turns, 0 = unlimited (overrides config agent.max_turns)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Maximum wall-clock time in seconds (overrides config agent.max_time_s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed LLM reasoning and tool calls",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write all output to this log file in addition to stdout",
    )
    args = parser.parse_args()

    # Build config: YAML file + env vars + CLI overrides (CLI wins over .env)
    cli: dict = {}
    if args.url is not None:
        cli.setdefault("agent", {})["server_url"] = args.url
    if args.base_url is not None:
        cli.setdefault("llm", {})["base_url"] = args.base_url
    if args.model is not None:
        cli.setdefault("llm", {})["model"] = args.model
    if args.api_key is not None:
        cli.setdefault("llm", {})["api_key"] = args.api_key
    if args.max_turns is not None:
        cli.setdefault("agent", {})["max_turns"] = args.max_turns
    if args.max_time is not None:
        cli.setdefault("agent", {})["max_time_s"] = args.max_time
    if args.verbose:
        cli.setdefault("agent", {})["verbose"] = True
    if args.log_file is not None:
        cli.setdefault("agent", {})["log_file"] = args.log_file

    config = load_config(config_path=args.config, cli_overrides=cli)
    verbose = config.agent.verbose

    # Set up logging to file if requested — tee all print() to both stdout and file
    if config.agent.log_file:
        import builtins
        _builtin_print = builtins.print
        _log_fh = open(config.agent.log_file, "w")

        def _tee_print(*pargs, **kwargs):
            _builtin_print(*pargs, **kwargs)
            kwargs.pop("file", None)
            _builtin_print(*pargs, file=_log_fh, **kwargs)
            _log_fh.flush()

        builtins.print = _tee_print

    # API key validation: only required for remote endpoints
    is_local = any(h in config.llm.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not config.llm.api_key and not is_local:
        print("Error: API key required for remote LLM endpoints.")
        print("  Set OPENROUTER_API_KEY or LLM_API_KEY environment variable, use --api-key,")
        print("  or use a config file with llm.api_key set.")
        print("  For local models (Ollama, LM Studio), use --base-url http://localhost:...")
        sys.exit(1)

    try:
        asyncio.run(run_agent(config, verbose))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except ConnectionRefusedError:
        print(f"\nCould not connect to {config.agent.server_url}")
        print("Is the OpenRA-RL server running?")
        print("  docker run -p 8000:8000 openra-rl")
        sys.exit(1)


if __name__ == "__main__":
    main()
