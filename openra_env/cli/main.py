"""CLI entry point for openra-rl."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openra-rl",
        description="Play Red Alert with AI agents",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── play ────────────────────────────────────────────────────────
    play_parser = subparsers.add_parser(
        "play", help="Run the LLM agent against the game",
    )
    play_parser.add_argument(
        "--provider", choices=["openrouter", "ollama", "lmstudio"],
        help="LLM provider (overrides saved config)",
    )
    play_parser.add_argument("--model", help="Model ID")
    play_parser.add_argument("--api-key", help="API key for LLM endpoint")
    play_parser.add_argument(
        "--difficulty", choices=["easy", "normal", "hard"], default="normal",
        help="AI opponent difficulty (default: normal)",
    )
    play_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    play_parser.add_argument("--port", type=int, default=8000, help="Game server port (default: 8000)")
    play_parser.add_argument("--server-url", help="Connect to existing server URL (skip Docker)")
    play_parser.add_argument("--local", action="store_true", help="Run server locally instead of Docker (for developers)")
    play_parser.add_argument("--version", dest="image_version", default=None, help="Docker image version to use (default: latest)")

    # ── config ──────────────────────────────────────────────────────
    subparsers.add_parser("config", help="Re-run the setup wizard")

    # ── server ──────────────────────────────────────────────────────
    server_parser = subparsers.add_parser("server", help="Manage the game server")
    server_sub = server_parser.add_subparsers(dest="server_command")

    start_parser = server_sub.add_parser("start", help="Start the game server")
    start_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    start_parser.add_argument(
        "--difficulty", choices=["easy", "normal", "hard"], default="normal",
    )
    start_parser.add_argument("--detach", action="store_true", default=True, help="Run in background (default)")

    server_sub.add_parser("stop", help="Stop the game server")
    server_sub.add_parser("status", help="Show server status")

    logs_parser = server_sub.add_parser("logs", help="Show server logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")

    # ── mcp-server ──────────────────────────────────────────────────
    mcp_parser = subparsers.add_parser("mcp-server", help="Start MCP stdio server")
    mcp_parser.add_argument("--server-url", help="Game server URL")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Game server port (default: 8000)")

    # ── replay ─────────────────────────────────────────────────────
    replay_parser = subparsers.add_parser("replay", help="Manage and watch game replays")
    replay_sub = replay_parser.add_subparsers(dest="replay_command")

    watch_parser = replay_sub.add_parser("watch", help="Watch a replay in your browser (via VNC)")
    watch_parser.add_argument("file", nargs="?", default=None, help="Replay file (local path or container path; default: latest)")
    watch_parser.add_argument("--port", type=int, default=6080, help="noVNC port (default: 6080)")

    replay_sub.add_parser("list", help="List available replays")
    replay_sub.add_parser("copy", help="Copy replays from Docker to ~/.openra-rl/replays/")
    replay_sub.add_parser("stop", help="Stop the replay viewer")

    # ── doctor ──────────────────────────────────────────────────────
    subparsers.add_parser("doctor", help="Check system prerequisites")

    # ── version ─────────────────────────────────────────────────────
    subparsers.add_parser("version", help="Print version")

    args = parser.parse_args()

    # Handle --version at top level
    if args.version:
        from openra_env.cli.commands import cmd_version
        cmd_version()
        return

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    from openra_env.cli import commands

    if args.command == "play":
        commands.cmd_play(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            difficulty=args.difficulty,
            verbose=args.verbose,
            port=args.port,
            server_url=args.server_url,
            local=args.local,
            image_version=args.image_version,
        )
    elif args.command == "config":
        commands.cmd_config()
    elif args.command == "server":
        if args.server_command == "start":
            commands.cmd_server_start(
                port=args.port,
                difficulty=args.difficulty,
                detach=args.detach,
            )
        elif args.server_command == "stop":
            commands.cmd_server_stop()
        elif args.server_command == "status":
            commands.cmd_server_status()
        elif args.server_command == "logs":
            commands.cmd_server_logs(follow=args.follow)
        else:
            server_parser.print_help()
    elif args.command == "replay":
        if args.replay_command == "watch":
            commands.cmd_replay_watch(file=args.file, port=args.port)
        elif args.replay_command == "list":
            commands.cmd_replay_list()
        elif args.replay_command == "copy":
            commands.cmd_replay_copy()
        elif args.replay_command == "stop":
            commands.cmd_replay_stop()
        else:
            replay_parser.print_help()
    elif args.command == "mcp-server":
        commands.cmd_mcp_server(
            server_url=args.server_url,
            port=args.port,
        )
    elif args.command == "doctor":
        commands.cmd_doctor()
    elif args.command == "version":
        commands.cmd_version()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
