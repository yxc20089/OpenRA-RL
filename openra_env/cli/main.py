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

    # Strategic directives (human-as-high-command)
    play_parser.add_argument(
        "--strategy",
        help="Strategic directives: 'rush', 'turtle', 'balanced', or path to custom YAML file"
    )
    play_parser.add_argument(
        "--directive", action="append", dest="directives",
        help="Add individual directive (can be used multiple times). Example: --directive 'Maintain 2 harvesters'"
    )

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
    watch_parser.add_argument(
        "--resolution", default=None,
        help="Replay viewer resolution WxH (default: 1280x960)",
    )
    watch_parser.add_argument(
        "--render", dest="render_mode", choices=["auto", "gpu", "cpu"], default=None,
        help="Render backend: auto tries GPU then CPU (default: auto)",
    )
    watch_parser.add_argument(
        "--vnc-quality", type=int, default=None,
        help="VNC quality 0-9, higher = sharper (default: 8)",
    )
    watch_parser.add_argument(
        "--vnc-compression", type=int, default=None,
        help="VNC compression 0-9, higher = smaller (default: 4)",
    )
    watch_parser.add_argument(
        "--cpus", type=int, default=None,
        help="CPU cores for software rendering (default: 4, 0 = all available).",
    )

    replay_sub.add_parser("list", help="List available replays")
    replay_sub.add_parser("copy", help="Copy replays from Docker to ~/.openra-rl/replays/")
    replay_sub.add_parser("stop", help="Stop the replay viewer")

    arena_parser = subparsers.add_parser("arena", help="Replay comparison and preference tools")
    arena_sub = arena_parser.add_subparsers(dest="arena_command")

    arena_compare = arena_sub.add_parser("compare", help="Compare two replays side by side")
    arena_compare.add_argument("left", nargs="?", default=None, help="Run/replay ref (default: newest saved runs)")
    arena_compare.add_argument("right", nargs="?", default=None, help="Run/replay ref (default: newest saved runs)")
    arena_compare.add_argument("--port", type=int, default=8090, help="Arena UI port (default: 8090)")
    arena_compare.add_argument("--left-port", type=int, default=6080, help="Left replay noVNC port (default: 6080)")
    arena_compare.add_argument("--right-port", type=int, default=6081, help="Right replay noVNC port (default: 6081)")
    arena_compare.add_argument(
        "--resolution", default=None,
        help="Replay viewer resolution WxH (default: 1280x960)",
    )
    arena_compare.add_argument(
        "--render", dest="render_mode", choices=["auto", "gpu", "cpu"], default=None,
        help="Render backend: auto tries GPU then CPU (default: auto)",
    )
    arena_compare.add_argument(
        "--vnc-quality", type=int, default=None,
        help="VNC quality 0-9, higher = sharper (default: 8)",
    )
    arena_compare.add_argument(
        "--vnc-compression", type=int, default=None,
        help="VNC compression 0-9, higher = smaller (default: 4)",
    )
    arena_compare.add_argument(
        "--cpus", type=int, default=None,
        help="CPU cores for software rendering (default: 4, 0 = all available).",
    )

    arena_export = arena_sub.add_parser("export", help="Export saved preference pairs as JSONL")
    arena_export.add_argument("--output", default=None, help="Output JSONL path")
    arena_sub.add_parser("stop", help="Stop both arena replay viewers")

    # ── bench ─────────────────────────────────────────────────────────
    bench_parser = subparsers.add_parser("bench", help="Benchmark leaderboard tools")
    bench_sub = bench_parser.add_subparsers(dest="bench_command")

    bench_submit_parser = bench_sub.add_parser("submit", help="Upload game result JSON to the leaderboard")
    bench_submit_parser.add_argument("json_file", type=str, help="Path to bench export JSON file")
    bench_submit_parser.add_argument("--agent-name", default=None, help="Override agent name")
    bench_submit_parser.add_argument("--agent-type", default=None, help="Override agent type (Scripted/LLM/RL)")
    bench_submit_parser.add_argument("--agent-url", default=None, help="GitHub/project URL")
    bench_submit_parser.add_argument("--replay", default=None, help="Path to .orarep replay file")
    bench_submit_parser.add_argument(
        "--bench-url", default=None,
        help="Bench leaderboard URL (default: https://openra-rl-openra-bench.hf.space)",
    )

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
            strategy=getattr(args, 'strategy', None),
            directives=getattr(args, 'directives', None),
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
            commands.cmd_replay_watch(
                file=args.file,
                port=args.port,
                resolution=args.resolution,
                render_mode=args.render_mode,
                vnc_quality=args.vnc_quality,
                vnc_compression=args.vnc_compression,
                cpu_cores=args.cpus,
            )
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
    elif args.command == "arena":
        if args.arena_command == "compare":
            commands.cmd_arena_compare(
                left=args.left,
                right=args.right,
                port=args.port,
                left_port=args.left_port,
                right_port=args.right_port,
                resolution=args.resolution,
                render_mode=args.render_mode,
                vnc_quality=args.vnc_quality,
                vnc_compression=args.vnc_compression,
                cpu_cores=args.cpus,
            )
        elif args.arena_command == "export":
            commands.cmd_arena_export(output=args.output)
        elif args.arena_command == "stop":
            commands.cmd_arena_stop()
        else:
            arena_parser.print_help()
    elif args.command == "bench":
        if args.bench_command == "submit":
            from openra_env.bench_submit import main as bench_submit_main
            # Patch sys.argv so bench_submit's argparse sees the right args
            submit_argv = ["openra-rl bench submit", args.json_file]
            if args.agent_name:
                submit_argv += ["--agent-name", args.agent_name]
            if args.agent_type:
                submit_argv += ["--agent-type", args.agent_type]
            if args.agent_url:
                submit_argv += ["--agent-url", args.agent_url]
            if args.replay:
                submit_argv += ["--replay", args.replay]
            if args.bench_url:
                submit_argv += ["--bench-url", args.bench_url]
            sys.argv = submit_argv
            bench_submit_main()
        else:
            bench_parser.print_help()
    elif args.command == "doctor":
        commands.cmd_doctor()
    elif args.command == "version":
        commands.cmd_version()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
