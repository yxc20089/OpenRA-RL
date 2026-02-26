"""CLI tool to upload bench export JSON to OpenRA-Bench leaderboard.

Usage:
    openra-rl bench submit result.json
    openra-rl bench submit result.json --agent-name DeathBot-9000 --agent-type RL
    openra-rl bench submit result.json --replay game.orarep
    openra-rl bench submit result.json --bench-url http://localhost:7860
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

DEFAULT_BENCH_URL = "https://openra-rl-openra-bench.hf.space"


def _gradio_call(bench_url: str, api_name: str, payload: dict, timeout: float = 30) -> str:
    """Call a Gradio SSE endpoint (two-step protocol).

    1. POST /gradio_api/call/<api_name> → {"event_id": "..."}
    2. GET  /gradio_api/call/<api_name>/<event_id> → SSE stream
    """
    base = bench_url.rstrip("/")

    resp = httpx.post(
        f"{base}/gradio_api/call/{api_name}",
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    event_id = resp.json().get("event_id")
    if not event_id:
        raise RuntimeError(f"No event_id in response: {resp.text[:200]}")

    with httpx.stream(
        "GET",
        f"{base}/gradio_api/call/{api_name}/{event_id}",
        timeout=timeout,
    ) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                result = json.loads(line[6:])
                if isinstance(result, list) and result:
                    return result[0]
                return str(result)

    raise RuntimeError("No result received from SSE stream")


def gradio_upload_file(bench_url: str, file_path: str, timeout: float = 30) -> dict:
    """Upload a file to a Gradio app and return the file reference.

    Returns a dict like {"path": "...", "orig_name": "...", "size": ...}
    that can be passed as a file input in a Gradio API call.
    """
    base = bench_url.rstrip("/")
    path = Path(file_path)

    with open(path, "rb") as f:
        resp = httpx.post(
            f"{base}/gradio_api/upload",
            files={"files": (path.name, f)},
            timeout=timeout,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"File upload failed: HTTP {resp.status_code}: {resp.text[:200]}")

    paths = resp.json()
    if not paths:
        raise RuntimeError("File upload returned empty response")

    return {
        "path": paths[0],
        "orig_name": path.name,
        "size": path.stat().st_size,
    }


def gradio_submit(
    bench_url: str,
    data: dict,
    replay_path: str = "",
    timeout: float = 30,
) -> str:
    """Submit bench results (and optional replay) to the Gradio leaderboard.

    If replay_path points to an existing file, uploads it and uses
    the submit_with_replay endpoint. Otherwise uses the JSON-only submit.
    """
    if replay_path and Path(replay_path).is_file():
        file_ref = gradio_upload_file(bench_url, replay_path, timeout=timeout)
        return _gradio_call(
            bench_url,
            "submit_with_replay",
            {"data": [json.dumps(data), file_ref]},
            timeout=timeout,
        )

    return _gradio_call(
        bench_url,
        "submit",
        {"data": [json.dumps(data)]},
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload bench export JSON to OpenRA-Bench leaderboard"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to bench export JSON file",
    )
    parser.add_argument("--agent-name", default=None, help="Override agent name in the submission")
    parser.add_argument("--agent-type", default=None, help="Override agent type (Scripted/LLM/RL)")
    parser.add_argument("--agent-url", default=None, help="GitHub/project URL for the agent")
    parser.add_argument("--replay", default=None, help="Path to .orarep replay file")
    parser.add_argument(
        "--bench-url",
        default=DEFAULT_BENCH_URL,
        help=f"Bench leaderboard URL (default: {DEFAULT_BENCH_URL})",
    )
    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: file not found: {args.json_file}")
        sys.exit(1)

    try:
        data = json.loads(args.json_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}")
        sys.exit(1)

    # Apply CLI overrides
    if args.agent_name:
        data["agent_name"] = args.agent_name
    if args.agent_type:
        data["agent_type"] = args.agent_type
    if args.agent_url:
        data["agent_url"] = args.agent_url

    print(f"Submitting {data.get('agent_name', '?')} vs {data.get('opponent', '?')}...")
    print(f"  Bench: {args.bench_url}")

    try:
        msg = gradio_submit(args.bench_url, data, replay_path=args.replay or "")
        print(f"  {msg}")
    except httpx.ConnectError:
        print(f"Error: could not connect to {args.bench_url}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
