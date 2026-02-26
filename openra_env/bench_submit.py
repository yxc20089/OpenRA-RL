"""CLI tool to upload bench export JSON to OpenRA-Bench leaderboard.

Usage:
    openra-rl bench submit result.json
    openra-rl bench submit result.json --bench-url http://localhost:7860
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BENCH_URL = "https://openra-rl-openra-bench.hf.space"


def gradio_submit(bench_url: str, data: dict, timeout: float = 30) -> str:
    """Submit data to a Gradio SSE endpoint.

    Gradio 5 uses a two-step protocol:
      1. POST /gradio_api/call/<api_name> → {"event_id": "..."}
      2. GET  /gradio_api/call/<api_name>/<event_id> → SSE stream

    Returns the result string or raises on error.
    """
    base = bench_url.rstrip("/")
    payload = {"data": [json.dumps(data)]}

    # Step 1: initiate call
    resp = httpx.post(
        f"{base}/gradio_api/call/submit",
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    event_id = resp.json().get("event_id")
    if not event_id:
        raise RuntimeError(f"No event_id in response: {resp.text[:200]}")

    # Step 2: read SSE result
    with httpx.stream(
        "GET",
        f"{base}/gradio_api/call/submit/{event_id}",
        timeout=timeout,
    ) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                result = json.loads(line[6:])
                if isinstance(result, list) and result:
                    return result[0]
                return str(result)

    raise RuntimeError("No result received from SSE stream")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload bench export JSON to OpenRA-Bench leaderboard"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to bench export JSON file",
    )
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

    print(f"Submitting {data.get('agent_name', '?')} vs {data.get('opponent', '?')}...")
    print(f"  Bench: {args.bench_url}")

    try:
        msg = gradio_submit(args.bench_url, data)
        print(f"  {msg}")
    except httpx.ConnectError:
        print(f"Error: could not connect to {args.bench_url}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
