"""CLI tool to upload bench export JSON to OpenRA-Bench leaderboard.

Usage:
    python -m openra_env.bench_submit result.json
    python -m openra_env.bench_submit result.json --bench-url http://localhost:7860
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BENCH_URL = "https://openra-rl-openra-bench.hf.space"


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
        resp = httpx.post(
            f"{args.bench_url.rstrip('/')}/api/submit",
            json={"data": [json.dumps(data)]},
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            msg = result.get("data", [""])[0] if isinstance(result.get("data"), list) else str(result)
            print(f"  {msg}")
        else:
            print(f"Error: HTTP {resp.status_code}")
            print(f"  {resp.text[:200]}")
            sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: could not connect to {args.bench_url}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
