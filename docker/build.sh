#!/bin/bash
# Build the OpenRA-RL Docker image.
#
# This script assembles the build context by copying the OpenRA source
# into the OpenRA-RL directory (Docker can't access files outside context).
#
# Usage:
#   ./docker/build.sh                             # Auto-detect ../OpenRA
#   OPENRA_DIR=/path/to/OpenRA ./docker/build.sh  # Specify OpenRA path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENRA_DIR="${OPENRA_DIR:-$PROJECT_DIR/OpenRA}"

if [ ! -d "$OPENRA_DIR" ]; then
    echo "ERROR: OpenRA source not found at $OPENRA_DIR"
    echo "Run: git submodule update --init"
    exit 1
fi

if [ ! -f "$OPENRA_DIR/OpenRA.sln" ]; then
    echo "ERROR: $OPENRA_DIR doesn't look like an OpenRA repo (no OpenRA.sln)"
    exit 1
fi

echo "=== OpenRA-RL Docker Build ==="
echo "OpenRA source: $OPENRA_DIR"
echo "Project dir:   $PROJECT_DIR"
echo ""

# If OpenRA source is external (not the submodule), copy it into build context
REAL_OPENRA="$(cd "$OPENRA_DIR" && pwd)"
REAL_SUBMODULE="$(cd "$PROJECT_DIR/OpenRA" 2>/dev/null && pwd || echo "")"
if [ "$REAL_OPENRA" != "$REAL_SUBMODULE" ]; then
    echo "Copying OpenRA source into build context..."
    rsync -a --delete \
        --exclude='.git' \
        --exclude='bin/' \
        --exclude='*/obj/' \
        --exclude='*.user' \
        "$OPENRA_DIR/" "$PROJECT_DIR/OpenRA/"
fi

echo "Building Docker image..."
docker build -t openra-rl "$PROJECT_DIR" "$@"

echo ""
echo "=== Build complete ==="
echo "Run with: docker run -p 8000:8000 openra-rl"
