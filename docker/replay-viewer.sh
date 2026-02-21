#!/bin/bash
set -e

REPLAY_FILE="$1"
if [ -z "$REPLAY_FILE" ]; then
    echo "Usage: /replay-viewer.sh <replay_file_path>"
    exit 1
fi

if [ ! -f "$REPLAY_FILE" ]; then
    echo "ERROR: Replay file not found: $REPLAY_FILE"
    exit 1
fi

# Copy replay to the expected directory structure so OpenRA can read metadata
REPLAY_DIR="/root/.config/openra/Replays/ra/{DEV_VERSION}"
mkdir -p "$REPLAY_DIR"
REPLAY_BASENAME=$(basename "$REPLAY_FILE")
cp "$REPLAY_FILE" "$REPLAY_DIR/$REPLAY_BASENAME"
REPLAY_PATH="$REPLAY_DIR/$REPLAY_BASENAME"
echo "Replay copied to: $REPLAY_PATH"

# Start Xvfb (virtual framebuffer)
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1280x960x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
export DISPLAY=:99

# Start x11vnc
echo "Starting VNC server on port 5900..."
x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -quiet &
VNC_PID=$!
sleep 1

# Start noVNC (websockify proxy)
echo "Starting noVNC on port 6080..."
websockify --web /usr/share/novnc 6080 localhost:5900 &
NOVNC_PID=$!
sleep 1

echo ""
echo "=== Replay viewer ready ==="
echo "Open in browser: http://localhost:6080/vnc.html"
echo "Press Ctrl+C to stop"
echo ""

# Clean shutdown on signals
cleanup() {
    echo "Shutting down replay viewer..."
    kill $NOVNC_PID 2>/dev/null || true
    kill $VNC_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Launch OpenRA with rendered display (Game.Platform=Default)
exec dotnet /opt/openra/bin/OpenRA.dll \
    Engine.EngineDir=/opt/openra \
    Game.Mod=ra \
    Game.Platform=Default \
    "Launch.Replay=$REPLAY_PATH"
