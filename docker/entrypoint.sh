#!/bin/bash
set -e

# Start Xvfb (virtual framebuffer) for headless display
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
sleep 2
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb started (PID: $XVFB_PID)"

export DISPLAY=:99

# Clean shutdown on signals
cleanup() {
    echo "Shutting down..."
    kill $XVFB_PID 2>/dev/null || true
    wait $XVFB_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Execute the main command (uvicorn by default)
echo "Starting OpenRA-RL environment server..."
exec "$@"
