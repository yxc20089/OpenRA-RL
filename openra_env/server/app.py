"""FastAPI application for the OpenRA-RL environment.

Creates the OpenEnv-compatible server using create_app().
"""

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_app

from openra_env.models import OpenRAAction, OpenRAObservation
from openra_env.server.openra_environment import OpenRAEnvironment

app = create_app(
    OpenRAEnvironment,
    OpenRAAction,
    OpenRAObservation,
    env_name="openra_env",
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page for the HuggingFace Space."""
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>OpenRA-RL</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; color: #333; }
  h1 { font-size: 1.8em; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
  a { color: #4a7cbc; }
  .endpoints { margin: 1em 0; }
  .endpoints li { margin: 0.4em 0; }
</style>
</head><body>
<h1>OpenRA-RL</h1>
<p>OpenEnv environment for training AI agents to play Red Alert.</p>
<ul class="endpoints">
  <li><a href="/docs">API Documentation</a></li>
  <li><a href="/health">Health Check</a></li>
  <li><a href="/schema">Environment Schema</a></li>
</ul>
<p>Connect with the Python client:</p>
<pre><code>from openra_env.client import OpenRAEnv

async with OpenRAEnv("https://openra-rl-openra-rl.hf.space") as env:
    obs = await env.reset()
    while not obs.done:
        obs = await env.step(action)</code></pre>
<p>
  <a href="https://openra-rl.dev">Docs</a> ·
  <a href="https://github.com/yxc20089/OpenRA-RL">GitHub</a> ·
  <a href="https://huggingface.co/spaces/openra-rl/OpenRA-Bench">Leaderboard</a>
</p>
</body></html>"""


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
