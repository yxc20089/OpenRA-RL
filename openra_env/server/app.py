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


LANDING_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenRA-RL &mdash; OpenEnv Environment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Teko:wght@400;600;700&display=swap" rel="stylesheet">
<style>
/* === Reset & base === */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Share Tech Mono', monospace;
  background: radial-gradient(circle at center, #1a0505 0%, #050505 100%);
  color: #d1d5db;
  min-height: 100vh;
}
a { color: #d1d5db; text-decoration: none; transition: color .2s; }
a:hover { color: #fff; }
h1, h2, h3, h4, .font-teko {
  font-family: 'Teko', sans-serif;
  letter-spacing: 1px;
  text-transform: uppercase;
}

/* === Scanlines === */
.scanlines {
  background: linear-gradient(
    to bottom,
    rgba(255,255,255,0),
    rgba(255,255,255,0) 50%,
    rgba(0,0,0,0.2) 50%,
    rgba(0,0,0,0.2)
  );
  background-size: 100% 4px;
  position: fixed; inset: 0;
  pointer-events: none; z-index: 50;
}

/* === CRT flicker === */
.crt-flicker { animation: crt .15s infinite; }
@keyframes crt {
  0% { opacity: .98; } 50% { opacity: 1; } 100% { opacity: .99; }
}

/* === Text effects === */
.terminal-text { color: #84cc16; text-shadow: 0 0 5px rgba(132,204,22,.5); }
.alert-text    { color: #ef4444; text-shadow: 0 0 8px rgba(239,68,68,.8); }

/* === Buttons === */
.btn-soviet {
  display: inline-flex; align-items: center; gap: .5rem;
  background: #dc2626; border: 2px solid #f87171;
  box-shadow: 4px 4px 0 #000; transition: all .1s;
  color: #fff; font-family: 'Teko', sans-serif;
  font-size: 1.6rem; padding: .4rem 1.8rem;
  text-transform: uppercase; cursor: pointer;
}
.btn-soviet:hover { transform: translate(2px,2px); box-shadow: 2px 2px 0 #000; background: #ef4444; color: #fff; }

.btn-ghost {
  display: inline-flex; align-items: center; gap: .5rem;
  background: #171717; border: 2px solid #525252;
  box-shadow: 4px 4px 0 #000; transition: all .1s;
  color: #a3a3a3; font-family: 'Teko', sans-serif;
  font-size: 1.6rem; padding: .4rem 1.8rem;
  text-transform: uppercase; cursor: pointer;
}
.btn-ghost:hover { transform: translate(2px,2px); box-shadow: 2px 2px 0 #000; border-color: #737373; color: #fff; }

/* === Military card === */
.card-military {
  background: #121212; border: 2px solid #262626;
  border-left: 4px solid #dc2626;
  box-shadow: 6px 6px 0 rgba(0,0,0,.8);
  transition: all .2s; padding: 1.5rem;
}
.card-military:hover { border-color: #dc2626; transform: translateY(-4px); box-shadow: 6px 10px 0 rgba(0,0,0,.8); }
.card-military h3 { color: #fff; font-size: 1.6rem; margin-bottom: .3rem; }
.card-military p { color: #9ca3af; font-size: .85rem; margin-bottom: .8rem; }
.card-military a { color: #ef4444; font-size: .85rem; }
.card-military a:hover { color: #f87171; }

/* === Nav === */
nav {
  border-bottom: 2px solid #991b1b;
  background: rgba(0,0,0,.9);
  position: sticky; top: 0; z-index: 40;
  backdrop-filter: blur(4px);
}
.nav-inner {
  max-width: 72rem; margin: 0 auto;
  padding: 0 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
  height: 4rem;
}
.nav-logo { display: flex; align-items: center; gap: .6rem; }
.nav-logo svg { width: 2rem; height: 2rem; color: #dc2626; animation: spin 4s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.nav-logo span { font-family: 'Teko', sans-serif; font-size: 1.8rem; font-weight: 700; color: #fff; letter-spacing: .15em; }
.nav-logo .rl { color: #dc2626; }
.nav-links { display: flex; gap: 1.5rem; align-items: center; }
.nav-links a {
  font-family: 'Teko', sans-serif; font-size: 1.15rem;
  letter-spacing: .1em; color: #9ca3af;
  text-transform: uppercase;
}
.nav-links a:hover { color: #fff; }

/* === Hero === */
.hero {
  border-bottom: 4px solid #991b1b;
  padding: 5rem 1rem 6rem;
  text-align: center;
  position: relative;
}
.hero::before {
  content: ''; position: absolute; inset: 0; opacity: .5;
  mix-blend-mode: overlay; pointer-events: none;
  background-image: url('https://www.transparenttextures.com/patterns/carbon-fibre.png');
}
.hero > * { position: relative; z-index: 1; }
.hero .subtitle {
  font-size: 1.1rem; margin-bottom: 1rem;
  font-weight: 700; display: flex; align-items: center; justify-content: center; gap: .5rem;
}
.hero h1 { font-size: clamp(3rem, 10vw, 5.5rem); font-weight: 700; line-height: 1; margin-bottom: 1.2rem; }
.hero .desc {
  font-size: 1.1rem; max-width: 40rem; margin: 0 auto 2.5rem;
  line-height: 1.7; background: rgba(0,0,0,.5);
  padding: 1.2rem 1.5rem; border: 1px solid #262626; border-radius: 4px;
}
.hero .desc strong { color: #fff; display: block; margin-top: .5rem; }
.hero .buttons { display: flex; flex-wrap: wrap; gap: 1.2rem; justify-content: center; }

/* === Sections === */
.section { padding: 4rem 1rem; max-width: 72rem; margin: 0 auto; }
.section-dark { background: #050505; border-top: 1px solid #262626; border-bottom: 1px solid #262626; }
.section h2 { color: #fff; font-size: 2.4rem; margin-bottom: 1.5rem; }
.cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; }

/* === Terminal === */
.terminal { border: 1px solid #404040; border-radius: 8px; overflow: hidden; box-shadow: 0 0 30px rgba(0,0,0,1); }
.terminal-bar {
  background: #1a1a1a; padding: .5rem 1rem; border-bottom: 1px solid #404040;
  display: flex; align-items: center; gap: .5rem;
}
.terminal-dot { width: .7rem; height: .7rem; border-radius: 50%; }
.terminal-bar span { margin-left: .5rem; font-size: .75rem; color: #9ca3af; }
.terminal-body { background: #0d0d0d; padding: 1.5rem; overflow-x: auto; font-size: .85rem; line-height: 1.8; }
.terminal-body pre { margin: 0; white-space: pre; }
.t-prompt { color: #6b7280; }
.t-cmd { color: #4ade80; }
.t-kw { color: #c084fc; }
.t-str { color: #facc15; }
.t-fn { color: #38bdf8; }
.t-comment { color: #6b7280; }
.t-plain { color: #d1d5db; }

/* === Footer === */
footer {
  background: #000; border-top: 2px solid #7f1d1d;
  margin-top: 4rem; padding: 3rem 1.5rem;
}
.footer-inner { max-width: 72rem; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 2rem; justify-content: space-between; }
.footer-brand { max-width: 24rem; }
.footer-brand .logo { font-family: 'Teko', sans-serif; font-size: 2rem; font-weight: 700; color: #fff; letter-spacing: .15em; display: flex; align-items: center; gap: .5rem; }
.footer-brand .logo .rl { color: #dc2626; }
.footer-status { margin-top: 1rem; font-size: .8rem; color: #84cc16; text-shadow: 0 0 5px rgba(132,204,22,.5); line-height: 1.8; }
.footer-col h3 { font-family: 'Teko', sans-serif; font-size: 1.4rem; color: #fff; margin-bottom: .8rem; }
.footer-col ul { list-style: none; }
.footer-col li { margin-bottom: .5rem; }
.footer-col a { font-size: .85rem; display: flex; align-items: center; gap: .4rem; }
.footer-col a:hover { color: #ef4444; }
.chevron::before { content: '\\203A'; margin-right: .3rem; }

/* === Fade in === */
.fade-in { animation: fadeIn .7s ease-in-out; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* === Responsive === */
@media (max-width: 640px) {
  .nav-links { display: none; }
  .cards-grid { grid-template-columns: 1fr; }
  .terminal-section { grid-template-columns: 1fr; }
}
.terminal-section {
  display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; align-items: center;
}
@media (max-width: 900px) { .terminal-section { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="scanlines"></div>

<!-- Nav -->
<nav>
  <div class="nav-inner">
    <a href="/" class="nav-logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="square" stroke-linejoin="miter">
        <circle cx="12" cy="12" r="10"/>
        <circle cx="12" cy="12" r="6" stroke-opacity="0.5"/>
        <circle cx="12" cy="12" r="2" fill="currentColor"/>
        <path d="M12 12l8.5-8.5"/>
        <path d="M12 2v10H2" stroke-opacity="0.5" stroke-dasharray="2 2"/>
      </svg>
      <span>OPENRA<span class="rl">-RL</span></span>
    </a>
    <div class="nav-links">
      <a href="https://openra-rl.dev/docs/getting-started">DOCS</a>
      <a href="/docs">API</a>
      <a href="https://github.com/yxc20089/OpenRA-RL">GITHUB</a>
    </div>
  </div>
</nav>

<!-- Hero -->
<section class="hero crt-flicker fade-in">
  <div class="subtitle terminal-text">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 17l6-6-6-6"/><path d="M12 19h8"/></svg>
    SYSTEM OVERRIDE ACTIVE
  </div>
  <h1 class="alert-text">OPENRA-RL</h1>
  <div class="desc">
    OpenEnv environment for training AI agents to play
    <strong>Red Alert</strong> through the OpenRA engine.
    Connect via WebSocket or HTTP, send actions, observe the battlefield.
  </div>
  <div class="buttons">
    <a href="https://openra-rl.dev/docs/getting-started" class="btn-soviet">
      DOCUMENTATION
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
    </a>
    <a href="https://openra-rl-openra-bench.hf.space" class="btn-ghost">
      LEADERBOARD
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6-6 6 6"/><path d="M12 3v18"/><path d="M3 21h18"/></svg>
    </a>
  </div>
</section>

<!-- Endpoint cards -->
<div class="section">
  <h2>Endpoints</h2>
  <div class="cards-grid">
    <div class="card-military">
      <h3>API DOCS</h3>
      <p>Interactive Swagger UI with all REST and WebSocket endpoints.</p>
      <a href="/docs">/docs &rarr;</a>
    </div>
    <div class="card-military">
      <h3>HEALTH CHECK</h3>
      <p>Server status and readiness probe for monitoring.</p>
      <a href="/health">/health &rarr;</a>
    </div>
    <div class="card-military">
      <h3>ENV SCHEMA</h3>
      <p>JSON schemas for actions, observations, and game state.</p>
      <a href="/schema">/schema &rarr;</a>
    </div>
  </div>
</div>

<!-- Terminal / code section -->
<div class="section-dark">
  <div class="section">
    <div class="terminal-section">
      <div>
        <h2>Connect to Environment</h2>
        <p style="color:#9ca3af;line-height:1.7;margin-bottom:1.5rem;">
          Use the Python client to connect, reset the environment,
          and step through the game loop. Works with both local
          Docker and this HuggingFace-hosted server.
        </p>
        <a href="https://openra-rl.dev/docs/api-reference" class="btn-soviet" style="font-size:1.3rem;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5a2 2 0 012-2h13l3 3v13a2 2 0 01-2 2H6a2 2 0 01-2-2z"/><path d="M9 3v4h4"/></svg>
          API REFERENCE
        </a>
      </div>
      <div class="terminal">
        <div class="terminal-bar">
          <div class="terminal-dot" style="background:#ef4444;"></div>
          <div class="terminal-dot" style="background:#eab308;"></div>
          <div class="terminal-dot" style="background:#22c55e;"></div>
          <span>terminal</span>
        </div>
        <div class="terminal-body">
<pre><span class="t-prompt">$ </span><span class="t-cmd">pip install</span><span class="t-plain"> openra-rl</span>

<span class="t-kw">from</span><span class="t-plain"> openra_env.client </span><span class="t-kw">import</span><span class="t-plain"> OpenRAEnv</span>
<span class="t-kw">from</span><span class="t-plain"> openra_env.models </span><span class="t-kw">import</span><span class="t-plain"> OpenRAAction</span>

<span class="t-plain">url = </span><span class="t-str">"https://openra-rl-openra-rl.hf.space"</span>

<span class="t-kw">async with</span><span class="t-plain"> </span><span class="t-fn">OpenRAEnv</span><span class="t-plain">(url) </span><span class="t-kw">as</span><span class="t-plain"> env:</span>
<span class="t-plain">    obs = </span><span class="t-kw">await</span><span class="t-plain"> env.</span><span class="t-fn">reset</span><span class="t-plain">()</span>
<span class="t-plain">    </span><span class="t-kw">while not</span><span class="t-plain"> obs.done:</span>
<span class="t-plain">        action = your_agent(obs)</span>
<span class="t-plain">        obs = </span><span class="t-kw">await</span><span class="t-plain"> env.</span><span class="t-fn">step</span><span class="t-plain">(action)</span></pre>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Footer -->
<footer>
  <div class="footer-inner">
    <div class="footer-brand">
      <div class="logo">
        <svg viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-linecap="square" stroke-linejoin="miter" width="28" height="28">
          <circle cx="12" cy="12" r="10"/>
          <circle cx="12" cy="12" r="6" stroke-opacity="0.5"/>
          <circle cx="12" cy="12" r="2" fill="#dc2626"/>
          <path d="M12 12l8.5-8.5"/>
        </svg>
        OPENRA<span class="rl">-RL</span>
      </div>
      <div class="footer-status">
        &gt; SYSTEM STATUS: OPERATIONAL<br>
        &gt; MISSION: TRAIN AI TO CONQUER<br>
        &gt; &copy; 2025 OPENRA-RL CONTRIBUTORS.
      </div>
    </div>
    <div class="footer-col">
      <h3>Intel</h3>
      <ul>
        <li><a href="https://openra-rl.dev/docs/getting-started"><span class="chevron"></span>Getting Started</a></li>
        <li><a href="https://openra-rl.dev/docs/architecture"><span class="chevron"></span>Architecture</a></li>
        <li><a href="https://openra-rl.dev/docs/api-reference"><span class="chevron"></span>API Reference</a></li>
      </ul>
    </div>
    <div class="footer-col">
      <h3>Alliances</h3>
      <ul>
        <li><a href="https://www.openra.net/"><span class="chevron"></span>OpenRA Engine</a></li>
        <li><a href="https://huggingface.co/openra-rl"><span class="chevron"></span>HuggingFace</a></li>
        <li><a href="https://openra-rl-openra-bench.hf.space"><span class="chevron"></span>Leaderboard</a></li>
      </ul>
    </div>
  </div>
</footer>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page for the HuggingFace Space."""
    return LANDING_PAGE


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
