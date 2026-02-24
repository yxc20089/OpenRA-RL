"""FastAPI application for the OpenRA-RL environment.

Creates the OpenEnv-compatible server using create_app().
"""

import asyncio
import json
import os
import time

from fastapi import Query
from fastapi.responses import HTMLResponse, StreamingResponse
from openenv.core.env_server import create_app

from openra_env.models import OpenRAAction, OpenRAObservation
from openra_env.server.openra_environment import OpenRAEnvironment

app = create_app(
    OpenRAEnvironment,
    OpenRAAction,
    OpenRAObservation,
    env_name="openra_env",
)


# ── Try Agent: LLM demo endpoint ────────────────────────────────────────────

_TRY_MAX_TURNS = 30
_TRY_MAX_TIME = 300  # 5 minutes

_COMMENTARY_SYSTEM_PROMPT = (
    "You are a real-time commentator for an AI playing Command & Conquer: Red Alert. "
    "Given the AI's recent actions and current game state, write 1-2 sentences "
    "explaining what the AI is doing and why, in an engaging style. "
    "Keep it concise and accessible to viewers who may not know RTS games well."
)
_COMMENTARY_MAX_TOKENS = 512


def _sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _generate_commentary(user_content: str, llm_config, broadcaster) -> None:
    """Generate commentary in the background and broadcast it."""
    import httpx as _httpx

    try:
        headers = dict(llm_config.extra_headers)
        if llm_config.api_key:
            headers["Authorization"] = f"Bearer {llm_config.api_key}"

        payload = {
            "model": llm_config.model,
            "messages": [
                {"role": "system", "content": _COMMENTARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": llm_config.max_tokens,
        }

        async with _httpx.AsyncClient() as client:
            resp = await client.post(
                llm_config.base_url,
                headers=headers,
                json=payload,
                timeout=llm_config.request_timeout_s,
            )

        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            # Reasoning models put thinking in 'reasoning', fall back to it
            if not text:
                reasoning = msg.get("reasoning") or ""
                if reasoning:
                    # Extract last sentence(s) as the summary
                    sentences = [s.strip() for s in reasoning.replace("\n", " ").split(".") if s.strip()]
                    text = ". ".join(sentences[-2:]) + "." if sentences else ""
            if text:
                broadcaster._broadcast(_sse("commentary", {"text": text.strip()}))
    except Exception:
        pass  # Commentary is non-essential


class TryGameBroadcaster:
    """Manages a single game broadcast to multiple SSE subscribers."""

    def __init__(self):
        self._event_history: list[str] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._game_running: bool = False
        self._game_task: asyncio.Task | None = None
        self._opponent: str = ""
        self._start_lock = asyncio.Lock()

    @property
    def game_running(self) -> bool:
        return self._game_running

    @property
    def has_replay(self) -> bool:
        return bool(self._event_history) and not self._game_running

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, event: str) -> None:
        self._event_history.append(event)
        for q in self._subscribers:
            q.put_nowait(event)

    async def replay_to(self, queue: asyncio.Queue) -> None:
        for event in list(self._event_history):
            await queue.put(event)

    async def start_game(self, opponent: str) -> None:
        async with self._start_lock:
            if self._game_running:
                return
            self._event_history.clear()
            self._opponent = opponent
            self._game_running = True
            self._game_task = asyncio.create_task(self._run_game(opponent))

    async def _run_game(self, opponent: str) -> None:
        try:
            async for event in _run_try_agent(opponent):
                self._broadcast(event)
        finally:
            self._game_running = False
            sentinel = _sse("_stream_end", {})
            for q in self._subscribers:
                q.put_nowait(sentinel)


_broadcaster = TryGameBroadcaster()


async def _run_try_agent(opponent: str):
    """Run LLM agent for one demo game, yielding SSE events."""
    from openra_env.agent import (
        SYSTEM_PROMPT,
        chat_completion,
        compose_pregame_briefing,
        compress_history,
        format_state_briefing,
        mcp_tools_to_openai,
    )
    from openra_env.config import LLMConfig
    from openra_env.mcp_ws_client import OpenRAMCPClient

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        yield _sse("error_event", {"message": "Server not configured for demo play (no API key)."})
        return

    llm_config = LLMConfig(
        api_key=api_key,
        model="stepfun/step-3.5-flash",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=1500,
        extra_headers={
            "HTTP-Referer": "https://openra-rl.dev",
            "X-Title": "OpenRA-RL Try Agent",
        },
    )
    commentary_config = LLMConfig(
        api_key=api_key,
        model="stepfun/step-3.5-flash",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=_COMMENTARY_MAX_TOKENS,
        request_timeout_s=15.0,
        extra_headers={
            "HTTP-Referer": "https://openra-rl.dev",
            "X-Title": "OpenRA-RL Commentary",
        },
    )

    # Configure opponent difficulty for the next game
    os.environ["BOT_TYPE"] = opponent.lower()

    yield _sse("status", {"message": f"Launching game vs {opponent} AI..."})

    try:
        async with OpenRAMCPClient(
            base_url="http://localhost:8000", message_timeout_s=300.0
        ) as env:
            yield _sse("status", {"message": "Resetting environment..."})
            await env.reset()

            # Discover tools
            mcp_tools = await env.list_tools()
            openai_tools = mcp_tools_to_openai(mcp_tools)

            # Start + end planning to trigger session start (unpauses game)
            yield _sse("status", {"message": "Starting game session..."})
            await env.call_tool("start_planning_phase")
            await env.call_tool("end_planning_phase", strategy="Demo game - aggressive rush")
            yield _sse("status", {"message": f"Game started. {len(mcp_tools)} tools available."})

            # Initialize conversation
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Get initial state and compose briefing
            state = await env.call_tool("get_game_state")
            briefing = compose_pregame_briefing(state)

            messages.append({
                "role": "user",
                "content": (
                    f"Game started!\n\n{briefing}\n\n"
                    f"## Current State\n```json\n{json.dumps(state, indent=2)}\n```\n\n"
                    f"ACT NOW! Deploy your MCV immediately, then start building power plant + barracks. "
                    f"Expand fast — every idle second costs you. Use plan() to chain: "
                    f"deploy MCV → build power plant → build barracks → build refinery. "
                    f"Then focus on economy (3+ refineries) and defense turrets toward the enemy."
                ),
            })

            yield _sse("game_state", {
                "tick": state.get("tick", 0),
                "units": state.get("own_units", 0),
                "buildings": state.get("own_buildings", 0),
                "cash": state.get("economy", {}).get("cash", 0),
            })

            total_tool_calls = 0
            total_api_calls = 0
            start_time = time.time()
            game_done = False
            consecutive_errors = 0

            for turn in range(1, _TRY_MAX_TURNS + 1):
                elapsed = time.time() - start_time
                if elapsed >= _TRY_MAX_TIME:
                    yield _sse("status", {"message": f"Time limit reached ({_TRY_MAX_TIME}s)."})
                    break

                # Compress history to stay within context limits
                messages = compress_history(messages, keep_last=40)

                # Inject state briefing (skip first turn — initial state already sent)
                if total_api_calls > 0:
                    try:
                        briefing_state = await env.call_tool("get_game_state")
                        brief = format_state_briefing(briefing_state)
                        if brief:
                            messages.append({"role": "user", "content": brief})
                        if isinstance(briefing_state, dict) and briefing_state.get("done"):
                            game_done = True
                            yield _sse("done", {
                                "result": briefing_state.get("result", "?"),
                                "tick": briefing_state.get("tick", 0),
                            })
                            break
                    except Exception:
                        pass

                # Call LLM
                try:
                    response = await chat_completion(messages, openai_tools, llm_config)
                except Exception as e:
                    yield _sse("error_event", {"message": f"LLM error: {e}"})
                    break

                total_api_calls += 1
                choice = response["choices"][0]
                assistant_msg = choice["message"]
                messages.append(assistant_msg)

                # Emit LLM reasoning
                if assistant_msg.get("content"):
                    yield _sse("llm", {"content": assistant_msg["content"][:500]})

                yield _sse("turn", {
                    "turn": turn,
                    "api_calls": total_api_calls,
                    "elapsed": round(elapsed),
                })

                # Handle tool calls
                tool_calls = assistant_msg.get("tool_calls", [])
                if not tool_calls:
                    messages.append({
                        "role": "user",
                        "content": "Please use the game tools to take action.",
                    })
                    continue

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"].get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {}

                    total_tool_calls += 1

                    args_str = json.dumps(fn_args)
                    if len(args_str) > 120:
                        args_str = args_str[:120] + "..."
                    yield _sse("tool_call", {"name": fn_name, "args": args_str})

                    try:
                        result = await env.call_tool(fn_name, **fn_args)
                        consecutive_errors = 0
                    except Exception as e:
                        result = {"error": str(e)}

                    # Detect game crash
                    if isinstance(result, dict) and "connection lost" in str(
                        result.get("error", "")
                    ).lower():
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            yield _sse("error_event", {"message": "Game connection lost."})
                            game_done = True

                    result_str = (
                        json.dumps(result) if not isinstance(result, str) else result
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

                    # Check game over
                    if isinstance(result, dict):
                        if result.get("done"):
                            game_done = True
                            yield _sse("done", {
                                "result": result.get("result", "?"),
                                "tick": result.get("tick", 0),
                            })
                        elif "tick" in result and "economy" in result:
                            yield _sse("game_state", {
                                "tick": result.get("tick", 0),
                                "units": result.get("own_units", 0),
                                "buildings": result.get("own_buildings", 0),
                                "cash": result.get("economy", {}).get("cash", 0),
                            })

                # Fire-and-forget async commentary (doesn't block game loop)
                if tool_calls and not game_done:
                    action_summaries = []
                    for tc in tool_calls:
                        fn = tc["function"]["name"]
                        try:
                            fa = json.loads(tc["function"].get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            fa = {}
                        action_summaries.append(f"{fn}({json.dumps(fa)})")

                    commentary_user = (
                        f"Turn {turn} actions:\n"
                        + "\n".join(f"- {a}" for a in action_summaries[:8])
                    )
                    asyncio.create_task(_generate_commentary(
                        commentary_user, commentary_config, _broadcaster,
                    ))

                if game_done:
                    break

                if choice.get("finish_reason") == "stop" and not tool_calls:
                    messages.append({
                        "role": "user",
                        "content": "Continue playing. Use game tools to check state and take actions.",
                    })

            # Surrender if game didn't end naturally
            if not game_done:
                try:
                    await env.call_tool("surrender")
                except Exception:
                    pass

            # Emit final scorecard
            try:
                final = await env.call_tool("get_game_state")
                mil = final.get("military", {})
                eco = final.get("economy", {})
                yield _sse("final", {
                    "result": final.get("result", "ongoing"),
                    "tick": final.get("tick", 0),
                    "turns": total_api_calls,
                    "tool_calls": total_tool_calls,
                    "elapsed": round(time.time() - start_time),
                    "kills_cost": mil.get("kills_cost", 0),
                    "deaths_cost": mil.get("deaths_cost", 0),
                    "units_killed": mil.get("units_killed", 0),
                    "units_lost": mil.get("units_lost", 0),
                    "cash": eco.get("cash", 0),
                    "units": final.get("own_units", 0),
                    "buildings": final.get("own_buildings", 0),
                })
            except Exception:
                pass

    except Exception as e:
        yield _sse("error_event", {"message": str(e)})


@app.get("/try-agent")
async def try_agent(
    opponent: str = Query("Normal", pattern="^(Easy|Normal|Hard)$"),
):
    """SSE stream of an LLM agent playing Red Alert.

    Multiple viewers can watch simultaneously. The first request starts
    a new game; subsequent requests join as spectators of the ongoing game.
    """
    queue = _broadcaster.subscribe()

    if _broadcaster.game_running:
        await queue.put(_sse("status", {"message": "Joining ongoing game as spectator..."}))
        await _broadcaster.replay_to(queue)
    elif _broadcaster.has_replay:
        await queue.put(_sse("status", {"message": "Replaying last game..."}))
        await _broadcaster.replay_to(queue)
        # Replay is finished — close stream so client re-enables the button
        await queue.put(_sse("_stream_end", {}))
    else:
        await _broadcaster.start_game(opponent)

    async def stream():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=360)
                if '"_stream_end"' in event:
                    break
                yield event
        except asyncio.TimeoutError:
            pass
        finally:
            _broadcaster.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
      <a href="/try" style="color:#ef4444;font-weight:700;">TRY</a>
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
    <a href="/try" class="btn-soviet">
      WATCH AI PLAY
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    </a>
    <a href="https://openra-rl.dev/docs/getting-started" class="btn-ghost">
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


# ── Try Page: Watch AI Play ──────────────────────────────────────────────────

TRY_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Try &mdash; Watch AI Play Red Alert</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Teko:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Share Tech Mono', monospace;
  background: radial-gradient(circle at center, #1a0505 0%, #050505 100%);
  color: #d1d5db;
  min-height: 100vh;
}
a { color: #d1d5db; text-decoration: none; transition: color .2s; }
a:hover { color: #fff; }
h1, h2, h3, .font-teko {
  font-family: 'Teko', sans-serif;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.scanlines {
  background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.2) 50%, rgba(0,0,0,0.2));
  background-size: 100% 4px;
  position: fixed; inset: 0; pointer-events: none; z-index: 50;
}
.terminal-text { color: #84cc16; text-shadow: 0 0 5px rgba(132,204,22,.5); }
.alert-text { color: #ef4444; text-shadow: 0 0 8px rgba(239,68,68,.8); }

nav {
  border-bottom: 2px solid #991b1b;
  background: rgba(0,0,0,.9);
  position: sticky; top: 0; z-index: 40;
  backdrop-filter: blur(4px);
}
.nav-inner {
  max-width: 72rem; margin: 0 auto; padding: 0 1.5rem;
  display: flex; align-items: center; justify-content: space-between; height: 4rem;
}
.nav-logo { display: flex; align-items: center; gap: .6rem; }
.nav-logo svg { width: 2rem; height: 2rem; color: #dc2626; animation: spin 4s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.nav-logo span { font-family: 'Teko', sans-serif; font-size: 1.8rem; font-weight: 700; color: #fff; letter-spacing: .15em; }
.nav-logo .rl { color: #dc2626; }
.nav-links { display: flex; gap: 1.5rem; align-items: center; }
.nav-links a { font-family: 'Teko', sans-serif; font-size: 1.15rem; letter-spacing: .1em; color: #9ca3af; text-transform: uppercase; }
.nav-links a:hover { color: #fff; }

.container { max-width: 56rem; margin: 0 auto; padding: 2rem 1.5rem; }
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-size: 2.5rem; color: #fff; margin-bottom: .5rem; }
.header p { color: #9ca3af; font-size: .95rem; }

.controls {
  display: flex; gap: 1rem; align-items: center; justify-content: center;
  margin-bottom: 1.5rem; flex-wrap: wrap;
}
.controls select {
  font-family: 'Share Tech Mono', monospace;
  background: #121212; color: #d1d5db; border: 2px solid #525252;
  padding: .5rem 1rem; font-size: 1rem; cursor: pointer;
}
.controls select:hover { border-color: #737373; }
.btn-soviet {
  display: inline-flex; align-items: center; gap: .5rem;
  background: #dc2626; border: 2px solid #f87171;
  box-shadow: 4px 4px 0 #000; transition: all .1s;
  color: #fff; font-family: 'Teko', sans-serif;
  font-size: 1.6rem; padding: .4rem 1.8rem;
  text-transform: uppercase; cursor: pointer;
}
.btn-soviet:hover { transform: translate(2px,2px); box-shadow: 2px 2px 0 #000; background: #ef4444; }
.btn-soviet:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: 4px 4px 0 #000; }

.game-log {
  background: #0a0a0a; border: 2px solid #262626; border-left: 4px solid #dc2626;
  padding: 1rem 1.2rem; font-size: .82rem; line-height: 1.7;
  height: 420px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
  margin-bottom: 1.5rem;
}
.game-log .log-status { color: #84cc16; }
.game-log .log-turn { color: #facc15; }
.game-log .log-llm { color: #c084fc; }
.game-log .log-tool { color: #38bdf8; }
.game-log .log-state { color: #6b7280; }
.game-log .log-done { color: #ef4444; font-weight: bold; }
.game-log .log-error { color: #ef4444; }
.game-log .log-commentary { color: #f59e0b; font-style: italic; padding-left: 1em; border-left: 2px solid #f59e0b; margin: 2px 0; }

.scorecard {
  background: #121212; border: 2px solid #262626; padding: 1.5rem;
  display: none;
}
.scorecard h2 { color: #fff; font-size: 1.8rem; margin-bottom: 1rem; }
.scorecard table { width: 100%; border-collapse: collapse; }
.scorecard td { padding: .4rem .8rem; border-bottom: 1px solid #1a1a1a; font-size: .85rem; }
.scorecard td:first-child { color: #9ca3af; }
.scorecard td:last-child { color: #fff; text-align: right; }
.scorecard .result-win { color: #22c55e; font-size: 1.2rem; font-weight: bold; }
.scorecard .result-loss { color: #ef4444; font-size: 1.2rem; font-weight: bold; }

footer {
  background: #000; border-top: 2px solid #7f1d1d;
  margin-top: 3rem; padding: 1.5rem;
  text-align: center; font-size: .8rem; color: #6b7280;
}
</style>
</head>
<body>
<div class="scanlines"></div>

<nav>
  <div class="nav-inner">
    <a href="/" class="nav-logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="square" stroke-linejoin="miter">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6" stroke-opacity="0.5"/>
        <circle cx="12" cy="12" r="2" fill="currentColor"/><path d="M12 12l8.5-8.5"/>
        <path d="M12 2v10H2" stroke-opacity="0.5" stroke-dasharray="2 2"/>
      </svg>
      <span>OPENRA<span class="rl">-RL</span></span>
    </a>
    <div class="nav-links">
      <a href="/try" style="color:#ef4444;font-weight:700;">TRY</a>
      <a href="https://openra-rl.dev/docs/getting-started">DOCS</a>
      <a href="/docs">API</a>
      <a href="https://github.com/yxc20089/OpenRA-RL">GITHUB</a>
    </div>
  </div>
</nav>

<div class="container">
  <div class="header">
    <h1 class="alert-text">Watch AI Play</h1>
    <p>A pre-configured LLM agent plays Red Alert against the built-in AI. No setup needed.</p>
  </div>

  <div class="controls">
    <select id="opponent">
      <option value="Easy">Easy</option>
      <option value="Normal" selected>Normal</option>
      <option value="Hard">Hard</option>
    </select>
    <button id="playBtn" class="btn-soviet" onclick="startGame()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      WATCH AI PLAY
    </button>
  </div>

  <div id="gameLog" class="game-log">Waiting to start...\n</div>

  <div id="scorecard" class="scorecard">
    <h2>Scorecard</h2>
    <table id="scorecardTable"></table>
  </div>
</div>

<footer>&copy; 2025 OpenRA-RL Contributors &mdash; <a href="/">Home</a></footer>

<script>
let eventSource = null;

function log(msg, cls) {
  const el = document.getElementById('gameLog');
  const span = document.createElement('span');
  span.className = cls || '';
  span.textContent = msg + '\\n';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

function startGame() {
  const btn = document.getElementById('playBtn');
  const logEl = document.getElementById('gameLog');
  const scorecard = document.getElementById('scorecard');
  const opponent = document.getElementById('opponent').value;

  // Close previous connection if any
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Reset UI
  logEl.innerHTML = '';
  scorecard.style.display = 'none';
  btn.disabled = true;
  document.getElementById('opponent').disabled = true;
  btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> WATCHING...';

  log('Connecting to game server...', 'log-status');

  // Use EventSource for reliable SSE streaming through proxies
  eventSource = new EventSource('/try-agent?opponent=' + encodeURIComponent(opponent));

  eventSource.addEventListener('status', function(e) {
    try { handleEvent('status', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('turn', function(e) {
    try { handleEvent('turn', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('llm', function(e) {
    try { handleEvent('llm', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('tool_call', function(e) {
    try { handleEvent('tool_call', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('game_state', function(e) {
    try { handleEvent('game_state', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('done', function(e) {
    try { handleEvent('done', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('final', function(e) {
    try { handleEvent('final', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('commentary', function(e) {
    try { handleEvent('commentary', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('error_event', function(e) {
    try { handleEvent('error', JSON.parse(e.data)); } catch(ex) {}
  });
  eventSource.addEventListener('_stream_end', function(e) {
    eventSource.close();
    eventSource = null;
    resetBtn();
  });

  eventSource.onerror = function() {
    eventSource.close();
    eventSource = null;
    resetBtn();
  };
}

function handleEvent(type, data) {
  switch(type) {
    case 'status':
      log(data.message, 'log-status');
      break;
    case 'turn':
      log('[Turn ' + data.turn + '] API calls: ' + data.api_calls + ' | ' + data.elapsed + 's', 'log-turn');
      break;
    case 'llm':
      if (data.content) {
        const text = data.content.length > 300 ? data.content.slice(0, 300) + '...' : data.content;
        log('  AI: ' + text, 'log-llm');
      }
      break;
    case 'tool_call':
      log('  >> ' + data.name + '(' + (data.args || '') + ')', 'log-tool');
      break;
    case 'game_state':
      log('  tick=' + data.tick + ' units=' + data.units + ' buildings=' + data.buildings + ' $' + data.cash, 'log-state');
      break;
    case 'done':
      log('\\nGAME OVER: ' + (data.result || '?').toUpperCase() + ' (tick ' + data.tick + ')', 'log-done');
      break;
    case 'final':
      showScorecard(data);
      break;
    case 'commentary':
      if (data.text) {
        log('  [COMMENTARY] ' + data.text, 'log-commentary');
      }
      break;
    case 'error_event':
      log('Error: ' + (data.message || 'Unknown'), 'log-error');
      break;
  }
}

function showScorecard(data) {
  const sc = document.getElementById('scorecard');
  const tbl = document.getElementById('scorecardTable');
  const result = (data.result || 'ongoing').toUpperCase();
  const cls = result === 'WIN' ? 'result-win' : 'result-loss';

  tbl.innerHTML =
    '<tr><td>Result</td><td class="' + cls + '">' + result + '</td></tr>' +
    '<tr><td>Game Ticks</td><td>' + data.tick + '</td></tr>' +
    '<tr><td>LLM Turns</td><td>' + data.turns + '</td></tr>' +
    '<tr><td>Tool Calls</td><td>' + data.tool_calls + '</td></tr>' +
    '<tr><td>Duration</td><td>' + data.elapsed + 's</td></tr>' +
    '<tr><td>Units Killed</td><td>' + data.units_killed + '</td></tr>' +
    '<tr><td>Units Lost</td><td>' + data.units_lost + '</td></tr>' +
    '<tr><td>Kill Value</td><td>$' + data.kills_cost + '</td></tr>' +
    '<tr><td>Death Value</td><td>$' + data.deaths_cost + '</td></tr>' +
    '<tr><td>Cash Remaining</td><td>$' + data.cash + '</td></tr>' +
    '<tr><td>Own Units</td><td>' + data.units + '</td></tr>' +
    '<tr><td>Own Buildings</td><td>' + data.buildings + '</td></tr>';

  sc.style.display = 'block';
}

function resetBtn() {
  const btn = document.getElementById('playBtn');
  btn.disabled = false;
  btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> WATCH AI PLAY';
  document.getElementById('opponent').disabled = false;
}

// Auto-connect if a game is already running
fetch('/try-status')
  .then(r => r.json())
  .then(status => {
    if (status.game_running || status.has_replay) {
      if (status.opponent) {
        document.getElementById('opponent').value = status.opponent;
      }
      startGame();
    }
  })
  .catch(() => {});
</script>
</body>
</html>"""


@app.get("/try-status")
async def try_status():
    """Check if a game is currently running or has a replay available."""
    return {
        "game_running": _broadcaster.game_running,
        "has_replay": _broadcaster.has_replay,
        "opponent": _broadcaster._opponent if (_broadcaster.game_running or _broadcaster.has_replay) else "",
    }


@app.get("/try", response_class=HTMLResponse)
async def try_page():
    """Interactive page to watch an LLM agent play Red Alert."""
    return TRY_PAGE


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
