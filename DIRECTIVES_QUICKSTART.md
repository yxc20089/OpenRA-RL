# Strategic Directives - Quick Start Guide

## TL;DR - User-Friendly Commands

```bash
# Use built-in strategies (easiest!)
openra-rl play --strategy=rush
openra-rl play --strategy=turtle
openra-rl play --strategy=balanced

# Add custom directives on the fly
openra-rl play --directive "Maintain 3 harvesters" --directive "Build Tesla Coils"

# Combine strategy with custom directives
openra-rl play --strategy=rush --directive "Attack at tick 1500"

# Use your own strategy file
openra-rl play --strategy=my-custom-strategy.yaml
```

---

## Built-in Strategies

### Rush Strategy
```bash
openra-rl play --strategy=rush
```

**What it does:**
- Builds minimal economy (1 harvester)
- Rushes barracks and military units
- Attacks early with 4-5 units
- No advanced tech until after first attack

**Best against:** Greedy/defensive opponents

---

### Turtle Strategy
```bash
openra-rl play --strategy=turtle
```

**What it does:**
- Strong economy (2-3 harvesters)
- Builds defensive structures
- Delays attack until War Factory + 8 units
- Scouts before major assault

**Best against:** Aggressive rushers

---

### Balanced Strategy
```bash
openra-rl play --strategy=balanced
```

**What it does:**
- Moderate economy (2 harvesters)
- Light defenses
- Scouts early and adapts
- Balanced spending on economy/military

**Best against:** Unknown opponents

---

## Custom Directives

### Add Individual Orders

```bash
# Single directive
openra-rl play --directive "Maintain 2 harvesters"

# Multiple directives
openra-rl play \
  --directive "Maintain 2 harvesters" \
  --directive "Build defenses near refinery" \
  --directive "Attack when you have 10 units"
```

### Combine with Built-in Strategy

```bash
# Start with rush, but add custom orders
openra-rl play --strategy=rush \
  --directive "Scout enemy base at tick 1000" \
  --directive "Build War Factory after first attack"
```

---

## Custom Strategy Files

### Create Your Own Strategy

Create `my-strategy.yaml`:
```yaml
pregame_strategy: "Aggressive expansion with heavy tanks"

standing_orders:
  - "Maintain 3 harvesters"
  - "Build 2 War Factories"
  - "Prioritize heavy tanks over light vehicles"
  - "Expand to secondary ore field when cash > $3000"

midgame_adjustments:
  - "If enemy has air units, build AA defenses"
  - "If winning, maintain pressure with continuous attacks"
```

Use it:
```bash
openra-rl play --strategy=my-strategy.yaml
```

---

## Full Example Session

### Example 1: Quick Rush Game

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Play with rush strategy
openra-rl play --strategy=rush --verbose

# Expected output:
# Strategy: rush
# Starting LLM agent...
# Model: anthropic/claude-sonnet-4-20250514 via openrouter
# Strategic directives enabled: 7 directive(s)
# ...
```

### Example 2: Custom Directives

```bash
openra-rl play \
  --strategy=balanced \
  --directive "Build Tesla Coils for defense" \
  --directive "Train 2 mammoth tanks before attacking" \
  --verbose
```

### Example 3: Local Model (Ollama)

```bash
# No API key needed!
openra-rl play \
  --provider ollama \
  --model qwen3:32b \
  --strategy=turtle \
  --verbose
```

---

## Verifying Directives Are Active

When you run with `--verbose`, look for these lines:

```
Strategy: rush  ← Confirms strategy loaded
Starting LLM agent...
Strategic directives enabled: 7 directive(s)  ← Directives loaded
Discovered 51 tools (48 MCP + 3 client-side)  ← Directive tools available
```

During gameplay, the agent will explicitly reference directives:

```
[LLM thinks] Following the rush directive: building barracks immediately...
[Tool] build_and_place({"building_type": "barr"})

[LLM thinks] Directive says maintain 1 harvester, currently have 1, focusing on military...
[Tool] build_unit({"unit_type": "e1", "count": 5})
```

---

## Advanced Usage

### Use with Specific Model

```bash
openra-rl play \
  --strategy=rush \
  --model anthropic/claude-sonnet-4-20250514 \
  --api-key sk-or-...
```

### Change Difficulty

```bash
openra-rl play --strategy=rush --difficulty=hard
```

### Save Replay

```bash
# Replays are automatically saved to ~/.openra-rl/replays/
openra-rl play --strategy=rush

# After game, watch replay:
openra-rl replay watch
```

---

## Troubleshooting

**Problem:** "Strategy: rush" doesn't appear in output

**Solution:** Make sure you're using the latest version:
```bash
cd OpenRA-RL
git pull
pip install -e .
```

---

**Problem:** Agent ignores directives

**Possible causes:**
1. Model doesn't support tool calling well (try Claude Sonnet)
2. Directives are too vague (be more specific)
3. Conflicting directives

**Solution:**
```bash
# Use Claude Sonnet (best tool-calling support)
openra-rl play \
  --strategy=rush \
  --model anthropic/claude-sonnet-4-20250514 \
  --verbose
```

---

## Comparison: Old vs New Interface

### Before (complex):
```bash
# Create config file
cat > my-config.yaml << EOF
directives:
  enabled: true
  directives_file: "examples/directives-rush.yaml"
llm:
  model: "anthropic/claude-sonnet-4-20250514"
EOF

# Run with config
python examples/llm_agent.py --config my-config.yaml --verbose
```

### After (simple):
```bash
openra-rl play --strategy=rush --verbose
```

**90% fewer steps!** 🎉

---

## Quick Reference

```bash
# Built-in strategies
openra-rl play --strategy=rush      # Fast attack
openra-rl play --strategy=turtle    # Defensive buildup
openra-rl play --strategy=balanced  # Adaptive play

# Custom directives
openra-rl play --directive "Your order here"

# Combine both
openra-rl play --strategy=rush --directive "Custom order"

# Custom file
openra-rl play --strategy=path/to/your-strategy.yaml

# With local model (no API key)
openra-rl play --provider ollama --model qwen3:32b --strategy=rush

# Full help
openra-rl play --help
```

---

## Next Steps

1. **Try a quick game**: `openra-rl play --strategy=rush --verbose`
2. **Watch the replay**: `openra-rl replay watch`
3. **Create custom strategy**: Copy `examples/directives-rush.yaml` and modify
4. **Share strategies**: Submit your best strategies to the community!

Happy commanding! 🎮
