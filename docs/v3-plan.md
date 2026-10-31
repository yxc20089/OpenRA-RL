# v3 — plan y criterios

v2 (AlphaLiteNet, currícula macro-first) quedó congelado en el tag `v2-final`.
Este documento es la fuente de verdad de v3.

## Por qué v2 no escalaba

1. **La política no era la que jugaba.** Una decisión cada 50–80 ticks sobre
   31 tipos × unidad × celda × ítem no alcanza para un RTS. El adapter reescribía
   la acción con heurísticas (remap, stage, guard, peel, pack-12, war_objective,
   auto_support) que cambiaban de run en run: crédito ruidoso y un MDP móvil.
2. **Muestras.** ~10² partidas/h y 4 episodios por iteración: `wr20` tenía ±0.2
   de ruido y guiaba promociones, restores y "sequías".
3. **Parches encadenados.** ~47 runs reanudados (Net2Net, cargas parciales,
   reward v3→v6) sin una línea base desde cero comparable.
4. **Maestro débil.** El BC imitaba a un script Python, no al bot C# del motor.
5. **La red no era el cuello.** Mejor resultado: 88% vs beginner, ~30% vs easy,
   0 vs medium.

## Principios

- La red decide **estrategia** (qué producir, a dónde va cada escuadrón, postura).
  Un ejecutor fijo, dentro del motor, hace los clics y el micro.
- El ejecutor y el contrato del MDP son **parte del entorno**: se versionan y no
  cambian durante un experimento.
- Todo cambio de régimen arranca **desde cero** y se compara con el mismo
  protocolo de evaluación (100 partidas, semillas fijas, intervalo de confianza).
- Juego en **lock-step headless**: el motor solo avanza dentro de `step()`.

## Alcance inicial

1v1, agente Aliados (`RandomAllies`) vs bot `Random`, un mapa.
"Experto" = ≥70% vs `hard` (bot OpenRA `normal`). Mapas extra y Soviet, después.

## Fases

| Fase | Qué | Pasa si |
|---|---|---|
| 0 | Limpieza, cliente gRPC lock-step, benchmark, determinismo | partidas/h medido; misma semilla → misma partida |
| 1 | Acciones macro + ejecutor C# + política a mano | el script gana a `hard` ≥60% |
| 2 | Grabar decisiones de los bots C# → BC | la red clonada ≥50% vs `easy` |
| 3 | PPO desde BC | ≥50% vs `hard` en eval de 100 partidas |
| 4 | Self-play / PFSP | ≥70% vs `hard`/`brutal`, sin colapsar a una estrategia |
| → | PR a `main` | Fase 3 estable |

## Estado

### Fase 0

- Repo reducido a: `OpenRA/` (submódulo), `proto/`, `ra_env/`, `tests/`, Docker.
- `ra_env`: cliente gRPC directo al daemon multi-sesión (sin FastAPI/OpenEnv/MCP).
- Imagen solo C# (`Dockerfile`), 2 daemons en `docker-compose.yaml`.
- Motor: `OPENRA_RL_DETERMINISTIC=1` siembra `World.LocalRandom` con la semilla
  de la partida (los módulos del bot lo usan; antes divergía a los ~300 ticks).
- Herramientas: `python -m ra_env.bench`, `python -m ra_env.determinism`.

Resultados medidos: ver [`v3-phase0-results.md`](v3-phase0-results.md).

### Fase 1 ✅

- Commander en el motor: el `rl-agent` corre los módulos del bot `normal`; el
  agente decide BUILD / TRAIN / ARMY. Contrato: [`v3-commander.md`](v3-commander.md).
- Paridad del ejecutor (`builtin` vs `hard`): 51%. Política a mano vs `hard`: 72%.
- `python -m ra_rl.eval` es el protocolo de evaluación oficial.

Resultados y bugs del motor corregidos: [`v3-phase1-results.md`](v3-phase1-results.md).

### Fase 2 ✅

- `builtin` responde por el mismo camino que un agente: cada decisión del maestro
  viene con el estado exacto en que se tomó.
- Features fijas (`ra_rl/features.py`), red MLP + puntero (`ra_rl/model.py`),
  grabación (`ra_rl.record`), BC (`ra_rl.bc`), DAgger con el `scripted` como
  etiquetador.
- BC(`builtin`) 100% vs `easy`; BC(`scripted`) + 2 rondas de DAgger (`bc_s2`)
  69% vs `hard` (66,5% en semillas nuevas, a la par del maestro).
- Regla nueva: se ajusta y se elige en semillas que no son las oficiales.

Resultados: [`v3-phase2-results.md`](v3-phase2-results.md).

### Fase 3 ✅

- PPO sobre la arquitectura Commander (`ra_rl/ppo.py`) inicializado desde el checkpoint `bc_s2`.
- `CommanderNet` extendido con cabeza de valor escalar $V(s)$.
- GAE por partida a nivel decisión ($\gamma = 0.997, \lambda = 0.95$).
- Rollout paralelo en 6 daemons headless, reciclado periódico de memoria y telemetría en TensorBoard.
- **Resultado récord (v1)**: **73.0% vs `hard`** (IC 95%: 63.6–80.7%, 73 victorias, 24 derrotas, 3 truncadas en 100 partidas de evaluación de selección) alcanzado en la **iteración 60** (`runs/phase3/ppo_v1/checkpoint_best.pt`). Supera tanto al maestro `scripted` (63.5%) como al baseline de imitación `bc_s2` (66.5%), pero presentaba colapsos tardíos de `argmax` al compartir tronco Actor–Crítico.
- Análisis detallado: [`v3-phase3-results.md`](v3-phase3-results.md).

### Fase 2.5 ✅

- Recuperación de features y maestro multi-arquetipo antes de la liga: `FEATURE_VERSION = 2` (`GLOBAL_DIM = 277`), topes de cordura en `item_mask` (`harv`, `mcv`, `tent`/`barr`, `pbox`/`hbox`), memoria persistente de amenaza/defensas y `ScriptedPolicy` reactiva contra `hard`, `brutal` (`rush`) y `turtle`.
- Grabación multi-bot (`1.800` partidas) + DAgger (`1.200` partidas) → `bc_s3` (`98,35%` acierto val. en producción, `92,35%` en ejército; **71,7%** vs `hard`, **83,3%** vs `brutal`, **86,7%** vs `turtle` en gates de 60 partidas, 0 errores).

### Fase 4 ✅

- PPO SMDP v2 (`ra_rl/ppo.py`) + Liga PFSP multi-bot (`ra_rl/league.py`) desde `bc_s3`:
  - **Crítico asimétrico desacoplado (CTDE)** $V(g, \text{priv})$ (`PRIV_DIM = 20`) + 6 iteraciones de warmup (`explained_var` sostenida en **+0.90 a +0.94**).
  - **Agrupamiento SMDP por tick** con descuento continuo $\gamma^{\Delta t / 100}$, **PBRS telescópico exacto** y **ancla KL enmascarada** ($\beta_{\text{KL}} = 0.05$).
  - **Curriculum PFSP** (`hard`, `brutal`, `turtle`) con normalización estratificada de ventajas por rival y reciclado determinista de daemons (**0 errores** en 5.340 partidas).
- **Resultado campeón (`runs/phase4/league_v2/checkpoint_best.pt`, iteración 20)**:
  - **Semillas oficiales reservadas `10000–10099` (100 partidas/bot)**: **80,0% vs `hard`** (IC 95%: 71,1–86,7%), **74,0% vs `brutal`** (IC 95%: 64,6–81,6%), **85,0% vs `turtle`** (IC 95%: 76,7–90,7%, solo 1 derrota en 100 partidas) — media armónica **0,794**.
  - **Semillas de selección `20000–20049` (50 partidas/bot)**: **84,0% vs `hard`**, **82,0% vs `brutal`**, **84,0% vs `turtle`** — media armónica **0,833**, manteniéndose entre **0,780 y 0,833** durante las 80 iteraciones sin un solo colapso.
- Resultados y diagnóstico comparativo `league_v1` vs `league_v2`: [`v3-phase4-results.md`](v3-phase4-results.md).
