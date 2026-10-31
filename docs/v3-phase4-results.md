# Fase 4 — Resultados (Recuperación Fase 2.5 + Liga PFSP CTDE v2)

Medido el 2026-10-31. Mapa `singles.oramap`, agente Aliados (`RandomAllies`) vs
liga multi-bot (`hard`, `brutal`, `turtle` — equivalentes en OpenRA a `normal`,
`rush` y `turtle`). Selección de modelo en semillas 20000–20049 (50 partidas por
bot) y evaluación oficial de reporte con `python -m ra_rl.eval` en las semillas
reservadas **10000–10099** (100 partidas por bot, nunca usadas en grabación,
DAgger, gates ni RL), `max_ticks` 54.000, IC de Wilson al 95%. Las partidas
truncadas no cuentan como victoria.

## Resumen de Hitos

| Criterio (`v3-plan.md`) | Resultado en selección (`20000–20049`, 50/bot) | Resultado oficial (`10000–10099`, 100/bot) | Estado |
|---|---:|---:|:---:|
| Win rate ≥ 70% vs `hard` | **84.0%** (IC 95%: 71.5–91.7%) | **80.0%** (IC 95%: 71.1–86.7%) | ✅ |
| Win rate ≥ 70% vs `brutal` (`rush`) | **82.0%** (IC 95%: 69.2–90.2%) | **74.0%** (IC 95%: 64.6–81.6%) | ✅ |
| Robustez anti-defensa vs `turtle` | **84.0%** (IC 95%: 71.5–91.7%) | **85.0%** (IC 95%: 76.7–90.7%) | ✅ |
| Media armónica 3 bots | **0.833** (83.3%) | **0.794** (79.4%) | ✅ |
| Estabilidad (sin colapso en 80 iteraciones) | Banda **0.780–0.833** en las 8 evals | **0 errores** en 5.340 partidas totales | ✅ |

El checkpoint campeón de v3 es **`runs/phase4/league_v2/checkpoint_best.pt`** (Iteración 20).
Resúmenes reproducibles de evaluación: [`v3-phase4-evals.jsonl`](v3-phase4-evals.jsonl).

---

## Por qué falló `league_v1` y qué corrigió `league_v2`

La primera corrida de liga (`runs/phase4/league_v1`, iniciada desde `ppo_v1`) se
degradó progresivamente de **68.2%** armónico (iteración 20, ya con 28 sesiones
caídas de 100) hasta **0.0%** vs `hard` en las iteraciones 60–80 y **30.0%** al
cierre (iteración 100). La auditoría identificó seis causas raíz acopladas:

| # | Causa raíz en `ppo_v1` / `league_v1` | Corrección en Fase 2.5 + `league_v2` |
|---|---|---|
| 1 | **Interferencia tronco Actor–Crítico**: `value_head` compartía `trunk` con `item_head` y `army_head`; al cambiar la distribución de rivales, los gradientes de $V(s)$ destruían las representaciones de producción. | **Crítico asimétrico desacoplado (CTDE)** en `ra_rl/model.py`: MLP independiente $V(g, \text{priv})$ con `PRIV_DIM = 20` (`global_summary` solo en entrenamiento) + `--critic-warmup-iters 6`. |
| 2 | **Bug SMDP intra-tick**: GAE aplicaba $\gamma = 0.997$ por cada *request* aunque ocurrieran 2–3 pedidos en el mismo tick ($\Delta t = 0$), pero trataba igual saltos de 250 ticks y penalizaba por decisión. | **Agrupamiento SMDP `StepNode` por tick** (`ra_rl/ppo.py`): descuento continuo exacto $\gamma^{\Delta t / 100}$ y **PBRS telescópico exacto** $\gamma(\Delta t)\Phi(s') - \Phi(s)$ (Ng et al., 1999) invariante a la política óptima. |
| 3 | **Sin topes de unidades en modo `external`**: al desactivar `UnitBuilder` nativo se perdían los `UnitLimits` del YAML; bajo exploración PPO la red spameaba cosechadoras/MCVs o bloqueaba `Building` en `WAIT`. | **Sanity caps en `Features.item_mask`** (`FEATURE_VERSION = 2`, `GLOBAL_DIM = 277`): topes dinámicos de `harv`, `mcv`, `tent`/`barr` y `pbox`/`hbox` + contadores anti-estancamiento y memoria persistente de amenaza/defensas. |
| 4 | **Deriva sin ancla respecto a imitación**: al bajar la entropía se disparaba *argmax collapse*; al subirla sin ancla se corrompía `DECIDE_BUILD`. | **Ancla KL enmascarada** ($\beta_{\text{KL}} = 0.05$) contra el actor congelado `bc_s3` sobre acciones válidas + piso de entropía calibrado (`0.003 → 0.001`). |
| 5 | **Varianza inter-rival en la liga**: normalizar ventajas mezclando partidas rápidas contra `brutal` con asedios largos contra `turtle` sesgaba el gradiente contra el rol más lento. | **Curriculum PFSP** ($p_i \propto (1 - \text{WR}_i)^{1.5}$, piso `0.18`) + **normalización estratificada de ventajas por tipo de bot** (`ra_rl/league.py`). |
| 6 | **Fuga de memoria en daemons durante evaluación**: `league_v1` reciclaba en rollout pero no antes/durante `eval`, contaminando métricas con sesiones `error`. | **Reciclado determinista** `--pre-recycle` y `--recycle-every 50` en `ra_rl/eval.py` (**0 errores** en las 5.040 partidas de `league_v2` y en las 300 de test oficial). |

---

## Fase 2.5 — Baseline Multi-Arquetipo (`bc_s3`)

Antes de lanzar RL se regeneró la base de imitación con `FEATURE_VERSION = 2`
(`GLOBAL_DIM = 277`) usando `ScriptedPolicy` reactiva (defensa anti-rush,
escalado de artillería anti-defensas estáticas y foco en estructuras productivas):

- **`data/scripted_v2`**: 1.800 partidas contra `hard`, `brutal`, `turtle` y `rush` → `bc_s0` (`val_prod_acc = 97.9%`, `val_army_acc = 91.8%`).
- **`data/dagger1_v2`**: 1.200 partidas on-policy etiquetadas por `scripted` → **`bc_s3`** (`val_prod_acc = 98.35%`, `val_army_acc = 92.35%`).
- **Gates de Fase 2.5** (60 partidas/bot, semillas `20000–20059`, 0 errores):
  - vs `hard`: **71.7%** (43W / 10L / 7T, IC 95%: 59.2–81.5%) — *74.0% en `20000–20049`*.
  - vs `brutal`: **83.3%** (50W / 10L / 0T, IC 95%: 72.0–90.7%) — *84.0% en `20000–20049`*.
  - vs `turtle`: **86.7%** (52W / 1L / 7T, IC 95%: 75.8–93.1%) — *86.0% en `20000–20049`*.
  - Media armónica inicial en `20000–20049`: **0.810**.

---

## Evaluación Oficial en Semillas Reservadas (`10000–10099`, 100 partidas/bot)

Comparación directa con las fases anteriores bajo el protocolo estándar del
proyecto (100 partidas por rival, semillas `10000–10099` intactas):

| Política | vs `hard` | IC 95% (`hard`) | vs `brutal` | IC 95% (`brutal`) | vs `turtle` | IC 95% (`turtle`) | Armónica |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `builtin` (Fase 2) | 56.0% | [46.2%, 65.3%] | — | — | — | — | — |
| `scripted` v1 (Fase 1) | 72.0% | [62.5%, 79.9%] | — | — | — | — | — |
| `bc_s2` (Fase 2) | 69.0% | [59.4%, 77.2%] | — | — | — | — | — |
| **`league_v2` (`checkpoint_best.pt`)** | **80.0%** (80W/12L/8T) | **[71.1%, 86.7%]** | **74.0%** (74W/25L/1T) | **[64.6%, 81.6%]** | **85.0%** (85W/1L/14T) | **[76.7%, 90.7%]** | **0.794** |

Incluso el **extremo inferior** del intervalo de confianza al 95% contra `hard`
(**71.1%**) supera el umbral de experto (**70%**) definido en [`v3-plan.md`](v3-plan.md),
y contra `turtle` la red sufre **una sola derrota en 100 partidas**.

---

## Evolución de Evaluaciones en Selección (`20000–20049`, 50 partidas/bot)

| Iteración | vs `hard` (W/L/T) | vs `brutal` (W/L/T) | vs `turtle` (W/L/T) | Media Armónica | Errores | Nota |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 (`bc_s3`) | 74.0% (37/7/6) | 84.0% (42/8/0) | 86.0% (43/1/6) | 0.810 | 0 | Baseline imitación v2 |
| 10 | 82.0% (41/6/3) | 82.0% (41/9/0) | 84.0% (42/2/6) | 0.827 | 0 | Post-warmup del crítico |
| **20** | **84.0% (42/7/1)** | **82.0% (41/9/0)** | **84.0% (42/1/7)** | **0.833** | **0** | 🏆 **`checkpoint_best.pt`** |
| 30 | 82.0% (41/5/4) | 76.0% (38/12/0) | 84.0% (42/1/7) | 0.805 | 0 | |
| 40 | 78.0% (39/8/3) | 76.0% (38/12/0) | 80.0% (40/2/8) | 0.780 | 0 | |
| 50 | **86.0% (43/6/1)** | 74.0% (37/12/1) | 84.0% (42/4/4) | 0.810 | 0 | Pico contra `hard` |
| 60 | 72.0% (36/8/6) | 80.0% (40/9/1) | **88.0% (44/1/5)** | 0.795 | 0 | PFSP corrige `brutal`; pico `turtle` |
| 70 | 72.0% (36/9/5) | **84.0% (42/7/1)** | 84.0% (42/1/7) | 0.796 | 0 | Pico contra `brutal` |
| 80 | 78.0% (39/7/4) | 80.0% (40/10/0) | 86.0% (43/1/6) | 0.812 | 0 | Cierre estable sin degradación |

---

## Telemetría de Rollout (`league_v2`, bloques de 10 iteraciones × 48 partidas)

Tiempo total de pared: **4.58 h** (16.482 s) para 3.840 partidas de rollout +
1.200 de evaluación (**0 partidas fallidas**).

| Bloque | WR Rollout | Ticks | `ExpVar` | `KLref` (P / A) | Entropía (P / A) | Mix PFSP (`H` / `B` / `T`) | `Veh` (`2tnk`/`arty`/`harv`) | `ArmyATK` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1–10 | 76.0% | 28.536 | +0.800 | 0.002 / 0.001 | 0.021 / 0.065 | 35.8 / 33.6 / 30.6% | 63.9 / 19.9 / 9.4% | 31.9% |
| **11–20** | **78.5%** | **27.715** | **+0.903** | **0.009 / 0.011** | **0.055 / 0.160** | **42.4 / 28.7 / 28.9%** | **64.1 / 20.1 / 9.3%** | **33.3%** |
| 21–30 | 76.9% | 27.920 | +0.915 | 0.017 / 0.020 | 0.057 / 0.154 | 40.2 / 30.1 / 29.7% | 63.7 / 20.6 / 9.4% | 32.2% |
| 31–40 | 76.5% | 27.905 | +0.916 | 0.017 / 0.020 | 0.055 / 0.160 | 40.8 / 32.0 / 27.2% | 63.3 / 20.5 / 9.6% | 32.6% |
| 41–50 | 78.3% | 28.258 | +0.907 | 0.021 / 0.026 | 0.059 / 0.162 | 39.3 / 33.7 / 27.0% | 63.4 / 21.0 / 8.9% | 33.0% |
| 51–60 | 76.7% | 27.887 | +0.912 | 0.018 / 0.027 | 0.059 / 0.156 | 40.1 / 31.5 / 28.4% | 63.9 / 20.3 / 9.5% | 32.2% |
| 61–70 | 79.8% | 27.886 | +0.905 | 0.017 / 0.038 | 0.054 / 0.156 | 37.4 / 30.8 / 31.8% | 63.2 / 20.4 / 9.5% | 32.8% |
| 71–80 | 74.8% | 28.062 | +0.909 | 0.019 / 0.035 | 0.055 / 0.150 | 41.5 / 34.4 / 24.1% | 62.9 / 20.5 / 9.8% | 32.9% |

---

## Conclusiones Técnicas

1. **Cirugía de precisión en lugar de destrucción de política**: Sobre las 68.474
   decisiones de validación offline, `checkpoint_best.pt` conserva **99.07%** de
   acuerdo `argmax` en `PROD` y **97.96%** en `ARMY` respecto a `bc_s3` (e incluso
   en la iteración 80 mantiene 98.45% / 95.78%). El ancla KL permitió refinar el
   1–2% crítico de estados sin desmantelar la macroeconomía.
2. **Qué aprendió exactamente RL sobre `bc_s3` (+10 pp vs `hard` en selección y caída de truncadas de 6/50 a 1/50)**:
   - **Selección de objetivos (`DECIDE_ARMY`)**: En los estados donde RL cambia la
     celda objetivo respecto a `bc_s3`, reduce el desvío hacia sectores
     `unexplored` (**31.0% → 20.7%**) y concentra el ejército en `enemy_spawn`
     (**14.9% → 21.8%**) y `enemy_building` (**54.0% → 57.5%**), cerrando
     partidas antes de que `hard` re-expanda.
   - **Composición vehicular (`Vehicle`)**: Desplazamiento neto **4.6× de `2tnk`
     hacia `arty`** (65 transiciones `2tnk → arty` frente a 14 inversas),
     rompiendo defensas estáticas con menor desgaste.
   - **Defensa y recuperación de infraestructura**: Reemplazo selectivo de `WAIT`
     por `gun` / `agun` en `Defense` y por `tent` / `apwr` / `weap` en `Building`
     cuando hay excedente de caja o pérdida de estructuras clave.
3. **Calidad del crítico CTDE**: La varianza explicada de $V(g, \text{priv})$ subió
   de `0.048` (iteración 1) a `0.921` al finalizar las 6 iteraciones de warmup y
   se mantuvo entre **+0.90 y +0.94** toda la corrida, eliminando por completo el
   desplome inicial de `ppo_v1` (donde el rollout caía a 6.2% en la iteración 8).
4. **Techo estructural del contrato Commander v3 (~80–84% armónico en `singles.oramap`)**:
   A partir de la iteración 20 la política oscila estacionariamente alrededor del
   óptimo sin degradarse (`0.780–0.833`). Las derrotas y truncadas remanentes
   provienen de límites del contrato C# actual ([`v3-commander.md`](v3-commander.md)):
   un único grupo terrestre global en `DECIDE_ARMY` (sin multi-squad para flanquear
   o explorar en paralelo), colocación geométrica fija en `BaseBuilder` cuando el
   perímetro se satura pasados los ~35k ticks, y un solo mapa/facción.
