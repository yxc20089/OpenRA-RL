# Fase 1 — resultados

Medido el 2026-10-28. Mapa `singles.oramap`, rival `hard` (bot OpenRA `normal`,
facción `Random`), agente Aliados (`RandomAllies`). 6 daemons deterministas, una
partida por daemon, semillas 10000–10099, `max_ticks` 54.000 (~36 min de juego).
Comando: `python -m ra_rl.eval --policy <builtin|scripted> --bot hard --games 100`.

> **Corrección (Fase 2):** la política a mano se ajustó mirando estas mismas
> semillas. En semillas nuevas (20000–20199, 200 partidas) gana **63,5%**
> (IC 57–70%), no 72%. Sigue pasando el criterio de ≥ 60%, por poco. Ver
> [`v3-phase2-results.md`](v3-phase2-results.md).
Contrato del entorno: [`v3-commander.md`](v3-commander.md).

## Criterios

| Criterio | Resultado | |
|---|---|---|
| Paridad del ejecutor: cerebro `builtin` (heurísticas del `normal`) vs `hard` entre 40 y 60% | **51%** (IC 95% 41–61%) · 35 derrotas · 14 truncadas | ✅ |
| Política a mano vs `hard` ≥ 60% | **72%** (IC 95% 62,5–80%) · 23 derrotas · 5 truncadas | ✅ |
| Determinismo | misma semilla → misma partida, con los dos cerebros (la paridad repitió 51/35/14 exacto en dos corridas) | ✅ |
| Throughput con partidas reales | 100 partidas en ~5 min ≈ **1.250 partidas/h** (partida media ~30–34k ticks) | ✅ |

Decisiones por partida: ~450 (política a mano, `external`), ~490 reportadas por el
maestro (`builtin`). Es el horizonte que verá la red en las fases 2–3.

## Cómo llegó la política a mano al 72%

| Versión | Cambio | vs `hard` |
|---|---|---|
| v1 | economía primero, tope de infantería, ataque a 9k | 1/12 |
| v2 | producción continua, ataque masivo a 16k | 2/24 |
| v2 + motor | la producción arranca al responder (antes esperaba al siguiente sondeo del módulo) | 2/24 |
| v3 | infantería sin freno 2:1, 2ª refinería antes de la fábrica, ataque a 6–8k | ~55–63% (30 partidas) |
| v3 + objetivos | `unexplored` / `stale` alcanzables, la spawn se descarta al limpiarla, abandona objetivos trabados | **72%** (100 partidas) |

El salto final vino de **terminar partidas ganadas**: las truncadas bajaron de
~20% a 5%. El ejército no tenía a dónde ir cuando el rival se expandía en zonas
no vistas.

## Bugs del motor encontrados y corregidos (submódulo, rama `v3`)

| Bug | Efecto | Fix |
|---|---|---|
| `ScreenMap.Add` sin `WorldRenderer` en headless | Al construir el Centro Técnico Aliado, el GPS lanzaba `NullReferenceException` en mitad de `World.Tick`; el pipeline de órdenes quedaba desfasado y **ninguna orden se procesaba el resto de la partida**. Pasaba en la mayoría de las partidas largas (también en v2) | guard en `ScreenMap.Add` |
| Carrera en `pendingAdvanceResult` | Se limpiaba después de completar la observación; si el cliente respondía en ese instante, su siguiente avance quedaba huérfano y la sesión se envenenaba a los 90 s (~1/100 partidas) | limpiar antes de completar, en los 4 puntos |
| Excepciones en `World.Tick` se "reparaban" | Partida corrupta que seguía jugándose en silencio | ahora envenenan la sesión: la evaluación la cuenta como `error` |

## Pendiente / límites conocidos

- **Memoria de los daemons:** crece ~2 MB por partida (350 MB → ~850 MB tras 200).
  Con 2 GB de límite alcanza para ~800 partidas por daemon; la Fase 2 necesita
  reiniciar daemons periódicamente o encontrar la fuga.
- **Micro omnisciente:** el `SquadManager` elige blancos sin niebla (igual que el
  bot rival). La decisión estratégica del agente sí respeta la niebla.
- **Edificios sobre el agua:** un astillero rival es inalcanzable para un ejército
  terrestre; explica parte de las truncadas que quedan.
- **El `AttackOrFleeFuzzy` puede retirar un ataque** que el agente ordenó (reflejo
  del ejecutor, igual que en el bot nativo).
- La spawn enemiga como objetivo asume mapas de 2 spawns.

## Lo que sigue (Fase 2)

El maestro está listo: `builtin` reporta ~490 decisiones por partida en el mismo
formato que las del agente, a ~1.250 partidas/h. Fase 2 = grabar esas decisiones
y entrenar la red por imitación (BC), con el criterio ≥50% vs `easy`.
