# Fase 0 — resultados

Medido el 2026-10-28 en la máquina de desarrollo (Ryzen 5600X 6c/12t, Docker
Desktop/WSL2), imagen `openra-rl:v3`, mapa `singles.oramap`, bot `hard`
(OpenRA `normal`), política sonda (despliega el MCV y no hace nada más).
Datos crudos: [`v3-phase0-bench.jsonl`](v3-phase0-bench.jsonl).

## Throughput

`games/h @20k` = ticks/s × 3600 / 20000 (partida de referencia de 20k ticks).

| Daemons × sesiones | Decisión cada | ticks/s | partidas/h @20k |
|---|---|---|---|
| 1 × 1 | 25 ticks | 4.595 | 827 |
| 1 × 1 | 100 ticks | 6.236 | 1.123 |
| 1 × 1 | 250 ticks | 6.628 | 1.193 |
| 6 × 1 | 100 ticks | 19.361 | 3.485 |
| 6 × 1 | 250 ticks | 26.425 | 4.757 |

- v2 rendía ~10² partidas/h. Lock-step directo por gRPC, sin FastAPI/WS/MCP,
  da **30–45×** con 6 daemons.
- Con decisiones cada ≥100 ticks el costo de ida y vuelta es chico: la
  simulación pura de un daemon ronda los ~7.000 ticks/s.
- Cada daemon ocupa ~350–400 MB de RAM.
- La sonda pierde contra `hard` en ~14k ticks, así que una partida real va a
  durar más (más ticks, menos partidas/h). Los ticks/s sí son comparables.

## Determinismo

| Caso | Resultado |
|---|---|
| Motor original, misma semilla | diverge a los ~300 ticks (`World.LocalRandom` sin semilla) |
| `OPENRA_RL_DETERMINISTIC=1`, partidas en secuencia | 3 de 4 semillas idénticas hasta el final (12–15k ticks); la 4ª idéntica hasta 3 decisiones antes del final, donde un avance termina 1 tick antes (mismo tick final y resultado) |
| Varias partidas a la vez en el mismo daemon | diverge a los ~1.500 ticks, con 1 o 2 workers |

Conclusiones:

- **Topología: 1 sesión por daemon, `OPENRA_RL_WORKERS=1`**, escalar con
  daemons (`docker-compose.yaml` levanta 6). Es determinista y además es lo
  que más rinde por núcleo.
- Las sesiones concurrentes de un mismo proceso **no son independientes**:
  comparten estado del motor (con 2 partidas idénticas en paralelo no hay
  diferencia; con una tercera distinta, sí). Además el daemon tickeaba con
  `ProcessorCount` hilos, no con uno solo como asumía v2.

## Cambios en el motor (submódulo `OpenRA`, rama `v3`)

| Cambio | Por qué |
|---|---|
| `OPENRA_RL_DETERMINISTIC=1` siembra `World.LocalRandom` con la semilla del lobby | Los módulos del bot usan `LocalRandom` |
| `OPENRA_RL_WORKERS` (default `ProcessorCount`) | 1 worker = sin carreras entre hilos |
| Regla `win_early` de v2 apagada salvo `OPENRA_RL_EARLY_WIN=1` | Declaraba victoria por "patrimonio <10%" sostenido 500 ticks: un MDP oculto en el motor. En v3, ganar es la condición real del juego; si hace falta adjudicar, se hace explícito en el cliente |

## Pendiente (no bloquea la Fase 1)

- Encontrar el estado compartido entre sesiones (candidatos: usos de
  `Game.RunAfterTick` en lógica de juego como `Reservable`/`ProximityCapturableBase`,
  cachés en `TraitInfo` compartidos vía `ModData`). Solo importa si alguna vez
  queremos varias sesiones por proceso.
- El ±1 tick al final de algunas partidas (`TickSession`/fin de avance).
- El código de "sidecar PPO" de v2 dentro de `ExternalBotBridge`
  (`OPENRA_RL_AUTOSTART`) sigue en el motor; la imagen lo desactiva. Se borra
  cuando la Fase 1 toque ese archivo.
