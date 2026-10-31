# Commander v3 — contrato del entorno

El jugador `rl-agent` (slot `Multi1`) es un **bot modular** de OpenRA: corre los
mismos módulos C# que el bot `normal` (`mods/ra/rules/ai.yaml`, condición
`enable-rl-agent`). El agente no hace clics: toma tres decisiones estratégicas y
los módulos las ejecutan.

Este contrato es parte del entorno. Si cambia, es una versión nueva: nunca a
mitad de un experimento.

## Cerebro

`CreateSessionRequest.agent_brain` / `MatchConfig.brain`:

| Cerebro | Quién decide | `step()` |
|---|---|---|
| `external` (default) | el agente (Python) | vuelve apenas hay decisiones nuevas (`pending`) |
| `builtin` | las heurísticas del bot `normal` | vuelve apenas deciden algo y lo reporta en `teacher` |

`builtin` sirve como test de paridad del ejecutor y como maestro para imitación
(Fase 2): sus decisiones vienen en el mismo formato que las del agente y siguen
el mismo camino. La elección nativa se responde como si viniera del agente: la
producción arranca al empezar el siguiente `step()`, así que la observación que
acompaña a cada decisión del maestro es exactamente la que habría visto un
agente `external` (la cola todavía libre, sin el ítem ya en producción). Un ítem
pedido que la cola no ofrece (p. ej. un MCV antes del taller) no arranca nada:
cuenta como "esperar".

## Decisiones

| Tipo | Cuándo se pide | Respuesta (`Decision`) |
|---|---|---|
| `DECIDE_BUILD` | la cola `Building` o `Defense` está libre (cadencia de `BaseBuilder`) | `choice` = un ítem de `options`, o `""` = esperar |
| `DECIDE_TRAIN` | una cola de unidades (`Infantry`, `Vehicle`, `Aircraft`, …) está libre (cada ~30 ticks por cola) | `choice` = un ítem de `options`, o `""` = esperar |
| `DECIDE_ARMY` | cada 250 ticks | `target_index` = `-1` hold (cancelar ataque, volver), `k` = atacar `targets[k]`; **sin respuesta = seguir igual** |

- El juego está pausado entre `step()`s: la respuesta llega en 0 ticks de juego
  y la producción arranca en ese mismo tick.
- Un pedido BUILD/TRAIN sin respuesta cuenta como "esperar"; la cola se vuelve a
  ofrecer a los 100 ticks.
- `options` = lo construible ahora en esa cola (sin vocabulario fijo en el motor).
- `targets` (≤ 12), todo conocimiento propio del agente:
  1. `enemy_spawn`: la posición inicial enemiga, mientras no esté explorada o haya
     edificios enemigos recordados a ≤ 10 celdas. En mapas de 2 spawns es
     información legítima; con más spawns habría que ofrecer todas.
  2. `enemy_building`: edificios enemigos vistos y no vistos destruidos (memoria de
     niebla), los más cercanos a la base propia primero (hasta 8 en total).
  3. Sectores de 16×16 (≤ 4, los más cercanos primero; la celda es una de tierra
     cerca del centro que un tanque puede alcanzar desde la base):
     `unexplored` = menos de la mitad explorada; `stale` = explorado pero sin verse
     hace más de 2.000 ticks. Sin esto, una expansión enemiga escondida dejaba
     partidas ganadas sin terminar.

## Ejecutor (reflejos fijos)

Lo hacen los módulos del bot `normal`, con estas diferencias en modo `external`:

| Módulo | Sigue haciendo | Deja de hacer (lo decide el agente) |
|---|---|---|
| `BaseBuilder` | colocar edificios, defensas hacia el enemigo, cadencia | elegir qué construir, filtro de dinero |
| `UnitBuilder` | cadencia por cola | elegir unidad, filtros de dinero/pausa, tope de unidades ociosas, pedidos de otros módulos (cosechadoras, MCV) |
| `SquadManager` | escuadrón de protección de la base, micro (`AttackOrFleeFuzzy`: reagrupar, attack-move, huir si pierde) | formar oleadas (`CreateAttackForce`, rush) |
| `Harvester`, `McvExpansion`, `BuildingRepair`, `PowerDown`, `SupportPower` | todo | — |

`attack(k)` junta todo el ejército terrestre que no está protegiendo la base y
crea un escuadrón contra el enemigo más cercano a `targets[k]` (radio 10 celdas);
sin enemigo ahí, el ejército hace attack-move a la celda. A partir de ahí el
escuadrón sigue con la lógica nativa: al perder el objetivo busca el enemigo más
cercano y puede retirarse solo si la pelea es desfavorable.

**Omnisciencia conocida:** el micro del `SquadManager` (de OpenRA) elige
objetivos sin mirar la niebla, igual que el bot rival. La decisión estratégica
del agente sí respeta la niebla.

## Resultado

`win` / `lose` / `draw` = la condición real del juego (sin adjudicación por
patrimonio). Partidas que llegan a `max_ticks` = `truncated` (no cuentan como
victoria). Un fallo dentro de `World.Tick` envenena la sesión: el cliente recibe
un error y la evaluación la cuenta como `error`, nunca como partida válida.
