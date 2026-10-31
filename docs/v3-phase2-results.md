# Fase 2 — resultados (imitación)

Medido el 2026-10-28. Mapa `singles.oramap`, agente Aliados (`RandomAllies`) vs
bot `Random`. Evaluación oficial: `python -m ra_rl.eval`, 100 partidas con
semillas 10000–10099 (nunca usadas para entrenar), `max_ticks` 54.000, IC de
Wilson al 95%. Las truncadas no cuentan como victoria.

## Criterios

| # | Criterio | Resultado | |
|---|---|---|---|
| 0 | Piso: política aleatoria (uniforme sobre las acciones válidas) | **0%** vs `easy` (22 derrotas, 78 truncadas) y **0%** vs `hard` (97 derrotas) | medido |
| 1 | BC del maestro `builtin` ≥ 50% vs `easy` | **100%** (IC 96–100%) | ✅ |
| 2 | BC del maestro `scripted` (+ DAgger) ≥ 60% vs `hard` | **71%** sin DAgger, **76%** / **69%** tras 1 / 2 rondas; en semillas nuevas el elegido (`bc_s2`) **66,5%** | ✅ |

## Todas las evaluaciones (100 partidas, semillas 10000–10099)

| Política | vs `easy` | vs `hard` | IC 95% (`hard`) |
|---|---|---|---|
| aleatoria | 0% | 0% | 0–4% |
| maestro `builtin` (bot `normal`) | 100% | 56% | 46–65% |
| BC(`builtin`) | 100% | 34% | 26–44% |
| maestro `scripted` (Fase 1) | — | 72% | 62–80% |
| BC(`scripted`) piloto, 593 partidas | — | 51% | 41–61% |
| BC(`scripted`) `bc_s0`, 1.800 partidas | — | 71% | 62–79% |
| + DAgger ronda 1 `bc_s1` | — | 76% | 67–83% |
| + DAgger ronda 2 `bc_s2` | 100% | 69% | 59–77% |

### Selección del modelo: semillas nuevas (20000–20199, 200 partidas vs `hard`)

Elegir el modelo por su número en las semillas oficiales sería sobreajustar la
evaluación, así que la selección se hizo en semillas que nadie había mirado:

| Política | vs `hard` | IC 95% | derrotas / truncadas |
|---|---|---|---|
| maestro `scripted` | 63,5% | 57–70% | 59 / 14 |
| `bc_s0` | 61% | 54–67,5% | 54 / 24 |
| `bc_s1` | 61% | 54–67,5% | 54 / 24 |
| **`bc_s2`** | **66,5%** | 60–73% | 44 / 23 |

Pareado contra el maestro en las mismas 200 semillas: `bc_s2` gana 30 partidas
que el maestro pierde y pierde 24 que el maestro gana (no significativo). El
modelo elegido para la Fase 3 es **`bc_s2`**: el mejor en la selección y el
entrenado con más datos (4.500 partidas).

**El 72% de la Fase 1 era optimista.** La política a mano se ajustó mirando las
semillas 10000–10099; en semillas nuevas gana 63,5%. Desde ahora ningún ajuste
se hace mirando las semillas oficiales: se ajusta y elige en otras, y las
oficiales solo se usan para reportar.

## Qué se entrenó

- **Entradas** (`ra_rl/features.py`, versión 1): vector global de 264 valores
  (economía, energía, conteos propios por tipo, colas y lo que producen, enemigos
  vistos con memoria propia, historial de órdenes de ejército) + hasta 12
  candidatos de ARMY con 43 valores cada uno. Nunca lee `global_summary.enemy`
  (test).
- **Red** (`ra_rl/model.py`, 210k parámetros): tronco MLP 2×256, cabeza de ítems
  condicionada por la cola y enmascarada por `options`, cabeza de ARMY tipo
  puntero + `hold` / `keep`.
- **Entrenamiento** (`ra_rl/bc.py`): entropía cruzada, AdamW + OneCycle, 20
  épocas, RTX 2070: 50–110 s por modelo. Validación por partida (semilla múltiplo
  de 20).

| Dataset | Partidas | Decisiones BUILD/TRAIN | Decisiones ARMY | Victorias del que jugó |
|---|---|---|---|---|
| `builtin` (rivales `easy`/`medium`/`hard`) | 1.800 | 504k | 196k | 67% |
| `scripted`, 15% acciones al azar (5% en ARMY) | 1.800 | 504k | 213k | 50% |
| DAgger 1 (juega `bc_s0`, etiqueta `scripted`, 5% / 2% azar) | 900 | 297k | 98k | 72% |
| DAgger 2 (juega `bc_s1`) | 900 | 284k | 97k | 68% |

| Modelo | Datos | Acierto val. BUILD/TRAIN | Acierto val. ARMY |
|---|---|---|---|
| BC(`builtin`) | `builtin` | 73,6% | 89,4% |
| `bc_s0` | `scripted` | 98,7% | 92,1% |
| `bc_s1` | + DAgger 1 | 99,2% | 92,1% |
| `bc_s2` | + DAgger 1 + 2 | 99,3% | 92,3% |

El maestro `builtin` elige unidades al azar ponderado: el 73,6% es casi el techo.
El `scripted` es determinista: 98,7%. En ARMY, la red reproduce el momento del
ataque el 82% de las veces y el objetivo exacto el 70%.

## Lecturas

1. **El ejecutor hace que la imitación funcione.** Con el commander, una red
   chica copia una política y juega lo mismo que ella (66,5% vs 63,5% del
   maestro en semillas nuevas). En v2 el
   BC imitaba clics y nunca pasó de ~30% vs `easy`.
2. **El maestro importa más que DAgger.** BC(`builtin`) pierde 22 puntos contra
   `hard` frente a su maestro (34% vs 56%); BC(`scripted`) no pierde nada. DAgger
   no movió el resultado de forma medible (61%, 61%, 66,5% en semillas nuevas,
   IC superpuestos): la copia ya era fiel.
3. **El ruido cuesta caro.** El maestro `scripted` con 15% de acciones al azar
   gana 50% contra el mix de rivales; la red, jugando sin ruido, 72%.

## Bugs del motor encontrados y corregidos (submódulo, rama `v3`)

| Bug | Efecto | Fix |
|---|---|---|
| Decisiones del maestro `builtin` desfasadas | El motor las tomaba a mitad del avance pero la observación llegaba al final (hasta 500 ticks después) y ya con el ítem en producción: la red habría aprendido a copiar la cola | `builtin` responde por el mismo camino que un agente (la producción arranca al empezar el siguiente `step()`) y el avance se corta en cada decisión |
| El worker atendía el avance siguiente | El RPC vuelve al completarse la observación; si el cliente respondía antes de que el worker saliera, el mismo worker seguía con el avance nuevo y la cola se llenaba (`RESOURCE_EXHAUSTED`, 2/100 partidas); al salir podía completar el avance nuevo con 0 ticks | cada `WorkItem` atiende solo su propio avance y nunca completa/falla otro |
| El avance se completaba antes de leer las entradas del tick | En el mismo tick el bridge completaba el avance y después leía el canal de acciones y corría el commander: si el cliente respondía muy rápido, sus decisiones entraban en ese tick o en el siguiente según el reloj | se completa al final del tick |
| Fuga de memoria | ~8–19 MB por partida (crece con los avances): a ~150 partidas un daemon ya no crea sesiones | sin fix de fondo; `ra_rl.record` reinicia cada daemon cada 60 partidas (`--recycle-every`) y reintenta una partida si la sesión no arranca |

Determinismo verificado con daemons limpios: misma semilla → misma partida en
12/12 corridas por cerebro (`builtin` y `external`), en 3 daemons distintos.

## Pendiente

- **La fuga de memoria** de los daemons (causa no encontrada).
- Lo de [Fase 1](v3-phase1-results.md) sigue igual: micro omnisciente del
  `SquadManager`, edificios sobre el agua, spawn enemiga para mapas de 2 spawns.

## Lo que sigue (Fase 3)

PPO desde `bc_s2`. El criterio del plan (≥ 50% vs `hard`) ya se cumple con BC,
así que hay que subirlo: superar al maestro de forma significativa (hoy ~64–66%
en semillas nuevas), camino al ≥ 70% vs `hard`/`brutal` de la Fase 4.
- Corridas: [`v3-phase2-evals.jsonl`](v3-phase2-evals.jsonl) (resumen de cada evaluación).
