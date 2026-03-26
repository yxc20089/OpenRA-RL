"""Debug visualisation for EnemySpatialMemory.

Renders ``memory_presence`` and ``memory_decay`` heatmaps side-by-side and
optionally overlays the current visible-enemy positions.  Intended for
qualitative evaluation of agent reasoning during development.

Quick-start::

    from openra_env.enemy_memory import EnemySpatialMemory
    from openra_env.debug_viz import render_enemy_memory_debug

    mem = EnemySpatialMemory(64, 64, max_age=200, decay_lambda=0.02)
    # ... run episode steps ...
    fig = render_enemy_memory_debug(
        mem,
        visible_enemies=[(10, 20), (15, 30)],
        step=42,
        save_path="frame_042.png",
        show=False,
    )

Frame files can be turned into a video with::

    ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 memory_debug.mp4

Requires ``matplotlib``.  Install with ``pip install matplotlib``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Re-use the Cell type alias from enemy_memory
Cell = Tuple[int, int]   # (row, col)


# ── Public API ────────────────────────────────────────────────────────────────

def render_enemy_memory_debug(
    memory,
    *,
    visible_enemies: Optional[Sequence[Cell]] = None,
    step: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12.0, 5.0),
    dpi: int = 120,
    presence_cmap: str = "Blues",
    decay_cmap: str = "Reds",
    enemy_marker: str = "x",
    enemy_color: str = "lime",
    enemy_size: float = 80.0,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a two-panel debug figure for *memory*.

    Left panel — **presence heatmap**: cells with active memory are blue; 0.0
    (no memory) is white.  When diffusion is enabled, neighbours of active cells
    will show a gradient proportional to the spread.

    Right panel — **decay heatmap**: confidence e^(−λ·age) in [0, 1]; cells
    that have never been seen, or have expired, are white.

    Both panels share the same (col, row) coordinate system so overlaid enemy
    markers align with the heatmap cells.

    Args:
        memory: An :class:`~openra_env.enemy_memory.EnemySpatialMemory` instance.
        visible_enemies: Sequence of ``(row, col)`` cells marking the *current*
            visible enemy positions (i.e. what the agent sees *this* step).
            Plotted as cross markers on both panels.  Pass ``None`` to omit.
        step: RL step index.  Shown in the figure title when provided.
        save_path: File path to write the PNG (or any matplotlib-supported
            format).  Directory is created if it does not exist.  Pass ``None``
            to skip saving.
        show: When ``True`` (default), call ``plt.show()`` which blocks in
            interactive environments.  Set ``False`` for headless rendering
            (e.g. CI, video generation).
        figsize: ``(width, height)`` in inches passed to ``plt.figure()``.
        dpi: Dots-per-inch for saved files and screen rendering.
        presence_cmap: Matplotlib colormap name for the presence panel.
        decay_cmap: Matplotlib colormap name for the decay panel.
        enemy_marker: Matplotlib marker character for visible-enemy overlays.
        enemy_color: Colour of the enemy-overlay markers.
        enemy_size: Marker size (pts²) for enemy overlays.

    Returns:
        The ``matplotlib.figure.Figure`` object.  You can further annotate or
        save it yourself if needed.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for debug visualisation. "
            "Install it with: pip install matplotlib"
        ) from exc

    planes = _extract_planes(memory)
    presence = planes["presence"]
    decay    = planes["decay"]
    ever_seen = planes["ever_seen"]

    title = _build_title(memory, step, planes)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle(title, fontsize=10)

    _plot_heatmap(
        axes[0],
        presence,
        cmap=presence_cmap,
        vmin=0.0,
        vmax=1.0,
        label="memory_presence",
    )
    _plot_heatmap(
        axes[1],
        decay,
        cmap=decay_cmap,
        vmin=0.0,
        vmax=1.0,
        label="memory_decay",
    )
    _plot_heatmap(
        axes[2],
        ever_seen,
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        label="ever_seen",
    )

    if visible_enemies:
        rows, cols = zip(*visible_enemies)
        for ax in axes:
            ax.scatter(
                cols,
                rows,
                marker=enemy_marker,
                c=enemy_color,
                s=enemy_size,
                linewidths=1.5,
                label="visible enemy",
                zorder=5,
            )
        axes[0].legend(
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )

    if save_path is not None:
        _save_figure(fig, save_path, dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def render_memory_sequence(
    frames: Sequence[dict],
    output_dir: Union[str, Path],
    *,
    filename_template: str = "frame_{step:06d}.png",
    **render_kwargs,
) -> List[Path]:
    """Render a sequence of memory snapshots to numbered PNG files.

    This is a convenience wrapper for batch-rendering all frames from a
    recorded episode, ready to assemble into a video with ``ffmpeg``.

    Args:
        frames: List of dicts, each with keys:

            * ``"memory"`` — :class:`~openra_env.enemy_memory.EnemySpatialMemory`
              instance (or a snapshot returned by a recording helper).
            * ``"step"`` — integer step index (used in the filename and title).
            * ``"visible_enemies"`` — optional list of ``(row, col)`` cells.

        output_dir: Directory where PNG files are written.  Created if absent.
        filename_template: Python format string using ``{step}`` for the step
            index.  Default: ``"frame_{step:06d}.png"``.
        **render_kwargs: Extra keyword arguments forwarded to
            :func:`render_enemy_memory_debug` (e.g. ``dpi``, ``figsize``).

    Returns:
        List of :class:`~pathlib.Path` objects for each written file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # show is always False in batch mode — remove it from render_kwargs if
    # the caller accidentally passed it to avoid a duplicate-keyword error.
    render_kwargs.pop("show", None)

    paths: List[Path] = []
    for frame in frames:
        step = frame.get("step", 0)
        fname = out_dir / filename_template.format(step=step)
        render_enemy_memory_debug(
            frame["memory"],
            visible_enemies=frame.get("visible_enemies"),
            step=step,
            save_path=fname,
            show=False,
            **render_kwargs,
        )
        paths.append(fname)

    return paths


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_planes(memory) -> dict:
    """Pull the relevant numpy arrays out of a memory object."""
    # Support both EnemySpatialMemory (new) and EnemySpatialMemory (old-style)
    planes: dict = {}

    # New-style: has _presence, _decay, _ever_seen as direct attributes
    for attr, key in [
        ("_presence",  "presence"),
        ("_decay",     "decay"),
        ("_ever_seen", "ever_seen"),
        ("_age",       "age"),
    ]:
        if hasattr(memory, attr):
            planes[key] = getattr(memory, attr)
        else:
            planes[key] = np.zeros((1, 1), dtype=np.float32)

    # Fall back to get_channels() for the old three-channel class
    if "presence" not in planes and hasattr(memory, "get_channels"):
        ch = memory.get_channels()
        planes["presence"]  = ch[:, :, 0]
        planes["decay"]     = ch[:, :, 2]
        planes["ever_seen"] = np.zeros_like(ch[:, :, 0])
        planes["age"]       = ch[:, :, 1]   # age_norm in old API

    return planes


def _build_title(memory, step: Optional[int], planes: dict) -> str:
    h = getattr(memory, "height", getattr(memory, "map_h", "?"))
    w = getattr(memory, "width",  getattr(memory, "map_w", "?"))
    active = int((planes["presence"] > 0).sum())
    step_str = f"step {step} — " if step is not None else ""
    return (
        f"{step_str}map {h}×{w}  |  "
        f"active cells: {active}  |  "
        f"ever seen: {int((planes['ever_seen'] > 0).sum())}"
    )


def _plot_heatmap(
    ax,
    data: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    """Draw a single heatmap panel with a colourbar and axis labels."""
    import matplotlib.pyplot as plt  # local import — already checked by caller

    im = ax.imshow(
        data,
        origin="upper",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("col (cell_x)", fontsize=8)
    ax.set_ylabel("row (cell_y)", fontsize=8)
    ax.tick_params(labelsize=7)


def _save_figure(fig, save_path: Union[str, Path], dpi: int) -> None:
    """Write *fig* to *save_path*, creating parent directories as needed."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
