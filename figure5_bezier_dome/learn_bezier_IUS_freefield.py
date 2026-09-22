"""
Figure 03 (part 1): Bézier-lens optimisation against the free-field target.

Six-panel companion to ``learn_bezier_IUS.py``, laid out as

    a) free-field pressure distribution produced by the acoustic hologram
       (the optimisation *target* -- no lens in the domain)
    b) pressure distribution with the final, optimised Bézier lens
    c) on-axis profiles: free-field vs initial lens vs final lens
    d) control-point trajectory through the optimisation
    e) loss-function value vs optimisation step

Physics, geometry and optimiser settings are copied verbatim from
``learn_bezier_IUS.py`` (``--readout peak``, the default) so the fields and the
control-point trajectory reproduce ``bezier_learning_IUS.*``.  Pass
``--readout amplitude`` to rebuild the same layout from the steady-state CW
amplitude variant of ``learn_bezier_IUS_steady.py`` instead.

The run caches every array the figure needs to an ``.npz`` next to the outputs,
so styling can be iterated with ``--replot`` without re-running the ~13 min
optimisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FILENAME = "bezier_learning_IUS_freefield"

# ---------------------------------------------------------------------------
# Grid and simulation config (identical to learn_bezier_IUS.py)
# ---------------------------------------------------------------------------
Nx, Nr = 256, 128
dx = dr = 0.5e-3
c0, rho0 = 1500.0, 1000.0
source_freq = 0.3e6
cfl = 0.1
dt = cfl * dx / c0
ramp_steps = round(3 * (1 / source_freq) / dt)

# Steady-state readout averages the amplitude over the final RECORD_CYCLES periods
RECORD_CYCLES = 3

zpos = (Nx / 2) * dx - 40e-3
rpos = 37e-3
H117_FOCUS_X = zpos + 60e-3  # focal length

# Per-readout objective/optimiser settings, matching the two source scripts
READOUT_SETTINGS = {
    # learn_bezier_IUS.py
    "peak": dict(roi_size=(30e-3, 10e-3), lr=0.1, cbar="Peak Pressure [MPa]"),
    # learn_bezier_IUS_steady.py
    "amplitude": dict(roi_size=(50e-3, 15e-3), lr=0.2, cbar="Pressure Amplitude [MPa]"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--readout",
        choices=sorted(READOUT_SETTINGS),
        default="peak",
        help="max|p| over all time (peak, default) or steady-state CW amplitude",
    )
    p.add_argument("--steps", type=int, default=100, help="optimisation steps")
    p.add_argument(
        "--replot",
        action="store_true",
        help="skip the optimisation and rebuild the figure from the cached .npz",
    )
    return p.parse_args()


def cache_path(readout: str) -> Path:
    return OUT / f"{FILENAME}_{readout}.npz"


# ---------------------------------------------------------------------------
# Simulation / optimisation
# ---------------------------------------------------------------------------
def run(readout: str, n_steps: int) -> dict[str, np.ndarray]:
    from jax import config

    config.update("jax_enable_x64", False)

    import jax
    import jax.numpy as jnp
    import optax
    from jaxisymmetric import (
        BoundedParam,
        SimConfig,
        Source,
        run_simulation,
        run_simulation_amplitude,
    )
    from jaxisymmetric.geometry import BezierGeometry
    from jaxisymmetric.loss import IntersectionPenalty, RoiMseLoss, rectangular_roi
    from jaxisymmetric.sources import make_holography_source
    from jaxisymmetric.train import run_optimization
    from scipy.io import loadmat

    settings = READOUT_SETTINGS[readout]
    cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

    if readout == "peak":
        def forward(medium):
            return run_simulation(medium, cfg, source)
    else:
        def forward(medium):
            return run_simulation_amplitude(
                medium, cfg, source, record_cycles=RECORD_CYCLES
            )

    # -- hologram source ----------------------------------------------------
    rdata = loadmat(OUT.parent / "data" / "rprofile_FF.mat")
    r_centers = jnp.array(rdata["r_centers"]).squeeze()
    radial_prof = jnp.array(rdata["radial_prof"]).squeeze()

    src_mask = make_holography_source(
        cfg,
        source_zpos=zpos,
        source_rpos=rpos,
        r_centers=r_centers,
        radial_prof=radial_prof,
    )
    source_binary = jnp.abs(src_mask) > 0
    source = Source(mask=src_mask, freq=source_freq, ramp_steps=ramp_steps)

    # -- Bézier geometry ----------------------------------------------------
    P1 = (zpos - 2e-3, rpos + 2e-3)
    P2 = (zpos + 46e-3, 15e-3)
    initial_cp = jnp.array([(P1[0] + P2[0]) / 2, (P2[1] + P1[1]) / 2])

    geometry = BezierGeometry(
        c=2500.0,
        rho=1200.0,
        control_point=BoundedParam.from_physical(
            initial_cp,
            lower=jnp.array([P1[0], P2[1]]),
            upper=jnp.array([P2[0], 2 * P1[1]]),
        ),
        P1=P1,
        P2=P2,
        thickness=1.0e-3,
    )

    # -- panel a): free-field target ---------------------------------------
    print(f"Running baseline simulation (free-field target, readout={readout})...")
    target_field = jax.jit(forward)(cfg.homogeneous_medium())
    target_field.block_until_ready()
    print("  baseline done.\n")

    roi_mask = rectangular_roi(
        cfg.X,
        cfg.R,
        focus_x=H117_FOCUS_X,
        focus_r=0.0,
        size_x=settings["roi_size"][0],
        size_r=settings["roi_size"][1],
    )
    loss_obj = RoiMseLoss(target_field, roi_mask) + 10.0 * IntersectionPenalty(
        source_binary
    )

    def loss_fn(geom):
        p = forward(geom.as_medium(cfg))
        return loss_obj(p, geom(cfg.X, cfg.R)), p

    result = run_optimization(
        loss_fn,
        geometry,
        n_steps=n_steps,
        opt=optax.adam(settings["lr"]),
        verbose=True,
        log_every=5,
        has_aux=True,
    )

    # ``run_optimization`` stores the *post-update* geometry alongside the
    # *pre-update* loss/aux, so aux_history[0] is the field of the untouched
    # initial geometry while geometry_history[-1] is the final geometry.  Take
    # the final field from a fresh forward run of the final geometry rather
    # than from aux_history[-1] (which lags it by one update).
    final_geometry = result.geometry
    print("\nRunning final simulation on the optimised geometry...")
    final_field = jax.jit(forward)(final_geometry.as_medium(cfg))
    final_field.block_until_ready()

    # Control-point trajectory, including the true initial position
    cps = jnp.stack(
        [geometry.control_point.value]
        + [g.control_point.value for g in result.geometry_history]
    )

    return dict(
        x=np.asarray((jnp.arange(Nx) * dx - zpos) * 1e3),
        r=np.asarray(cfg.r * 1e3),
        target_field=np.asarray(target_field),
        initial_field=np.asarray(result.aux_history[0]),
        final_field=np.asarray(final_field),
        source_binary=np.asarray(source_binary, dtype=np.float32),
        initial_mask=np.asarray(geometry(cfg.X, cfg.R)),
        final_mask=np.asarray(final_geometry(cfg.X, cfg.R)),
        loss_history=np.asarray(result.loss_history),
        cps=np.asarray((cps - jnp.array([zpos, 0.0])) * 1e3),
        P1=np.asarray([(P1[0] - zpos) * 1e3, P1[1] * 1e3]),
        P2=np.asarray([(P2[0] - zpos) * 1e3, P2[1] * 1e3]),
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Journal geometry: the figure is laid out in millimetres at its final printed
# size (183 mm double-column width), so nothing is rescaled downstream and the
# 8 pt text stays 8 pt on the page.  Save without ``bbox_inches="tight"`` --
# that would crop the canvas and change the width.
FIG_W_MM, FIG_H_MM = 183.0, 95.0
FONT_PT = 8.0

# Top row: three square field panels, each with its own colourbar.
FIELD_S_MM = 31.7
FIELD_BOTTOM_MM = 58.0
FIELD_LEFT_MM = (13.0, 73.7, 134.4)
CBAR_PAD_MM, CBAR_W_MM = 1.5, 2.5

# Bottom row: three line panels.
LINE_W_MM, LINE_H_MM = 47.3, 33.0
LINE_BOTTOM_MM = 12.0
LINE_LEFT_MM = (13.0, 73.3, 133.6)

# Panel letters sit just outside the top-left corner of each axes; a larger
# offset pushes the leftmost column's letter off the canvas.
LABEL_X = -0.03

# Line weights and marker sizes, scaled for 31.7 mm panels rather than the
# 100 mm ones an on-screen figure would have.  Tune here, not inline.
LW_CONTOUR, LW_TRACE, LW_TRACK = 0.9, 1.0, 0.7
MS_FIXED, MEW_FIXED = 4.0, 1.2
MS_INITIAL, MS_FINAL = 5.0, 7.0
MS_TRACK_INITIAL, MS_TRACK_FINAL = 6.0, 9.0
SCATTER_S = 8.0

# Sparse ticks: at 31.7 mm a field panel cannot carry seven 8 pt labels.
# An 8 pt legend is ~14 x 9 mm; a field panel is 31.7 mm square, so there is
# room for at most one key per panel and none at all in c), where the control
# point markers sit at r = +/-56 mm and the lens arc fills the upper left.
# Each marker is therefore explained exactly once: hologram in a), fixed
# endpoints in b), initial/final control point in e).
HOLOGRAM_LEGEND_LOC = "lower right"
FIXED_LEGEND_LOC = "upper right"
FIELD_XTICKS = (0, 50, 100)
FIELD_YTICKS = (-50, 0, 50)

VMAX = 0.8  # colour-scale ceiling [MPa], shared by all three field panels
LOSS_SCALE = 1e9  # loss is O(1e9); plot in these units
# Explicit majors, minors unlabelled: a decade-spanning log axis would
# otherwise crowd 0.8/0.9 against 1.
LOSS_YTICKS = (1, 2, 3)
# Plain Unicode, not mathtext: mathtext would render the exponent in DejaVu
# Sans at a smaller size, breaking the uniform 8 pt Arial. Arial has U+00D7/U+2079.
LOSS_YLABEL = "Loss Function Value [" + chr(0x00D7) + "10" + chr(0x2079) + "]"
DARK_LEGEND = dict(facecolor="0.25", edgecolor="0.4", labelcolor="white", framealpha=0.85)


def _apply_style() -> None:
    """Uniform 8 pt Arial, and text kept as text in the vector outputs."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # ``path`` (the default) converts every glyph to outlines, which is
            # why the text in the earlier SVG could not be edited.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,  # embed TrueType, not Type 3
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": FONT_PT,
            "axes.labelsize": FONT_PT,
            "axes.titlesize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            # U+2212 often renders as a missing glyph once the font is no
            # longer embedded; a plain hyphen survives any SVG editor.
            "axes.unicode_minus": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.0,
            "grid.linewidth": 0.4,
            "lines.linewidth": LW_TRACE,
            # Compact legend boxes without shrinking the text.
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3,
        }
    )


def _rect(left_mm, bottom_mm, w_mm, h_mm):
    """Millimetre rectangle -> matplotlib figure-fraction rectangle."""
    return [
        left_mm / FIG_W_MM,
        bottom_mm / FIG_H_MM,
        w_mm / FIG_W_MM,
        h_mm / FIG_H_MM,
    ]


def _mirror(field: np.ndarray) -> np.ndarray:
    """Reflect an (Nx, Nr) half-plane field about the axis for display."""
    return np.concatenate([field[:, 1:][:, ::-1], field], axis=1)


def _panel_label(ax, letter: str) -> None:
    ax.text(
        LABEL_X,
        1.02,
        f"{letter})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )


def _draw_field(fig, ax, d, field, cbar_label, left_mm):
    """Common imshow + hologram-source overlay for the three field panels."""
    x, r = d["x"], d["r"]
    full_r = np.concatenate([-r[1:][::-1], r])
    extent = [x[0], x[-1], full_r[-1], full_r[0]]

    im = ax.imshow(
        _mirror(field).T / 1e6,
        extent=extent,
        cmap="magma",
        origin="upper",
        aspect="auto",  # box is already square; keeps the data filling it
        vmin=0.0,
        vmax=VMAX,
    )
    for sign in (1, -1):
        ax.contour(
            x,
            sign * r,
            d["source_binary"].T,
            levels=[0.5],
            colors="cyan",
            linewidths=LW_CONTOUR,
        )
    ax.set_xlabel("Axial Position [mm]")
    ax.set_ylabel("Radial Position [mm]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks(FIELD_XTICKS)
    ax.set_yticks(FIELD_YTICKS)

    cax = fig.add_axes(
        _rect(
            left_mm + FIELD_S_MM + CBAR_PAD_MM,
            FIELD_BOTTOM_MM,
            CBAR_W_MM,
            FIELD_S_MM,
        )
    )
    cbar = fig.colorbar(im, cax=cax, label=cbar_label)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, size=2.5, pad=2.0)
    return im


def make_figure(d: dict[str, np.ndarray], readout: str) -> None:
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

    cbar_label = READOUT_SETTINGS[readout]["cbar"]
    x, cps = d["x"], d["cps"]

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))

    def field_axes(slot):
        return fig.add_axes(
            _rect(FIELD_LEFT_MM[slot], FIELD_BOTTOM_MM, FIELD_S_MM, FIELD_S_MM)
        )

    def line_axes(slot):
        return fig.add_axes(
            _rect(LINE_LEFT_MM[slot], LINE_BOTTOM_MM, LINE_W_MM, LINE_H_MM)
        )

    def lens_overlay(ax, mask, cp, marker, marker_size, legend_loc=None):
        """Lens outline, fixed endpoints and the control point of one geometry."""
        for sign in (1, -1):
            ax.contour(
                x,
                sign * d["r"],
                mask.T,
                levels=[0.5],
                colors="white",
                linewidths=LW_CONTOUR,
            )
        for pt in (d["P1"], d["P2"]):
            for sign in (1, -1):
                ax.plot(
                    [pt[0]],
                    [sign * pt[1]],
                    "wx",
                    markersize=MS_FIXED,
                    markeredgewidth=MEW_FIXED,
                )
        for sign in (1, -1):
            ax.plot([cp[0]], [sign * cp[1]], marker, markersize=marker_size)
        if legend_loc is not None:
            ax.legend(
                handles=[
                    Line2D(
                        [], [], color="white", marker="x", ls="",
                        ms=MS_FIXED, mew=MEW_FIXED, label="Fixed",
                    )
                ],
                loc=legend_loc,
                **DARK_LEGEND,
            )

    # -- a) free-field target ----------------------------------------------
    ax_a = field_axes(0)
    _draw_field(fig, ax_a, d, d["target_field"], cbar_label, FIELD_LEFT_MM[0])
    ax_a.legend(
        handles=[Line2D([], [], color="cyan", lw=LW_CONTOUR, label="Hologram")],
        loc=HOLOGRAM_LEGEND_LOC,
        **DARK_LEGEND,
    )
    _panel_label(ax_a, "a")

    # -- b) initial "cone" guess -------------------------------------------
    ax_b = field_axes(1)
    _draw_field(fig, ax_b, d, d["initial_field"], cbar_label, FIELD_LEFT_MM[1])
    lens_overlay(ax_b, d["initial_mask"], cps[0], "gs", MS_INITIAL, FIXED_LEGEND_LOC)
    _panel_label(ax_b, "b")

    # -- c) final optimised lens -------------------------------------------
    ax_c = field_axes(2)
    _draw_field(fig, ax_c, d, d["final_field"], cbar_label, FIELD_LEFT_MM[2])
    lens_overlay(ax_c, d["final_mask"], cps[-1], "r*", MS_FINAL)
    _panel_label(ax_c, "c")

    # -- d) on-axis profiles ------------------------------------------------
    ax_d = line_axes(0)
    ax_d.plot(x, d["target_field"][:, 0] / 1e6, color="C0", label="Free-Field")
    ax_d.plot(x, d["initial_field"][:, 0] / 1e6, color="C2", ls="--", label="Initial")
    ax_d.plot(x, d["final_field"][:, 0] / 1e6, color="C1", ls="-.", label="Final")
    ax_d.set_xlabel("Axial Position [mm]")
    ax_d.set_ylabel(cbar_label)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_xlim(x[0], x[-1])
    ax_d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.grid(True, alpha=0.3)
    ax_d.legend()
    _panel_label(ax_d, "d")

    # -- e) control-point trajectory ---------------------------------------
    ax_e = line_axes(1)
    ax_e.plot(cps[:, 0], cps[:, 1], "-", color="0.75", lw=LW_TRACK, zorder=1)
    ax_e.scatter(
        cps[:, 0],
        cps[:, 1],
        c=np.arange(len(cps)),
        cmap="plasma",
        s=SCATTER_S,
        zorder=2,
        linewidths=0,
    )
    ax_e.plot(
        cps[0, 0], cps[0, 1], "gs", markersize=MS_TRACK_INITIAL,
        label="Initial", zorder=3,
    )
    ax_e.plot(
        cps[-1, 0], cps[-1, 1], "r*", markersize=MS_TRACK_FINAL,
        label="Final", zorder=3,
    )
    ax_e.set_xlabel("Control Point Axial Position [mm]")
    ax_e.set_ylabel("Control Point Radial Position [mm]")
    ax_e.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="upper left")
    _panel_label(ax_e, "e")

    # -- f) loss curve ------------------------------------------------------
    ax_f = line_axes(2)
    loss = d["loss_history"] / LOSS_SCALE
    ax_f.plot(np.arange(len(loss)), loss, color="C0")
    ax_f.set_yscale("log")
    # plain "1, 2, 3" rather than the log axis's default 10^0 / 1.00 mix
    ax_f.set_yticks(LOSS_YTICKS)
    ax_f.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_f.yaxis.set_minor_formatter(NullFormatter())
    ax_f.set_xlabel("Optimisation Step")
    ax_f.set_ylabel(LOSS_YLABEL)
    ax_f.set_xlim(0, max(len(loss) - 1, 1))
    ax_f.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_f.grid(True, alpha=0.3, which="both")
    _panel_label(ax_f, "f")

    stem = f"{FILENAME}_{readout}"
    for ext in ("pdf", "png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=600)
        print(f"  wrote {path}")


def main() -> None:
    args = parse_args()
    cache = cache_path(args.readout)

    if args.replot:
        if not cache.exists():
            raise SystemExit(f"No cache at {cache}; run without --replot first.")
        d = dict(np.load(cache))
        print(f"Re-plotting from {cache}")
    else:
        d = run(args.readout, args.steps)
        np.savez_compressed(cache, **d)
        print("")
        print(f"Cached run data to {cache}")

    make_figure(d, args.readout)


if __name__ == "__main__":
    main()
