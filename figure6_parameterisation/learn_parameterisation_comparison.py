"""
Figure 03 (part 2): lens-parameterisation comparison.

One self-consistent study of the three lens parameterisations that Figure 3
compares -- an RBF radial profile, a single cubic-Hermite spline, and a
five-spline union -- optimised against the *same* objective, on the *same*
grid, for the *same* number of steps, so the resulting geometries and fields
can be read side by side.

    a) optimised RBF lens + pressure field
    b) optimised single-spline lens + pressure field
    c) optimised 5-spline lens + pressure field
    d) on-axis pressure profiles, free-field vs the three lenses
    e) mean ROI pressure vs optimisation step
    f) radial pressure profiles averaged over the ROI's axial extent

What is held identical across the three runs
--------------------------------------------
Grid (512 x 256 at 0.25 mm, the finer of the two source scripts' grids and the
one ``learn_multispline.py`` uses), hologram source, background medium, lens
material, focal ROI, objective (``FocalPressureLoss`` + 10x
``IntersectionPenalty``), readout, and the step budget -- 200 steps, set by the
5-spline lens, which has by far the most parameters and so converges slowest.

What deliberately still differs
-------------------------------
Two per-method settings are carried over from the source scripts rather than
forced equal, because they are part of each parameterisation's definition and
not of the comparison:

* **Learning rate** (RBF 0.3, spline 0.2, 5-spline 0.05).  Adam steps act in
  the *latent* space of each :class:`BoundedParam`, so a shared value is not
  the same physical step size for a 6-parameter RBF and a 100-parameter
  5-spline; the 5-spline run oscillates at 0.2.  Pass ``--lr`` to force one
  rate on all three if a strict same-optimiser comparison is wanted.
* **Shell thickness** (RBF and single spline 2.0 mm, each 5-spline tube
  1.0 mm), again as in the source scripts: five 2 mm tubes would put far more
  material in the beam than one.  Override with ``--thickness-*``.

Both are printed in the run header and stored in the cache, so whichever
choice ends up in the manuscript is recoverable from the ``.npz``.

Sources merged here: ``learn_rbf+spline_steady.py`` (RBF + single spline,
steady-state readout, 50 steps) and ``learn_multispline.py`` (5-spline, peak
readout, 200 steps).

Transducer model
----------------
``--source hologram`` (default) injects the measured radial amplitude/phase
profile on a single plane at ``zpos`` -- what every figure-3 script has used.
``--source bowl`` instead models the transducer as its NOMINAL curved geometry:
a spherical shell with figure 2's H-117 parameters (ROC 63.2 mm, outer 64 mm,
22.6 mm central hole), placed so its geometric focus lands on the ROI centre.

The bowl mask is binary, i.e. a 1 Pa drive, while the hologram mask carries the
measured amplitudes in Pa.  The bowl is therefore rescaled so both models put
the same free-field pressure in the ROI, making the absolute MPa values
comparable between the two analyses; the gains are ratios and so would be
unaffected either way.  Normalisation always uses the amplitude readout, so the
peak and amplitude runs share one physical source.

Outputs for the bowl are named ``parameterisation_comparison_bowl_<readout>.*``;
the hologram keeps the original unprefixed names.

Readout
-------
``--readout amplitude`` (default) trains and reports the steady-state CW
pressure amplitude, matching the rest of the paper's ``_steady`` figures.
``--readout peak`` uses ``max|p|`` over all time instead, which is what
``learn_multispline.py`` originally trained on -- keep it in mind if the
5-spline run turns out to be less well behaved on the amplitude readout.

Caching
-------
Every array the figure needs is cached to an ``.npz`` beside the outputs, so
styling can be iterated with ``--replot`` without repeating the optimisation.
``--methods`` restricts a run to a subset; results for methods that are absent
from a partial run are carried over from the existing cache, so the three
optimisations can be run one at a time and the figure assembled at the end.

Run with::

    python figure3/learn_parameterisation_comparison.py
    python figure3/learn_parameterisation_comparison.py --replot
    python figure3/learn_parameterisation_comparison.py --source bowl --readout peak
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FILENAME = "parameterisation_comparison"

# ---------------------------------------------------------------------------
# Grid and simulation config
# ---------------------------------------------------------------------------
# 512 x 256 at 0.25 mm: the grid of learn_multispline.py, shared by all three
# methods here.  Nx and Nr are powers of two, which matters because the
# solver's spectral derivatives pad each FFT axis up to the next power of two.
Nx, Nr = 512, 256
dx = dr = 0.25e-3
c0, rho0 = 1500.0, 1000.0
source_freq = 0.3e6
cfl = 0.1
dt = cfl * dx / c0
ramp_steps = round(3 * (1 / source_freq) / dt)

# Steady-state readout averages the amplitude over the final RECORD_CYCLES periods
RECORD_CYCLES = 3

zpos = (Nx / 2) * dx - 40e-3  # source plane; x is measured from here
rpos = 37e-3
H117_FOCUS_X = zpos + 60e-3  # geometric focus of the H-117

# Focal ROI (identical in both source scripts), full extents
ROI_SIZE_X, ROI_SIZE_R = 5e-3, 2e-3

# Lens material (identical in both source scripts)
LENS_C, LENS_RHO = 2500.0, 1200.0

READOUT_CBAR = {
    "peak": "Peak Pressure [MPa]",
    "amplitude": "Pressure Amplitude [MPa]",
}

# ---------------------------------------------------------------------------
# Transducer models
# ---------------------------------------------------------------------------
# "hologram" is the measured radial amplitude/phase profile injected on a single
# plane at ``zpos`` -- what every figure-3 script has used so far.  "bowl" is the
# NOMINAL H-117 geometry: a spherical shell, same parameters as figure 2's
# freefield_comparison.py, positioned so its geometric focus sits on the ROI
# centre.  The shell is given figure 2's 0.5 mm thickness explicitly rather than
# defaulting to ``cfg.dx``; at 0.25 mm a one-cell shell is only marginally
# watertight, and a leaky hard-Dirichlet source lets sound through the bowl.
SOURCE_MODELS = ("hologram", "bowl")
BOWL_ROC = 63.2e-3
BOWL_OUTER_D = 64e-3
BOWL_INNER_D = 22.6e-3  # central hole
BOWL_SHELL_TH = 0.5e-3

METHODS = ("rbf", "spline", "multispline")

# Defaults inherited from the source scripts; see the module docstring.
DEFAULT_LR = {"rbf": 0.3, "spline": 0.2, "multispline": 0.05}
DEFAULT_THICKNESS = {"rbf": 2.0e-3, "spline": 2.0e-3, "multispline": 1.0e-3}

# 5-spline lens layout
N_SPLINES, N_CTRL = 5, 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--readout",
        choices=sorted(READOUT_CBAR),
        default="amplitude",
        help="steady-state CW amplitude (default) or max|p| over all time",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="optimisation steps, applied to every method (default: 200)",
    )
    p.add_argument(
        "--source",
        choices=SOURCE_MODELS,
        default="hologram",
        help="transducer model: the measured hologram plane (default) or the "
        "nominal curved H-117 bowl",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="run only these methods; the rest are reused from the cache",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="force one learning rate on all three methods "
        f"(default: per-method {DEFAULT_LR})",
    )
    for m in METHODS:
        p.add_argument(f"--lr-{m}", type=float, default=None, help=argparse.SUPPRESS)
        p.add_argument(
            f"--thickness-{m}",
            type=float,
            default=None,
            help=f"{m} lens thickness [m] (default: {DEFAULT_THICKNESS[m]})",
        )
    p.add_argument(
        "--save-history",
        type=int,
        default=0,
        metavar="STRIDE",
        help="also cache the field and lens mask every STRIDE steps, so the "
        "optimisation can be animated afterwards (0 = off, the default). "
        "Costs ~0.5 MB per stored frame per method and nothing in run time, "
        "but it cannot be added retrospectively: a run without it keeps only "
        "the first and last geometry",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="colour-scale ceiling [MPa] shared by a), b) and c) "
        "(default: rounded up from the brightest of the three fields)",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help="skip the optimisation and rebuild the figure from the cached .npz",
    )
    return p.parse_args()


def output_stem(readout: str, source_model: str) -> str:
    """Output basename. The default hologram source keeps the original names so
    the caches and figures already produced stay valid."""
    base = FILENAME if source_model == "hologram" else f"{FILENAME}_{source_model}"
    return f"{base}_{readout}"


def cache_path(readout: str, source_model: str = "hologram") -> Path:
    return OUT / f"{output_stem(readout, source_model)}.npz"


def resolve_settings(args: argparse.Namespace) -> dict[str, dict]:
    """Per-method learning rate and thickness after applying the overrides."""
    settings = {}
    for m in METHODS:
        lr = getattr(args, f"lr_{m}")
        if lr is None:
            lr = args.lr if args.lr is not None else DEFAULT_LR[m]
        thickness = getattr(args, f"thickness_{m}")
        if thickness is None:
            thickness = DEFAULT_THICKNESS[m]
        settings[m] = dict(lr=float(lr), thickness=float(thickness))
    return settings


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------
def build_rbf(thickness: float):
    """RBF radial profile, as in ``learn_rbf+spline_steady.py``."""
    import jax.numpy as jnp
    from jaxisymmetric import BoundedParam
    from jaxisymmetric.geometry import RbfGeometry

    return RbfGeometry(
        c=LENS_C,
        rho=LENS_RHO,
        rbf_weights=BoundedParam.from_physical(
            jnp.array([35e-3, 20e-3, 25e-3, 20e-3, 15e-3]),
            lower=jnp.array([0e-3, 2e-3, 2e-3, 2e-3, 2e-3]),
            upper=jnp.array([40e-3, 40e-3, 40e-3, 40e-3, 30e-3]),
        ),
        r_end=BoundedParam.from_physical(
            jnp.array(10.0e-3), lower=jnp.array(0.5e-3), upper=jnp.array(20e-3)
        ),
        x_start=zpos - 0.5e-3,
        x_end=zpos + 38e-3,
        r_start=rpos + 2e-3,
        thickness=thickness,
        bandwidth=0.15,
    )


def build_spline(thickness: float):
    """Single 6-point Hermite spline, as in ``learn_rbf+spline_steady.py``.

    Degenerate bounds (lower == upper) pin fixed coordinates with zero
    gradient: row 0 is fully fixed at the aperture rim, and row 5's axial
    coordinate is pinned to the common end plane while its radius stays free.
    """
    import jax.numpy as jnp
    from jaxisymmetric import BoundedParam
    from jaxisymmetric.geometry import SplineGeometry

    P1 = jnp.array([zpos - 0.5e-3, rpos + 2e-3])
    P6_x_fixed = zpos + 38e-3

    lower = jnp.array(
        [
            P1,
            [zpos + 2e-3, 2e-3],
            [zpos + 10e-3, 2e-3],
            [zpos + 18e-3, 2e-3],
            [zpos + 28e-3, 2e-3],
            [P6_x_fixed, 2e-3],
        ]
    )
    upper = jnp.array(
        [
            P1,
            [zpos + 8e-3, 60e-3],
            [zpos + 17e-3, 60e-3],
            [zpos + 27e-3, 60e-3],
            [zpos + 36e-3, 60e-3],
            [P6_x_fixed, 20e-3],
        ]
    )
    initial = jnp.array(
        [
            P1,
            [zpos + 8e-3, 40e-3],
            [zpos + 17e-3, 30e-3],
            [zpos + 27e-3, 20e-3],
            [zpos + 36e-3, 10e-3],
            [P6_x_fixed, 10e-3],
        ]
    )

    return SplineGeometry(
        c=LENS_C,
        rho=LENS_RHO,
        control_points=BoundedParam.from_physical(initial, lower=lower, upper=upper),
        thickness=thickness,
    )


def build_multispline(thickness: float):
    """Union of ``N_SPLINES`` Hermite tubes, as in ``learn_multispline.py``."""
    import jax.numpy as jnp
    from jaxisymmetric import BoundedParam
    from jaxisymmetric.geometry import MultiSplineGeometry

    x_start = zpos + 4e-3  # a few mm in front of the source, to avoid overlap
    x_end = zpos + 38e-3  # common end plane, shared with the single spline

    xs = np.linspace(x_start, x_end, N_CTRL)
    start_r = np.linspace(6e-3, 30e-3, N_SPLINES)
    end_r = np.linspace(3e-3, 12e-3, N_SPLINES)

    init = np.zeros((N_SPLINES, N_CTRL, 2))
    lower = np.zeros((N_SPLINES, N_CTRL, 2))
    upper = np.zeros((N_SPLINES, N_CTRL, 2))
    for n in range(N_SPLINES):
        init[n, :, 0] = xs
        init[n, :, 1] = np.linspace(start_r[n], end_r[n], N_CTRL)

    # Axial: interior points free in [x_start, x_end], first and last pinned so
    # every spline spans the same axial extent and shares the end plane.
    lower[:, :, 0] = x_start
    upper[:, :, 0] = x_end
    lower[:, 0, 0] = upper[:, 0, 0] = x_start
    lower[:, -1, 0] = upper[:, -1, 0] = x_end
    # Radial: free, with the end points allowed closer to the axis.
    lower[:, :, 1] = 2e-3
    upper[:, :, 1] = 40e-3
    lower[:, -1, 1] = 0.5e-3
    upper[:, -1, 1] = 30e-3

    return MultiSplineGeometry(
        c=LENS_C,
        rho=LENS_RHO,
        control_points=BoundedParam.from_physical(
            jnp.array(init), jnp.array(lower), jnp.array(upper)
        ),
        thickness=thickness,
        num_samples=250,
    )


BUILDERS = {"rbf": build_rbf, "spline": build_spline, "multispline": build_multispline}


def free_dof(geometry) -> int:
    """Number of genuinely learnable coordinates in a geometry.

    Counting latent entries alone overstates it: the source scripts pin fixed
    coordinates with degenerate :class:`BoundedParam` bounds (``lower ==
    upper``), which leaves the latent entry in the tree but gives it zero
    gradient.  Only the non-degenerate entries are counted here, so the
    comparison reports 6 / 9 / 90 rather than 6 / 12 / 100.
    """
    import jax
    from jaxisymmetric import BoundedParam

    total = 0
    leaves = jax.tree_util.tree_leaves(
        geometry, is_leaf=lambda n: isinstance(n, BoundedParam)
    )
    for leaf in leaves:
        if isinstance(leaf, BoundedParam):
            lower = np.asarray(leaf.lower, dtype=float)
            upper = np.asarray(leaf.upper, dtype=float)
            total += int(np.count_nonzero(upper > lower))
    return total


# ---------------------------------------------------------------------------
# Simulation / optimisation
# ---------------------------------------------------------------------------
def run(
    readout: str,
    n_steps: int,
    methods: list[str],
    settings: dict,
    checkpoint: "Path | None" = None,
    seed: dict | None = None,
    source_model: str = "hologram",
    stride: int = 0,
) -> dict:
    """Optimise each method in turn and collect everything the figure needs.

    Args:
        readout:    ``"amplitude"`` or ``"peak"``.
        n_steps:    Gradient steps per method.
        methods:    Which methods to optimise, in panel order.
        settings:   Per-method ``{"lr":, "thickness":}`` from
                    :func:`resolve_settings`.
        checkpoint: If given, the cache is rewritten after *each* method
                    finishes rather than only at the end.  A 200-step run is
                    hours per method, so a failure in the third one must not
                    throw away the first two.
        seed:       Existing cache contents to fold into those checkpoints, so
                    a partial re-run does not drop the methods it skipped.
        source_model: ``"hologram"`` (measured plane) or ``"bowl"`` (nominal
                    curved H-117). See :func:`build_source`.
        stride:     If > 0, keep the field and lens mask every ``stride`` steps
                    so the run can be animated. Off by default; see
                    ``--save-history``.
    """
    save_history = stride > 0
    from jax import config

    config.update("jax_enable_x64", False)

    import jax
    import jax.numpy as jnp
    import optax
    from jaxisymmetric import SimConfig, Source, run_simulation, run_simulation_amplitude
    from jaxisymmetric.loss import (
        FocalPressureLoss,
        IntersectionPenalty,
        rectangular_roi,
    )
    from jaxisymmetric.sources import make_focused_bowl_source, make_holography_source
    from jaxisymmetric.train import run_optimization
    from scipy.io import loadmat

    cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

    # -- transducer model ---------------------------------------------------
    rdata = loadmat(OUT.parent / "data" / "rprofile_FF.mat")
    holo_mask = make_holography_source(
        cfg,
        source_zpos=zpos,
        source_rpos=rpos,
        r_centers=jnp.array(rdata["r_centers"]).squeeze(),
        radial_prof=jnp.array(rdata["radial_prof"]).squeeze(),
    )

    roi_mask = rectangular_roi(
        cfg.X,
        cfg.R,
        focus_x=H117_FOCUS_X,
        focus_r=0.0,
        size_x=ROI_SIZE_X,
        size_r=ROI_SIZE_R,
    )
    roi_weight = roi_mask / jnp.sum(roi_mask)

    bowl_scale = 1.0
    if source_model == "hologram":
        src_mask = holo_mask
    else:
        # The hologram mask carries the measured amplitudes in Pa; the bowl mask
        # is binary, i.e. a 1 Pa drive.  Rescale the bowl so the two models put
        # the SAME free-field pressure in the ROI, which makes the absolute MPa
        # values comparable across the two analyses (the gains are ratios, so
        # they are normalisation-independent either way).  Normalisation always
        # uses the amplitude readout, so the peak and amplitude runs share one
        # physical source; the solver is linear, so a single scalar suffices.
        # Position the bowl by its APERTURE, not by its focus: the rim is the
        # transducer's rightmost point and must sit on x = 0, the same plane the
        # measured hologram is injected on.  Both source models then have the
        # same aperture-to-ROI distance (60 mm), which is what makes them
        # comparable, and the lens bounds -- which start at x = -0.5 mm -- can no
        # longer put a quarter of the lens behind the transducer.
        #
        # Consequence, and it is a real one: a 63.2 mm ROC bowl whose rim sits at
        # x = 0 has its geometric focus at sqrt(ROC^2 - (D/2)^2) = 54.25 mm, so
        # the nominal transducer focuses SHORT of the 60 mm ROI and the lens has
        # to make up the difference.
        def _bowl(focus_pos):
            return make_focused_bowl_source(
                cfg,
                focus_pos=focus_pos,
                radius_curvature=BOWL_ROC,
                outer_diameter=BOWL_OUTER_D,
                inner_diameter=BOWL_INNER_D,
                thickness=BOWL_SHELL_TH,
            )

        def _rim_index(mask):
            return int(jnp.max(jnp.nonzero(jnp.any(mask > 0, axis=1))[0]))

        # Shift by a whole number of cells so the discretised shell is identical
        # to the trial one, just translated, and the rim lands exactly on zpos.
        shift_cells = round((zpos - _rim_index(_bowl(H117_FOCUS_X)) * dx) / dx)
        bowl_focus = H117_FOCUS_X + shift_cells * dx
        unit_bowl = _bowl(bowl_focus)
        rim_x = _rim_index(unit_bowl) * dx
        assert abs(rim_x - zpos) < 1e-12, (
            f"bowl rim at x = {(rim_x - zpos) * 1e3:+.2f} mm, expected 0.00"
        )
        print("Normalising the nominal bowl against the hologram free field…")
        amp = jax.jit(
            lambda m, s: run_simulation_amplitude(
                m, cfg, s, record_cycles=RECORD_CYCLES
            )
        )
        ref = amp(
            cfg.homogeneous_medium(),
            Source(mask=holo_mask, freq=source_freq, ramp_steps=ramp_steps),
        )
        raw = amp(
            cfg.homogeneous_medium(),
            Source(mask=unit_bowl, freq=source_freq, ramp_steps=ramp_steps),
        )
        bowl_scale = float(jnp.sum(ref * roi_weight) / jnp.sum(raw * roi_weight))
        src_mask = unit_bowl * bowl_scale
        print(
            f"  bowl drive = {bowl_scale:.1f} Pa, "
            f"{int(jnp.sum(unit_bowl > 0))} source cells "
            f"(ROC {BOWL_ROC * 1e3:g} mm, outer {BOWL_OUTER_D * 1e3:g} mm, "
            f"hole {BOWL_INNER_D * 1e3:g} mm)"
        )
        print(
            f"  rim on x = 0 (shifted {shift_cells * dx * 1e3:+.2f} mm); "
            f"geometric focus x = {(bowl_focus - zpos) * 1e3:.2f} mm, "
            f"ROI centre x = {(H117_FOCUS_X - zpos) * 1e3:.2f} mm\n"
        )

    source_binary = jnp.abs(src_mask) > 0
    source = Source(mask=src_mask, freq=source_freq, ramp_steps=ramp_steps)

    if readout == "peak":

        def forward(medium):
            return run_simulation(medium, cfg, source)

    else:

        def forward(medium):
            return run_simulation_amplitude(
                medium, cfg, source, record_cycles=RECORD_CYCLES
            )

    jit_forward = jax.jit(forward)

    # -- objective ----------------------------------------------------------
    # roi_mask / roi_weight are built above, before the source, because the bowl
    # normalisation needs them.
    loss_obj = FocalPressureLoss(roi_mask) + 10.0 * IntersectionPenalty(source_binary)

    print(
        f"Grid {Nx} x {Nr} at {dx * 1e3:g} mm, Nt = {cfg.Nt}, "
        f"readout = {readout}, source = {source_model}"
    )
    print(f"ROI {ROI_SIZE_X * 1e3:g} x {ROI_SIZE_R * 1e3:g} mm at x = 60 mm, {n_steps} steps\n")

    # -- free-field reference ----------------------------------------------
    print("Running baseline (free-field) simulation...")
    t0 = time.perf_counter()
    target_field = jit_forward(cfg.homogeneous_medium())
    target_field.block_until_ready()
    print(f"  done in {time.perf_counter() - t0:.1f} s.\n")

    out: dict[str, np.ndarray] = dict(
        x=np.asarray((jnp.arange(Nx) * dx - zpos) * 1e3),
        r=np.asarray(cfg.r * 1e3),
        target_field=np.asarray(target_field),
        source_binary=np.asarray(source_binary, dtype=np.float32),
        roi_mask=np.asarray(roi_mask, dtype=np.float32),
        roi_x_mm=np.asarray([(H117_FOCUS_X - zpos) * 1e3, ROI_SIZE_X * 1e3]),
        roi_r_mm=np.asarray([0.0, ROI_SIZE_R * 1e3]),
        n_steps=np.asarray(n_steps),
        source_model=np.asarray(source_model),
        bowl_scale=np.asarray(bowl_scale),
    )

    for method in methods:
        lr = settings[method]["lr"]
        thickness = settings[method]["thickness"]
        geometry = BUILDERS[method](thickness)

        # aux is normally the mean ROI pressure -- a scalar, not the field,
        # because keeping the whole (Nx, Nr) field for every step costs ~100 MB
        # per method here and the ROI mean is all panel e) needs.  With
        # --save-history the field is kept instead, which is the only way to
        # animate the run: the intermediate geometries cannot be recovered
        # afterwards, and re-running does not reproduce them (XLA CPU
        # reductions are not bit-reproducible, so a repeat run diverges).
        def loss_fn(geom, _keep=save_history):
            p = forward(geom.as_medium(cfg))
            aux = p if _keep else jnp.sum(p * roi_weight)
            return loss_obj(p, geom(cfg.X, cfg.R)), aux

        n_dof = free_dof(geometry)
        print(
            f"--- {method}: {n_dof} free parameters, {n_steps} steps, "
            f"adam(lr={lr:g}), thickness = {thickness * 1e3:g} mm ---"
        )

        t_start = time.perf_counter()

        # verbose=False: run_optimization's own progress line contains U+2502,
        # which raises UnicodeEncodeError on a cp1252 Windows console.
        def progress(i, _geom, loss_val, _n=n_steps, _t0=t_start):
            if i % 5 and i != _n - 1:
                return
            elapsed = time.perf_counter() - _t0
            eta = elapsed / (i + 1) * (_n - i - 1)
            print(
                f"  step {i + 1:4d}/{_n} | loss = {loss_val:.4e} | "
                f"{elapsed / (i + 1):.1f} s/step | ETA {eta / 60:.0f} min",
                flush=True,
            )

        result = run_optimization(
            loss_fn,
            geometry,
            n_steps=n_steps,
            opt=optax.adam(lr),
            verbose=False,
            callback=progress,
            has_aux=True,
        )

        # run_optimization appends the *post-update* geometry alongside the
        # *pre-update* loss/aux, so aux_history[i] is the ROI pressure of the
        # geometry after i updates (aux_history[0] = untouched initial
        # geometry) and there is no aux entry for the final geometry.  Run one
        # more forward pass on result.geometry and append it, giving a trace
        # over steps 0..n_steps.
        final_geometry = result.geometry
        final_field = jit_forward(final_geometry.as_medium(cfg))
        final_field.block_until_ready()
        roi_w = np.asarray(roi_weight)
        final_roi = float(np.sum(np.asarray(final_field) * roi_w))
        if save_history:
            per_step = [float(np.sum(np.asarray(a) * roi_w)) for a in result.aux_history]
        else:
            per_step = [float(a) for a in result.aux_history]
        roi_history = np.asarray(per_step + [final_roi], dtype=np.float64)

        if save_history:
            # aux_history[i] is the field of the geometry after i updates, so
            # its geometry is the initial one at i = 0 and geometry_history[i-1]
            # after that.  Keep the final state as the last frame either way.
            idx = list(range(0, n_steps, stride))
            frames, masks = [], []
            for i in idx:
                frames.append(np.asarray(result.aux_history[i], dtype=np.float32))
                g = geometry if i == 0 else result.geometry_history[i - 1]
                masks.append(np.asarray(g(cfg.X, cfg.R)) > 0.5)
            frames.append(np.asarray(final_field, dtype=np.float32))
            masks.append(np.asarray(final_geometry(cfg.X, cfg.R)) > 0.5)
            out[f"field_history_{method}"] = np.stack(frames)
            out[f"mask_history_{method}"] = np.stack(masks).astype(np.uint8)
            out[f"frame_steps_{method}"] = np.asarray(idx + [n_steps])
            print(
                f"  kept {len(frames)} history frames "
                f"({np.stack(frames).nbytes / 1e6:.0f} MB of field)"
            )

        out[f"final_field_{method}"] = np.asarray(final_field)
        out[f"final_mask_{method}"] = np.asarray(final_geometry(cfg.X, cfg.R))
        out[f"initial_mask_{method}"] = np.asarray(geometry(cfg.X, cfg.R))
        out[f"roi_history_{method}"] = roi_history
        out[f"loss_history_{method}"] = np.asarray(result.loss_history)
        out[f"lr_{method}"] = np.asarray(lr)
        out[f"thickness_{method}"] = np.asarray(thickness)
        out[f"n_dof_{method}"] = np.asarray(n_dof)
        cps = getattr(final_geometry, "control_points", None)
        if cps is not None:
            out[f"control_points_{method}"] = np.asarray(cps.value)

        print(
            f"  done in {(time.perf_counter() - t_start) / 60:.1f} min | "
            f"ROI pressure {roi_history[0] / 1e6:.4f} -> {final_roi / 1e6:.4f} MPa"
        )

        if checkpoint is not None:
            merged = dict(seed or {})
            merged.update(out)
            np.savez_compressed(checkpoint, **merged)
            print(f"  checkpointed to {checkpoint}")
        print("")

    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Journal geometry: laid out in millimetres at the final printed size (183 mm
# double-column width), so nothing is rescaled downstream and the 8 pt text
# stays 8 pt on the page.  These numbers match learn_bezier_IUS_freefield.py so
# the two halves of Figure 3 sit together.  Save without bbox_inches="tight" --
# that would crop the canvas and change the width.
FIG_W_MM, FIG_H_MM = 183.0, 95.0
FONT_PT = 8.0

FIELD_S_MM = 31.7
FIELD_BOTTOM_MM = 58.0
FIELD_LEFT_MM = (13.0, 73.7, 134.4)
CBAR_PAD_MM, CBAR_W_MM = 1.5, 2.5

LINE_W_MM, LINE_H_MM = 47.3, 33.0
LINE_BOTTOM_MM = 12.0
LINE_LEFT_MM = (13.0, 73.3, 133.6)

LABEL_X = -0.03

LW_CONTOUR, LW_TRACE, LW_ROI = 0.9, 1.0, 0.7

FIELD_XTICKS = (0, 50, 100)
FIELD_YTICKS = (-50, 0, 50)

# Radial extent of panel f): the beam is a few mm wide, so showing the full
# 64 mm half-domain would compress every curve into the first eighth.
RADIAL_XMAX_MM = 15.0

DARK_LEGEND = dict(
    facecolor="0.25", edgecolor="0.4", labelcolor="white", framealpha=0.85
)

# One style per method, shared by panels d), e) and f).  Colour *and* dash
# differ so the curves stay separable in greyscale print.
STYLE = {
    "target": dict(color="k", ls="-", label="Free-Field"),
    "rbf": dict(color="C3", ls="-", label="RBF"),
    "spline": dict(color="C0", ls="--", label="Spline"),
    "multispline": dict(color="C2", ls="-.", label=f"{N_SPLINES}-Spline"),
}
PANEL_TITLE = {
    "rbf": "RBF",
    "spline": "Spline",
    "multispline": f"{N_SPLINES}-Spline",
}


def _apply_style() -> None:
    """Uniform 8 pt Arial, and text kept as text in the vector outputs."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # "path" (the default) converts every glyph to outlines, which
            # makes the text in the SVG uneditable.
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
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3,
        }
    )


def _rect(left_mm, bottom_mm, w_mm, h_mm):
    """Millimetre rectangle -> matplotlib figure-fraction rectangle."""
    return [left_mm / FIG_W_MM, bottom_mm / FIG_H_MM, w_mm / FIG_W_MM, h_mm / FIG_H_MM]


def _mirror(field: np.ndarray) -> np.ndarray:
    """Reflect an (Nx, Nr) half-plane field about the axis for display."""
    return np.concatenate([field[:, 1:][:, ::-1], field], axis=1)


def _panel_label(ax, letter: str) -> None:
    ax.text(
        LABEL_X, 1.02, f"{letter})", transform=ax.transAxes, ha="right", va="bottom"
    )


def _roi_slice(d: dict) -> np.ndarray:
    """Boolean mask over the axial axis selecting the ROI's x extent."""
    centre, size = d["roi_x_mm"]
    return np.abs(d["x"] - centre) <= size / 2


def _radial_profile(d: dict, field: np.ndarray) -> np.ndarray:
    """Pressure vs radius, averaged over the ROI's axial extent."""
    return field[_roi_slice(d)].mean(axis=0)


def _on_axis_at_focus(d: dict, field: np.ndarray) -> float:
    """On-axis pressure at the ROI centre (x = 60 mm), the single station a
    hydrophone reads when it is aligned on the focus.

    Distinct from the ROI mean, which averages 100 cells over a 5 x 2 mm box:
    this is one point, so it is what an experimental measurement at that
    position compares against directly.
    """
    i = int(np.argmin(np.abs(d["x"] - d["roi_x_mm"][0])))
    return float(field[i, 0])


def _draw_field(fig, ax, d, field, mask, cbar_label, left_mm, vmax):
    """Field panel: pressure, lens outline, hologram source and the ROI box."""
    from matplotlib.patches import Rectangle

    x, r = d["x"], d["r"]
    full_r = np.concatenate([-r[1:][::-1], r])
    extent = [x[0], x[-1], full_r[-1], full_r[0]]

    im = ax.imshow(
        _mirror(field).T / 1e6,
        extent=extent,
        cmap="magma",
        origin="upper",
        aspect="auto",  # the box is already square; this keeps data filling it
        vmin=0.0,
        vmax=vmax,
    )
    for sign in (1, -1):
        ax.contour(
            x, sign * r, d["source_binary"].T, levels=[0.5],
            colors="cyan", linewidths=LW_CONTOUR,
        )
        ax.contour(
            x, sign * r, mask.T, levels=[0.5],
            colors="white", linewidths=LW_CONTOUR,
        )

    cx, sx = d["roi_x_mm"]
    _, sr = d["roi_r_mm"]
    ax.add_patch(
        Rectangle(
            (cx - sx / 2, -sr / 2), sx, sr,
            fill=False, edgecolor="lime", linewidth=LW_ROI,
        )
    )

    ax.set_xlabel("Axial Position [mm]")
    ax.set_ylabel("Radial Position [mm]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks(FIELD_XTICKS)
    ax.set_yticks(FIELD_YTICKS)

    cax = fig.add_axes(
        _rect(left_mm + FIELD_S_MM + CBAR_PAD_MM, FIELD_BOTTOM_MM, CBAR_W_MM, FIELD_S_MM)
    )
    cbar = fig.colorbar(im, cax=cax, label=cbar_label)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, size=2.5, pad=2.0)
    return im


def make_figure(
    d: dict,
    readout: str,
    methods: list[str],
    vmax: float | None = None,
    source_model: str = "hologram",
) -> None:
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    cbar_label = READOUT_CBAR[readout]
    x = d["x"]

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))

    # One colour scale across a), b) and c) so the three lenses are directly
    # comparable; rounded up to a tidy tick value.  Override with --vmax if a
    # hot spot at the source ends up washing out the focus.
    if vmax is None:
        peak = max(float(d[f"final_field_{m}"].max()) for m in methods) / 1e6
        vmax = float(np.ceil(peak * 10) / 10)

    # -- a), b), c) optimised geometries + fields ---------------------------
    for slot, method in enumerate(methods):
        ax = fig.add_axes(
            _rect(FIELD_LEFT_MM[slot], FIELD_BOTTOM_MM, FIELD_S_MM, FIELD_S_MM)
        )
        _draw_field(
            fig, ax, d,
            d[f"final_field_{method}"],
            d[f"final_mask_{method}"],
            cbar_label,
            FIELD_LEFT_MM[slot],
            vmax,
        )
        ax.set_title(PANEL_TITLE[method])
        _panel_label(ax, "abc"[slot])
        if slot == 0:
            ax.legend(
                handles=[
                    Line2D(
                        [], [], color="cyan", lw=LW_CONTOUR,
                        label="Bowl" if source_model == "bowl" else "Hologram",
                    ),
                    Line2D([], [], color="lime", lw=LW_ROI, label="ROI"),
                ],
                loc="lower right",
                **DARK_LEGEND,
            )

    def line_axes(slot):
        return fig.add_axes(
            _rect(LINE_LEFT_MM[slot], LINE_BOTTOM_MM, LINE_W_MM, LINE_H_MM)
        )

    # -- d) on-axis pressure profiles ---------------------------------------
    ax_d = line_axes(0)
    ax_d.plot(x, d["target_field"][:, 0] / 1e6, **STYLE["target"])
    for method in methods:
        ax_d.plot(x, d[f"final_field_{method}"][:, 0] / 1e6, **STYLE[method])
    cx, sx = d["roi_x_mm"]
    for edge in (cx - sx / 2, cx + sx / 2):
        ax_d.axvline(edge, color="lime", ls=":", lw=LW_ROI)
    ax_d.set_xlabel("Axial Position [mm]")
    ax_d.set_ylabel(cbar_label)
    ax_d.set_xlim(x[0], x[-1])
    ax_d.set_ylim(0, None)
    ax_d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper left")
    _panel_label(ax_d, "d")

    # -- e) ROI pressure vs optimisation step -------------------------------
    ax_e = line_axes(1)
    free_roi = float(np.sum(d["target_field"] * d["roi_mask"]) / d["roi_mask"].sum())
    ax_e.axhline(free_roi / 1e6, **{**STYLE["target"], "lw": 0.7, "ls": ":"})
    n_steps = 0
    for method in methods:
        hist = d[f"roi_history_{method}"] / 1e6
        ax_e.plot(np.arange(len(hist)), hist, **STYLE[method])
        n_steps = max(n_steps, len(hist) - 1)
    ax_e.set_xlabel("Optimisation Step")
    # Short y-labels in e) and f): at 8 pt the full "Mean ROI Pressure
    # Amplitude [MPa]" is wider than the 33 mm panel is tall.
    ax_e.set_ylabel("Mean ROI Pressure [MPa]")
    ax_e.set_xlim(0, n_steps)
    ax_e.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_e.grid(True, alpha=0.3)
    _panel_label(ax_e, "e")

    # -- f) ROI-averaged radial profiles ------------------------------------
    ax_f = line_axes(2)
    r = d["r"]
    ax_f.plot(r, _radial_profile(d, d["target_field"]) / 1e6, **STYLE["target"])
    for method in methods:
        ax_f.plot(
            r, _radial_profile(d, d[f"final_field_{method}"]) / 1e6, **STYLE[method]
        )
    ax_f.axvline(d["roi_r_mm"][1] / 2, color="lime", ls=":", lw=LW_ROI)
    ax_f.set_xlabel("Radial Position [mm]")
    ax_f.set_ylabel("ROI-Averaged Pressure [MPa]")
    ax_f.set_xlim(0, RADIAL_XMAX_MM)
    ax_f.set_ylim(0, None)
    ax_f.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_f.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_f.grid(True, alpha=0.3)
    _panel_label(ax_f, "f")

    stem = output_stem(readout, source_model)
    for ext in ("pdf", "png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=600)
        print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Reporting and export
# ---------------------------------------------------------------------------
def _fwhm(r: np.ndarray, profile: np.ndarray) -> float:
    """-6 dB (half-amplitude) full width of an on-axis-peaked radial profile."""
    half = profile[0] / 2.0
    below = np.nonzero(profile < half)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    # linear interpolation between the last sample above and the first below
    frac = (profile[i - 1] - half) / (profile[i - 1] - profile[i])
    return 2.0 * float(r[i - 1] + frac * (r[i] - r[i - 1]))


def summarise(d: dict, methods: list[str]) -> None:
    roi = d["roi_mask"]
    free_roi = float(np.sum(d["target_field"] * roi) / roi.sum())
    free_pt = _on_axis_at_focus(d, d["target_field"])
    focus_mm = float(d["roi_x_mm"][0])
    free_w = _fwhm(d["r"], _radial_profile(d, d["target_field"]))

    print("\nSummary (ROI mean over 100 cells; p(x) on axis at the ROI centre)")
    print(
        f"  {'method':<14}{'dof':>5}{'ROI [MPa]':>11}{'gain':>8}"
        f"{f'p({focus_mm:g}mm)':>11}{'gain':>8}{'-6 dB width [mm]':>19}"
    )
    print(
        f"  {'free-field':<14}{'-':>5}{free_roi / 1e6:>11.4f}{'1.00x':>8}"
        f"{free_pt / 1e6:>11.4f}{'1.00x':>8}{free_w:>19.2f}"
    )
    for m in methods:
        val = float(d[f"roi_history_{m}"][-1])
        pt = _on_axis_at_focus(d, d[f"final_field_{m}"])
        width = _fwhm(d["r"], _radial_profile(d, d[f"final_field_{m}"]))
        n_dof = int(d[f"n_dof_{m}"]) if f"n_dof_{m}" in d else 0
        print(
            f"  {PANEL_TITLE[m]:<14}{n_dof:>5}{val / 1e6:>11.4f}"
            f"{val / free_roi:>7.2f}x{pt / 1e6:>11.4f}{pt / free_pt:>7.2f}x"
            f"{width:>19.2f}"
        )


def export_mat(
    d: dict, readout: str, methods: list[str], source_model: str = "hologram"
) -> None:
    """Masks and fields in a form the SolidWorks / MATLAB pipeline can read."""
    from scipy.io import savemat

    payload = {
        "x": d["x"] / 1e3,  # back to metres, as the other figure-3 .mat files
        "r": d["r"] / 1e3,
        "target_field": d["target_field"],
    }
    for m in methods:
        payload[f"final_mask_{m}"] = (d[f"final_mask_{m}"] > 0.5).astype(np.float32)
        payload[f"final_field_{m}"] = d[f"final_field_{m}"]
        payload[f"roi_history_{m}"] = d[f"roi_history_{m}"]
        if f"control_points_{m}" in d:
            payload[f"control_points_{m}"] = d[f"control_points_{m}"]
    path = OUT / f"{output_stem(readout, source_model)}.mat"
    savemat(str(path), payload)
    print(f"  wrote {path}")


def main() -> None:
    args = parse_args()
    cache = cache_path(args.readout, args.source)
    settings = resolve_settings(args)

    if args.replot:
        if not cache.exists():
            raise SystemExit(f"No cache at {cache}; run without --replot first.")
        d = dict(np.load(cache))
        print(f"Re-plotting from {cache}")
    else:
        d = dict(np.load(cache)) if cache.exists() else {}
        d.update(
            run(
                args.readout,
                args.steps,
                args.methods,
                settings,
                checkpoint=cache,
                seed=d,
                source_model=args.source,
                stride=args.save_history,
            )
        )
        np.savez_compressed(cache, **d)
        print(f"Cached run data to {cache}")

    # Only plot methods the cache actually holds, so a partial run still draws.
    methods = [m for m in METHODS if f"final_field_{m}" in d]
    missing = [m for m in METHODS if m not in methods]
    if missing:
        print(f"Note: no cached result for {', '.join(missing)}; panels shifted left.")

    summarise(d, methods)
    print("")
    make_figure(d, args.readout, methods, args.vmax, args.source)
    export_mat(d, args.readout, methods, args.source)


if __name__ == "__main__":
    main()
