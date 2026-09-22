"""
figure4/dome_sweep.py
=====================

Cone versus dome: how the focal field depends on the radius of curvature of the
coupling shape's wall.

The shape family has fixed endpoints and one free parameter.  Its inner surface
- the acoustically relevant boundary - is a circular arc from

* the C-103's base, ``(z, r) = (0, 38.73) mm``, taken from ``data/C103cone.mat``,
* to an opening at the C-103's tip plane, ``z = 37.24 mm``, widened to an
  aperture diameter of 28.8 mm, i.e. ``r = 14.4 mm``.

The arc bows *away* from the axis, so the cavity bellies out and the beam gets
more clearance than a straight wall would give it.  Its radius of curvature is
the swept parameter, and the **cone is the limiting member at infinite radius**,
where the arc degenerates to the straight chord.

Sampling is uniform in curvature ``1/R`` rather than in ``R``, in steps of
2.5 m^-1 from the cone up to ``1/--roc-min``.  That makes the cone an ordinary
member of the family at ``1/R = 0`` instead of an unreachable limit, and spaces
the geometries evenly by how much they actually deviate from it: the deviation
from the chord is proportional to curvature to first order.  At the 26.67 mm
minimum the wall stands 14.3 mm further from the axis than the cone's at
mid-length.

Two limits sit below that minimum.  The hard geometric floor is a semicircle at
``chord/2 = 22.25`` mm.  The binding one is 26.57 mm: below it the arc's tangent
at the opening passes vertical, so the wall overshoots the aperture plane and
folds back into a re-entrant lip instead of staying a monotonic funnel.  The
default stops just above that; ``--roc-min`` will go lower if you want the
folded shapes.

The wall is ``--thickness`` thick measured *normal* to the arc, so its radial
cross-section is thicker where the wall is steeply inclined: a uniform 1.20 mm
along the cone, and 1.00 mm at the base widening to 2.55 mm at the opening for
the 40 mm dome.  The opening is capped by the plane perpendicular to the inner
surface there, not by a flat ``z = z2`` cut, which would shave the wall to a
wedge exactly at the aperture.  The wall therefore reaches a little past the
aperture plane - by ``thickness`` times the z-component of the outward normal,
0.55 mm for the cone rising to 0.96 mm at the tightest radius - while the inner
surface, and so the acoustic aperture, still ends on ``z2``.  The focal analysis
box is pinned to that shared aperture plane so it is identical for every member.
Thickness is held constant across the
family so that shape, not wall mass, is the variable.  1 mm matches the shells
``figure3``'s Bezier and spline optimisations use, which makes this family
directly comparable to the paper's learned geometries; note it is only
0.12 wavelengths of the wall material at 0.3 MHz (0.14 for the 1.20 mm radial
cross-section along the cone), so the wall is partly transmissive - as those
learned shells also are.

Relation to the k-Wave arc-cone references
------------------------------------------
This follows ``SimulateH113arccone.m`` / ``SimulateH113arcconeSmall.m``, which
build the same kind of wall with ``makeArcEnds.m`` - two fixed endpoints, a
radius of curvature, and ``Inf`` degenerating to a straight line.  Registration
matches theirs: ``startpoint(1) = axialposmax`` puts the wall's base on the bowl
**rim** plane, which is also the C-103 base and the measured holography plane,
and their ``endpoint(2)`` is at r = 14.4 mm - the 28.8 mm aperture used here.

Three deliberate differences:

* the endpoints are *derived* from ``data/C103cone.mat`` rather than typed in
  (the references hard-code r = 38.4 mm and an opening 47 mm from the bowl apex,
  i.e. 38.3 mm from the rim, against 38.73 mm and 37.24 mm here);
* the radius is swept on a schedule rather than picked by hand (the references
  use ROC in {Inf, 120, 60, 40, 30} mm);
* thickness is measured normal to the arc, where the references extend the wall
  a fixed 11 voxels (2.2 mm) *radially* outward, which thins the effective shell
  where the wall is steeply inclined.

The wall material, ``--wall-c`` / ``--wall-rho``, is Formlabs Clear V4 at the
approximate properties this paper standardises on (2500 m/s, 1200 kg/m^3) - the
dome is a printed part, so it takes the print resin.  That is neither the
commercial C-103's polycarbonate shell (2270 m/s, 1200 kg/m^3) that
``frequency_sweep`` uses for the as-built comparison, nor the 2750 m/s /
1190 kg/m^3 resin of the arc-cone k-Wave references.

Domain and the ripple on the axial profiles
-------------------------------------------
This runs on a larger domain than figure4's, with a 20 mm absorbing layer
(4 wavelengths at 0.3 MHz) rather than 8 mm, because the axial profiles are read
much closer to the downstream boundary than any of figure4's metrics are.
Comparing the R = 40 mm dome's on-axis profile across three domains over their
common region, against a 32 mm layer as reference:

    layer    profile vs 32 mm      ripple rms
     8 mm     2.36 % rel L2         1.06 %
    20 mm     0.91 %                1.05 %
    32 mm     -                     1.06 %

So thickening the layer is worth it for overall accuracy - 8 mm carries a 2.4 %
error, 20 mm under 1 % - but it does **not** touch the ripple, which sits at
1.06 % rms whatever the boundary does.  That ripple is physical: interference
between the direct beam and the waves diffracted from the aperture lip and the
bowl edge.  It is not a numerical artefact and no amount of extra domain will
remove it.  Plotting a subset of the family rather than all sixteen profiles is
what makes it legible.

``--pml``, ``--z-min``, ``--z-max`` and ``--r-max`` set this domain; they patch
the corresponding module globals in ``frequency_sweep``, leaving figure4's own
defaults alone.

Metrics come from :func:`frequency_sweep.measure`, unchanged, so the numbers are
directly comparable with figure4's.  The reference for every member is the free
field, and the question the sweep asks is how close the field gets to it as the
wall curves.

The source is the **geometric bowl**, not the measured holography plane: this is
a design study, so an idealised source keeps the wall's own effect from being
mixed with the measured plane's aberration, and it carries no source-sampling
limit.  Fields are therefore pressure gains relative to the unit-amplitude shell
rather than absolute pascals.  ``--source hologram`` switches back.  The drive
frequency is fixed at 0.3 MHz.

Run with::

    python figure4/dome_sweep.py --dry-run
    python figure4/dome_sweep.py
    python figure4/dome_sweep.py --plot-only
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np

# The grid, source, focal box and metric definitions all come from figure 3's
# sweep, so the two analyses stay directly comparable and there is one
# definition of `measure` rather than two that can drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figure3_frequency_sweep"))

import frequency_sweep as fs  # noqa: E402

# Opening of the shape: same axial plane as the C-103 tip, wider aperture.
APERTURE_DIAM = 28.8e-3

# Wall thickness, normal to the inner surface.  Matches figure3's shells.
THICKNESS = 1.0e-3

# Wall material: Formlabs Clear V4 at the approximate properties used for every
# printed part in this paper.  The arc-cone k-Wave references
# (SimulateH113arccone*.m) used 2750 m/s / 1190 kg/m^3 for their own resin; the
# dome here is printed in Clear V4, so it takes Clear V4's numbers.
WALL_C, WALL_RHO = 2500.0, 1200.0

# Tightest arc in the family.  Two limits sit below this.  The hard geometric
# floor is a semicircle at chord/2 = 22.25 mm.  The more useful one is 26.57 mm:
# below that the arc's tangent at the opening turns past vertical, so the wall
# overshoots the aperture plane and folds back on itself into a re-entrant lip
# rather than staying a monotonic funnel.  26.67 mm sits just above that, and
# gives a curvature range 50 % wider than the 40 mm this started at.
ROC_MIN = 1.0 / 37.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def cone_endpoints(data_dir: str) -> tuple[float, float]:
    """``(z_tip, r_base)`` of the C-103's inner surface [m], from its own mask.

    Read from the ``.mat`` rather than hard-coded so the dome family stays
    pinned to the real cone if that file is ever regenerated.
    """
    from scipy.io import loadmat

    data = loadmat(str(Path(data_dir) / "C103cone.mat"))
    mask = np.asarray(data["C103array2D"]) > 0
    x = np.asarray(data["x_vec"]).squeeze().astype(float)
    y = np.asarray(data["y_vec"]).squeeze().astype(float)

    occupied = np.flatnonzero(mask.any(axis=1))
    z_tip = float(x[occupied.max()])

    half = y >= 0.0
    base = np.flatnonzero(mask[occupied.min()][half])
    r_base = float(y[half][base.min()])
    return z_tip, r_base


def roc_schedule(roc_min: float, intervals: int) -> list[float]:
    """Radii of curvature, evenly spaced in curvature, cone (``inf``) first."""
    kappa = np.linspace(0.0, 1.0 / roc_min, intervals + 1)
    return [np.inf if k == 0.0 else 1.0 / k for k in kappa]


def arc_wall_mask(X, R, dx: float, roc: float, p1, p2, thickness: float):
    """Soft occupancy mask for a wall whose inner surface is a circular arc.

    Args:
        X, R:      Grid coordinate arrays [m].
        dx:        Cell size, used as the width of the antialiasing ramp [m].
        roc:       Radius of curvature [m]; ``np.inf`` gives the straight cone.
        p1, p2:    ``(z, r)`` endpoints of the *inner* surface [m].
        thickness: Wall thickness normal to that surface [m].

    Returns:
        ``(Nx, Nr)`` array in [0, 1].  Edges are ramped over one cell, which
        matches the soft edge the C-103 mask acquires from being interpolated
        onto the grid, so neither geometry is unfairly staircased.
    """
    z1, r1 = p1
    z2, r2 = p2
    dz, dr = z2 - z1, r2 - r1
    chord = float(np.hypot(dz, dr))

    # Unit normal to the chord, pointing away from the axis.
    nz, nr = -dr / chord, dz / chord
    if nr < 0.0:
        nz, nr = -nz, -nr

    # Outward unit normal at the opening, and the forward tangent there.  The
    # opening is capped by the plane through p2 perpendicular to that tangent,
    # rather than by a flat z = z2 cut: a flat cut shaves a wedge off the wall
    # where the surface is steeply inclined, leaving it thinner than
    # `thickness` exactly at the aperture.  Capping perpendicular instead keeps
    # full thickness right to the lip and lets the wall reach slightly past z2,
    # while the *inner* surface - the acoustic aperture - still ends on z2.
    if np.isinf(roc):
        # Signed distance to the chord line; positive on the far-from-axis side.
        signed = (X - z1) * nz + (R - r1) * nr
        tz, tr = dz / chord, dr / chord
    else:
        if roc < chord / 2.0:
            raise ValueError(
                f"radius of curvature {1e3 * roc:.2f} mm cannot span a "
                f"{1e3 * chord:.2f} mm chord (minimum {1e3 * chord / 2:.2f} mm)"
            )
        # Centre sits on the near-axis side, so the arc bows away from it.
        h = float(np.sqrt(roc**2 - (chord / 2.0) ** 2))
        cz = 0.5 * (z1 + z2) - h * nz
        cr = 0.5 * (r1 + r2) - h * nr
        signed = np.hypot(X - cz, R - cr) - roc
        # Tangent at p2 is perpendicular to the radius there; orient it forward.
        ez, er = (z2 - cz) / roc, (r2 - cr) / roc
        tz, tr = -er, ez
        if tz * dz + tr * dr < 0.0:
            tz, tr = -tz, -tr

    def ramp(v):
        return np.clip(v / dx + 0.5, 0.0, 1.0)

    return np.minimum.reduce(
        [
            ramp(signed),               # inside the inner face
            ramp(thickness - signed),   # inside the outer face
            ramp(X - z1),               # downstream of the flat base plane
            ramp(-((X - z2) * tz + (R - r2) * tr)),  # inside the capped opening
        ]
    ).astype(np.float32)


def inner_profile(roc: float, p1, p2, n: int = 200):
    """Sampled ``(z, r)`` of the inner surface, for drawing the family."""
    z1, r1 = p1
    z2, r2 = p2
    dz, dr = z2 - z1, r2 - r1
    chord = float(np.hypot(dz, dr))
    nz, nr = -dr / chord, dz / chord
    if nr < 0.0:
        nz, nr = -nz, -nr

    if np.isinf(roc):
        t = np.linspace(0.0, 1.0, n)
        return z1 + t * dz, r1 + t * dr

    h = float(np.sqrt(roc**2 - (chord / 2.0) ** 2))
    cz = 0.5 * (z1 + z2) - h * nz
    cr = 0.5 * (r1 + r2) - h * nr
    a1 = np.arctan2(r1 - cr, z1 - cz)
    a2 = np.arctan2(r2 - cr, z2 - cz)
    a = np.linspace(a1, a2, n)
    return cz + roc * np.cos(a), cr + roc * np.sin(a)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def label_of(roc: float) -> str:
    return "cone" if np.isinf(roc) else f"{1e3 * roc:.4g}mm"


def cache_path(roc: float, args) -> Path:
    name = (
        f"dome_{label_of(roc)}_{args.source}_f{round(args.freq / 1e3):04d}kHz"
        f"_ppw{args.ppw:g}_cfl{args.cfl:g}_tf{args.transit_factor:g}"
        f"_t{1e3 * args.thickness:g}mm_ap{1e3 * args.aperture:g}mm"
        f"_c{args.wall_c:g}r{args.wall_rho:g}_cap2"
        f"_dom{1e3 * args.z_min:g}_{1e3 * args.z_max:g}_{1e3 * args.r_max:g}"
        f"_pml{1e3 * args.pml:g}.npz"
    )
    return Path(args.cache_dir) / name


def _base_record(case, args, p_free, p_cone, src_mask, geom_mask, seconds):
    return dict(
        freq=case.freq,
        dx=case.dx,
        dt=case.dt,
        Nx=case.Nx,
        Nr=case.Nr,
        Nt=case.Nt,
        pml=case.pml,
        ppw=args.ppw,
        cfl=args.cfl,
        source=args.source,
        peak_unit="gain" if args.source == "bowl" else "MPa",
        seconds=seconds,
        p_free=p_free,
        p_cone=p_cone,
        src_mask=src_mask,
        cone_mask=geom_mask,
    )


def run_sweep(args) -> list[dict]:
    """Simulate the free field once, then every member of the family."""
    import jax

    from jaxisymmetric import run_simulation_amplitude
    from jaxisymmetric.geometry import FixedGeometry

    z_tip, r_base = cone_endpoints(args.data_dir)
    p1 = (0.0, r_base)
    p2 = (z_tip, args.aperture / 2.0)

    case = fs.Case(args.freq, args)
    cfg, source, _ = fs.build_config(case, args)
    # One jit for every medium: the grid, step count and source never change.
    sim = jax.jit(
        lambda medium, c, s: run_simulation_amplitude(
            medium, c, s, record_cycles=args.record_cycles
        )
    )

    def run(medium):
        t0 = time.perf_counter()
        field = np.asarray(
            sim(medium, cfg, source).block_until_ready(), dtype=np.float32
        )
        return field, time.perf_counter() - t0

    print("  running free field...", flush=True)
    p_free, t_free = run(cfg.homogeneous_medium())
    src_mask = np.abs(np.asarray(source.mask)).astype(np.float32)
    # cfg.X runs 0..(Nx-1)*dx; p1/p2 are quoted in z, which has its origin on
    # the cone base plane.  Shift into z before building the wall - forgetting
    # this puts the whole family Z_MIN (20 mm) upstream of where it belongs.
    Z = np.asarray(cfg.X) + fs.Z_MIN
    Rg = np.asarray(cfg.R)

    def finish(geom_mask, p_cone, seconds, tag):
        rec = _base_record(case, args, p_free, p_cone, src_mask, geom_mask, seconds)
        fs.measure(rec, args)
        print(
            f"  {tag:>8}: L2 = {100 * float(rec['rel_l2']):6.2f} %,"
            f"  Linf = {100 * float(rec['rel_linf']):6.2f} %,"
            f"  {fs._fmt_peak(rec['cone_peak'], rec)},"
            f"  width {1e3 * float(rec['fwhm_env_cone']):5.2f} mm  ({seconds:.1f} s)",
            flush=True,
        )
        return rec

    results = []
    for roc in args.rocs:
        path = cache_path(roc, args)
        if path.exists() and not args.force:
            with np.load(path) as z:
                rec = {k: z[k] for k in z.files}
            fs.measure(rec, args)
            rec["roc"] = roc
            print(f"  {label_of(roc):>8}: cached")
            results.append(rec)
            continue
        if args.plot_only:
            raise FileNotFoundError(f"--plot-only but no cache for {label_of(roc)}: {path}")

        geom_mask = arc_wall_mask(Z, Rg, case.dx, roc, p1, p2, args.thickness)
        if geom_mask.sum() <= 0.0:
            raise RuntimeError(f"empty wall mask at RoC {label_of(roc)}")
        geom = FixedGeometry(c=args.wall_c, rho=args.wall_rho, mask=geom_mask)
        p_cone, t_g = run(geom.as_medium(cfg))
        rec = finish(geom_mask, p_cone, t_g, label_of(roc))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **rec)
        rec["roc"] = roc
        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _no_pml_window(res: dict):
    """Display window excluding the absorbing layer, as ``((z_lo, z_hi), r)`` mm."""
    dx = float(res["dx"])
    crop = int(res["pml"])
    Nx, Nr = int(res["Nx"]), int(res["Nr"])
    return (
        (1e3 * (fs.Z_MIN + crop * dx), 1e3 * (fs.Z_MIN + (Nx - crop - 1) * dx)),
        1e3 * (Nr - crop - 1) * dx,
    )


def make_figure(results: list[dict], args, out_dir: Path):
    import matplotlib as mpl

    if not args.show:
        mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z_tip, r_base = cone_endpoints(args.data_dir)
    p1 = (0.0, r_base)
    p2 = (z_tip, args.aperture / 2.0)

    rocs = np.array([float(r["roc"]) for r in results])
    kappa = np.where(np.isinf(rocs), 0.0, 1.0 / np.where(np.isinf(rocs), 1.0, rocs))
    order = np.argsort(kappa)
    kx = kappa[order]

    def curve(key, scale=1.0):
        return scale * np.array([float(r[key]) for r in results])[order]

    # The member that lands closest to the free field - not the most curved one,
    # since the approach to free field turns around before the family ends.
    i_best = int(np.argmin([float(r["rel_l2"]) for r in results]))

    # Publication layout: no titles beyond the panel letters, and the field map
    # cropped to the physical domain so the absorbing layer is out of shot.
    # That crop leaves it square, so column 0 is given enough width for the
    # panel's height - not its width - to be what the cell constrains, which is
    # what makes it match (b) and (c) in height.
    fig = plt.figure(figsize=(13.6, 4.3))
    # 1.45 rather than the ~1.38 that just fits: erring wide makes the panel
    # height-limited, so its height matches (b) and (c) exactly instead of
    # falling a percent short.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.34, top=0.93)
    cmap = plt.get_cmap("viridis")
    shade = kx / kx.max() if kx.max() > 0 else np.zeros_like(kx)

    # a) the field of the best member
    res = results[i_best]
    vmax_ref = float(res["free_peak"])
    best_peak = float(res["cone_peak"])
    ax_a = fig.add_subplot(gs[0, 0])
    fs._panel(
        fig, ax_a, res, "p_cone", "", vmax_ref,
        _no_pml_window(res), max(1.0, best_peak / vmax_ref),
    )
    ax_a.set_xlabel("Axial position from cone base [mm]")
    ax_a.set_title("a)", loc="left", fontsize=11)
    ax_a.legend(
        handles=[
            Line2D([], [], color="#ff7f0e", lw=1.1, ls=(0, (5, 3)),
                   label=r"$L_2$ / $L_\infty$ region"),
            # Grey, not the white it is on the map: white would vanish against
            # the legend's own light background.  Carries the radius, since
            # without a title nothing else identifies which member this is.
            Line2D([], [], color="0.35", lw=0.8,
                   label=f"dome wall, R = {1e3 * rocs[i_best]:.0f} mm"),
        ],
        loc="lower left", fontsize=7, framealpha=0.65, borderpad=0.4,
    )

    # b) on-axis profiles over exactly the metric region, one per curvature
    ax_b = fig.add_subplot(gs[0, 1])
    peak_scale = 1e-6 if str(results[0]["peak_unit"]) == "MPa" else 1.0
    unit = "MPa" if peak_scale != 1.0 else "gain"
    z_axis = fs._axes_vectors(results[0])[0]
    lo = int(np.searchsorted(z_axis, 1e3 * args.box_z_start))
    hi = int(np.searchsorted(z_axis, 1e3 * float(results[0]["box_z_hi"])))

    # A subset of the family: every --profile-stride'th member, with the cone,
    # the 40 mm dome and the tightest arc always in.  All 16 overplotted is
    # unreadable, and with a handful they can carry a legend instead of a
    # colour bar.
    ordered = [results[i] for i in order]
    ordered_roc = np.array([float(r["roc"]) for r in ordered])
    picks = set(range(0, len(ordered), max(1, args.profile_stride)))
    picks.add(len(ordered) - 1)
    picks.add(int(np.argmin(np.abs(ordered_roc - 40.0e-3))))

    for i in sorted(picks):
        roc_i = ordered_roc[i]
        label = (
            r"R = $\infty$ (cone)" if np.isinf(roc_i) else f"R = {1e3 * roc_i:.0f} mm"
        )
        ax_b.plot(
            z_axis[lo:hi],
            peak_scale * np.asarray(ordered[i]["p_cone"])[lo:hi, 0],
            color=cmap(0.88 * shade[i]), lw=1.5, label=label,
        )
    ax_b.plot(
        z_axis[lo:hi],
        peak_scale * np.asarray(results[0]["p_free"])[lo:hi, 0],
        "k--", lw=2.0, label="free field",
    )
    ax_b.set_xlabel("Axial position from cone base [mm]")
    ax_b.set_ylabel(f"on-axis pressure [{unit}]")
    ax_b.set_title("b)", loc="left", fontsize=11)
    ax_b.grid(alpha=0.3)
    ax_b.legend(frameon=True, framealpha=0.85, fontsize=8, loc="upper right")

    # c) every metric on one axis, as a percentage departure from the free field
    ax_c = fig.add_subplot(gs[0, 2])
    free_peak = float(results[0]["free_peak"])
    free_width = float(results[0]["fwhm_env_free"])
    ax_c.plot(kx, curve("rel_l2", 100), "o-", color="#1f77b4", label=r"relative $L_2$")
    ax_c.plot(kx, curve("rel_linf", 100), "s-", color="#d62728", label=r"relative $L_\infty$")
    ax_c.plot(
        kx, 100 * (curve("cone_peak") / free_peak - 1.0), "D-",
        color="#2ca02c", label="focal peak excess",
    )
    ax_c.plot(
        kx, 100 * (curve("fwhm_env_cone") / free_width - 1.0), "^-",
        color="#9467bd", label=r"focal $-6$ dB width excess",
    )
    ax_c.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax_c.set_ylabel("departure from free field [%]")
    ax_c.set_title("c)", loc="left", fontsize=11)
    ax_c.set_xlabel(r"wall curvature $1/R$ [m$^{-1}$]")
    ax_c.grid(alpha=0.3)
    ax_c.set_xlim(-0.02 * kx.max(), 1.02 * kx.max())

    # Headroom above the data for the legend, so it never sits on the L-infinity
    # curve - which is the highest of the four and the one it would collide with.
    stack = np.concatenate(
        [
            curve("rel_l2", 100),
            curve("rel_linf", 100),
            100 * (curve("cone_peak") / free_peak - 1.0),
            100 * (curve("fwhm_env_cone") / free_width - 1.0),
        ]
    )
    span = float(stack.max() - stack.min())
    ax_c.set_ylim(stack.min() - 0.08 * span, stack.max() + 0.44 * span)
    top = ax_c.secondary_xaxis("top")
    ticks = kx[:: max(1, len(kx) // 5)]
    top.set_xticks(ticks)
    top.set_xticklabels(
        [r"$\infty$" if t == 0 else f"{1e3 / t:.0f}" for t in ticks], fontsize=8
    )
    top.set_xlabel("radius of curvature R [mm]", fontsize=9)
    ax_c.legend(frameon=True, framealpha=0.85, fontsize=8, loc="upper right")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"dome_sweep_{args.source}_ppw{args.ppw:g}"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", format=ext, bbox_inches="tight", dpi=180)
    print(f"\nFigure written to {stem}.pdf / .svg / .png")
    if args.show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="C-103-derived cone/dome family swept over wall curvature."
    )
    # Geometric bowl by default: this family is a design study, so an idealised
    # source keeps the wall's own effect from being mixed with the measured
    # plane's aberration, and it carries no source-sampling limit.
    p.add_argument("--source", choices=("bowl", "hologram"), default="bowl")
    p.add_argument("--freq", type=float, default=fs.HOLOGRAM_FREQ)
    p.add_argument("--ppw", type=float, default=20.0)
    p.add_argument("--cfl", type=float, default=0.1)
    p.add_argument("--roc-min", type=float, default=ROC_MIN)
    # 15 intervals over 1/R = 0 .. 37.5 puts the samples 2.5 m^-1 apart.
    p.add_argument("--intervals", type=int, default=15)
    p.add_argument("--thickness", type=float, default=THICKNESS)
    p.add_argument("--aperture", type=float, default=APERTURE_DIAM)
    p.add_argument("--wall-c", type=float, default=WALL_C)
    p.add_argument("--wall-rho", type=float, default=WALL_RHO)
    p.add_argument("--ramp-cycles", type=float, default=3.0)
    p.add_argument("--settle-cycles", type=float, default=3.0)
    p.add_argument("--record-cycles", type=float, default=3.0)
    p.add_argument("--transit-factor", type=float, default=3.0)
    p.add_argument("--focal-radial-factor", type=float, default=2.0)
    # figure5 runs on a deliberately larger domain with a thicker absorbing
    # layer than figure4's.  Its axial profiles are read much closer to the
    # downstream boundary, where figure4's 8 mm layer - only 1.6 wavelengths at
    # 0.3 MHz - reflects enough to ring visibly on them.
    p.add_argument("--pml", type=float, default=20.0e-3)
    p.add_argument("--z-min", type=float, default=-32.0e-3)
    p.add_argument("--z-max", type=float, default=108.0e-3)
    p.add_argument("--r-max", type=float, default=70.0e-3)
    p.add_argument("--profile-stride", type=int, default=5,
                   help="show every Nth axial profile in panel (b)")
    p.add_argument("--data-dir", default=str(here.parent / "data"))
    p.add_argument("--out-dir", default=str(here))
    p.add_argument("--cache-dir", default=str(here / "sweep_cache"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--throughput", type=float, default=fs.DEFAULT_THROUGHPUT)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    # These are module globals in frequency_sweep, read at call time, so setting
    # them here reconfigures Case, build_config, measure and the display axes
    # consistently - and leaves figure4's own defaults untouched.
    fs.Z_MIN, fs.Z_MAX, fs.R_MAX = args.z_min, args.z_max, args.r_max
    fs.PML_METRES = args.pml

    args.rocs = roc_schedule(args.roc_min, args.intervals)
    # frequency_sweep.Case reads this to size the grid; only one frequency here.
    args.freqs = [args.freq]
    # Pin the focal box to the shared aperture plane rather than to each wall's
    # outermost cell, which the perpendicular end cap pushes past the aperture
    # by a curvature-dependent amount.  Read by frequency_sweep.measure.
    args.box_z_start = cone_endpoints(args.data_dir)[0]

    chord = float(
        np.hypot(args.box_z_start, cone_endpoints(args.data_dir)[1] - args.aperture / 2.0)
    )
    if args.roc_min < chord / 2.0:
        raise SystemExit(
            f"--roc-min {1e3 * args.roc_min:.2f} mm is below the geometric floor "
            f"of chord/2 = {1e3 * chord / 2:.2f} mm"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    z_tip, r_base = cone_endpoints(args.data_dir)
    chord = float(np.hypot(z_tip, r_base - args.aperture / 2.0))
    case = fs.Case(args.freq, args)

    print(
        f"Cone/dome curvature sweep - source '{args.source}', "
        f"{args.freq / 1e6:.2f} MHz, {args.ppw:g} PPW, CFL {args.cfl:g}"
    )
    print(
        f"inner surface from (z=0, r={1e3 * r_base:.2f}) mm to "
        f"(z={1e3 * z_tip:.2f}, r={1e3 * args.aperture / 2:.2f}) mm, chord "
        f"{1e3 * chord:.2f} mm; z = 0 is the bowl rim plane"
    )
    print(
        f"wall {1e3 * args.thickness:g} mm normal, c = {args.wall_c:g} m/s, "
        f"rho = {args.wall_rho:g} kg/m3; focus at z = "
        f"{1e3 * fs.CONE_BASE_TO_FOCUS:.2f} mm"
    )
    print(
        f"grid {case.Nx} x {case.Nr}, dx = {case.dx * 1e6:.1f} um, Nt = {case.Nt}, "
        f"{len(args.rocs)} geometries + free field"
    )
    # `lip` is how far the perpendicular end cap carries the wall past the
    # aperture plane: thickness times the z-component of the outward normal at
    # the opening, which grows as the wall there steepens.
    print("\n      R/mm | 1/R m^-1 | bulge/mm |  lip/mm")
    print("  " + "-" * 44)
    for roc in args.rocs:
        # Perpendicular sagitta, converted to the radial gap between the arc and
        # the chord at fixed z - which is what "further from the axis than the
        # cone" means.  Both are largest at the arc midpoint, where the arc's
        # tangent runs parallel to the chord, so the conversion is exact.
        sag = 0.0 if np.isinf(roc) else roc - np.sqrt(roc**2 - (chord / 2) ** 2)
        radial = sag * chord / z_tip
        dz, dr = z_tip, args.aperture / 2.0 - r_base
        if np.isinf(roc):
            nz = -dr / chord
        else:
            h = np.sqrt(roc**2 - (chord / 2) ** 2)
            cz = 0.5 * z_tip - h * (-dr / chord)
            nz = (z_tip - cz) / roc
        print(
            f"  {'inf' if np.isinf(roc) else f'{1e3 * roc:8.2f}':>8} |"
            f" {0.0 if np.isinf(roc) else 1.0 / roc:8.4f} | {1e3 * radial:8.2f} |"
            f" {1e3 * args.thickness * abs(nz):8.2f}"
        )
    est = (len(args.rocs) + 1) * case.Nt * case.work_per_step / args.throughput
    print("  " + "-" * 44)
    print(f"  estimated total: {est / 60:.1f} min at {args.throughput / 1e6:.1f} Mpt/s\n")

    if args.dry_run:
        return

    results = run_sweep(args)

    print("\n      R/mm |  rel L2 % | rel Linf % | focal peak | width/mm | peak ratio")
    print("  " + "-" * 76)
    scale = 1e-6 if str(results[0]["peak_unit"]) == "MPa" else 1.0
    for rec in results:
        roc = float(rec["roc"])
        print(
            f"  {'inf' if np.isinf(roc) else f'{1e3 * roc:8.2f}':>8} |"
            f" {100 * float(rec['rel_l2']):9.2f} | {100 * float(rec['rel_linf']):10.2f} |"
            f" {scale * float(rec['cone_peak']):10.3f} |"
            f" {1e3 * float(rec['fwhm_env_cone']):8.2f} |"
            f" {float(rec['peak_ratio']):10.3f}"
        )
    print(
        f"  free field: {scale * float(results[0]['free_peak']):.3f}, "
        f"width {1e3 * float(results[0]['fwhm_env_free']):.2f} mm"
    )

    make_figure(results, args, Path(args.out_dir))


if __name__ == "__main__":
    main()
