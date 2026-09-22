"""
figure4/frequency_sweep.py
==========================

Frequency dependence of the focal aberration introduced by the C-103 coupling
cone on the H-117 transducer.

The sweep drives the *same physical transducer* at every frequency from 0.2 to
1.2 MHz and, at each one, runs two simulations that differ only in the medium:

* **free field** - the H-117 radiating into water,
* **C-103** - the same source with the measured C-103 cone in the beam path.

Both are read out with :func:`jaxisymmetric.run_simulation_amplitude`, i.e. the
steady-state CW amplitude at the drive frequency recovered by a lock-in DFT over
the final ``--record-cycles`` periods.  The difference between the two fields,
over the focal region defined below, is the aberration the cone introduces; its
relative L2 and L-infinity norms are plotted against frequency, alongside what
the cone does to the focus itself - its peak amplitude and its -6 dB width, both
as a ratio to the free field.  The two say different things: the peak ratio is
how much the cone concentrates or attenuates, the width ratio is how much it
squeezes or broadens the beam.

Analysis region
---------------
The metrics are taken over the *focal* region, not the whole domain, and it is
drawn on every field panel as a dashed box:

* axially, from the cone's own tip to the inner edge of the absorbing layer -
  the same start plane in the free-field case, so the two fields are compared
  over identical ground;
* radially, out to ``--focal-radial-factor`` times the free-field radial FWHM
  on the free-field focal plane (default 2, so the box spans +/-2 FWHM; pass 1
  for a box whose total width is 2 FWHM).

Sizing the box from the free field keeps it a property of the reference rather
than of the thing being measured, so the cone cannot move its own goalposts.
Because the box shrinks with frequency as the focus does, the reported errors
are relative to a focal region of comparable size at every frequency rather
than to a fixed volume of mostly-empty water.

The metrics depend on this region but the simulated fields do not, so
:func:`measure` re-derives them from the cache on every load - changing
``--focal-radial-factor`` is free and needs no re-simulation.

Source models
-------------
``--source`` selects between two ways of driving the same transducer, and
running both is the point - they have complementary weaknesses.

``hologram`` (default) starts from the **measured** H-117 plane in
``data/rprofile_FF.mat``, the same acoustic-holography input that
``figure2/c103_comparison_steady.py`` uses, so the amplitude apodisation and the
phase aberration are experimental rather than idealised.  Carrying it to another
frequency needs care: the measured phase is not simply ``omega * delay``, so it
is decomposed into a fitted geometric delay, which is re-scaled, plus a residual
that is not - see :func:`load_hologram`.  At 0.3 MHz the original measurement is
reproduced exactly.  Its limitation is sampling: the plane is measured every
0.5 mm, which is only 2.5 points per wavelength at 1.2 MHz, so above ~0.75 MHz
the *source* rather than the grid sets the accuracy (the script warns).

``bowl`` uses the nominal focused shell (``make_focused_bowl_source``, RoC
63.2 mm, OD 64 mm, ID 22.6 mm).  It is idealised - uniform amplitude, no
measured aberration - but it is exact at every frequency and has no sampling
limit, so it is the cross-check for the hologram at the top of the band.

Registration
------------
``z = 0`` is the C-103 base plane, where ``x = 0`` of ``data/C103cone.mat`` sits.
That plane is also the measured holography plane **and** the bowl rim, so both
source models drive the identical geometry.

Note that this is *not* 45 mm in front of the focus, which is how the
``(Nx/2)*dx - 45e-3`` in the figure2 scripts reads at first glance - that
expression only positions the plane inside its domain.  Three independent checks
put the focus 54.5 mm downstream of the plane instead:

* the bowl's own rim-to-focus distance, ``sqrt(RoC^2 - (OD/2)^2) = 54.50`` mm;
* an amplitude-weighted spherical fit to the measured phase, 55.07 mm;
* simulating the unmodified measured plane at 0.3 MHz, whose free-field peak
  lands 54.0 mm downstream (the profile is flat to within 1 % over 48-64 mm).

The cone is 37.24 mm long, so its tip sits 17.3 mm short of the focus with a
bore radius of 8.4 mm, against a geometric beam radius of 10.1 mm there - the
tip clips the outer rays, which is the aberration mechanism this figure probes.

Validation
----------
The bowl free field was checked against the independent k-Wave reference in
``data/H117_FreeFieldkWave_steady.mat``, which also uses ``x = 0`` at the rim
(so ``z = x_kwave``).  At 0.3 MHz, over the overlap of the two domains excluding
the source and the absorbing layer: peak gain -1.5 %, relative L2 3.5 %,
relative L-infinity 4.4 % at 20 PPW (-0.8 %, 2.5 %, 3.5 % at 10 PPW).  The
residual does not shrink with resolution because it is not a discretisation
error in the field but a difference in how the two codes discretise the *bowl* -
``make_focused_bowl_source`` lays down a hard shell one cell thick, which is a
physically thinner source as the grid refines.

The bowl peak gain tracks the analytic focal gain of a spherical cap,
``2*pi*(R - sqrt(R^2 - a^2))/lambda`` minus the same for the central hole, i.e.
32.2 per MHz.  At 20 PPW the simulated ratio runs 34.8 per MHz at 0.2 MHz down
to 32.3 by 1.2 MHz, converging on the analytic value as the aperture covers more
wavelengths.

The hologram arm was checked against ``data/EXPH117_C103_kWave_steady.mat``, the
k-Wave field for this transducer *with* the cone, driven from the same measured
plane and gridded with ``x = 0`` on it.  At 0.3 MHz and 20 PPW the two solvers
put the focal peak 0.4 mm apart (41.2 vs 41.6 mm) and agree to 7.4 % L2 /
11.7 % L-infinity, improving from 8.1 % / 15.1 % at 10 PPW.  Both overshoot the
1.020 MPa measured in the experiment - this code by +5.5 %, k-Wave by +6.9 % -
so the overshoot is common to the two simulations rather than numerical, and is
more likely a modelling gap (no absorption, idealised cone material) than a
solver error.  The closer +2.7 % this code gave at 10 PPW was partly
under-resolution flattering the peak.

Resolution convergence: going from 10 to 20 PPW moves the reported relative L2
by at most 1.7 percentage points and L-infinity by at most 4.9, for both source
models, so 10 PPW is already close to converged and 20 PPW is a refinement
rather than a correction.

The absorbing layer was checked at 0.2 MHz, where its 8 mm is thinnest in
wavelengths (1.10 lambda): re-running on a domain extended 60 mm further
changes the metric region by 2.0 % L2 and 2.5 % L-infinity, an order of
magnitude below the cone effect being measured.

Resolution
----------
The grid holds a constant ``--ppw`` points per wavelength in water, so ``dx``
scales as ``1/f`` while the physical domain and the geometry stay fixed.  Cost
therefore grows as ``f^3`` (two grid dimensions and the time step).  Run
``--dry-run`` first: it prints the grid, step count and a wall-clock estimate
for every frequency without simulating anything.

Memory is not the binding constraint.  The worst case, 1.2 MHz at 20 PPW, is a
1792 x 1024 grid and peaks at 473 MB resident - the solver scans in place and
keeps only O(1) fields.  Time is what hurts: that same case is roughly 64 000
steps, and the whole 0.2-1.2 MHz sweep at 20 PPW came to an estimated 9.6 h on
one CPU (about 40 M point-updates/s).  A GPU is the practical way to run it.

One inefficiency is worth knowing about when choosing ``--ppw``: the solver pads
every spectral derivative to the next power of two, so a grid just over 1024
cells costs the same as one of 2048.  The ``pad`` column of the ``--dry-run``
table shows the resulting waste - at 20 PPW it is 3.35x at 0.7 MHz but only
1.14x at 1.2 MHz.

Steady state
------------
``--transit-factor`` sets how many domain diagonals of propagation time elapse
before the recording window opens.  It has to cover more than the first arrival:
the source is a *hard* boundary, so sound reverberates between the bowl and the
cone for several round trips.  Sweeping it at 10 PPW showed the metrics settle
by 3.0 and stay flat out to 6.0::

    transit factor   0.2 MHz L2   0.6 MHz L2   0.6 MHz Linf
             1.5        34.66 %      21.94 %        32.17 %
             2.0        34.90 %      22.09 %        32.22 %
             3.0        35.07 %      22.33 %        32.87 %
             4.0        35.00 %      22.36 %        33.00 %
             6.0        35.00 %      22.36 %        32.97 %

Hence the default of 3.0.  Dropping to 1.5 halves the run time and costs about
2 % on the reported errors, which is a reasonable trade for exploratory runs.

Every frequency is cached to ``sweep_cache/*.npz`` as soon as it finishes, so an
interrupted sweep resumes where it stopped and the figure can be redrawn with
``--plot-only``.

The cache key includes ``--source``, ``--ppw``, ``--cfl`` and
``--transit-factor``, so those have to match when redrawing.

Run with::

    python figure4/frequency_sweep.py --dry-run                     # cost table
    python figure4/frequency_sweep.py --ppw 10                      # hologram
    python figure4/frequency_sweep.py --ppw 10 --source bowl        # cross-check
    python figure4/frequency_sweep.py --ppw 10 --plot-only          # redraw
    python figure4/source_model_comparison.py --ppw 10              # both, overlaid
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical setup - all lengths in metres.
# ---------------------------------------------------------------------------

# H-117 nominal geometry (Sonic Concepts), as used by figure2/freefield_*.py.
RADIUS_CURV = 63.2e-3
OUTER_DIAM = 64.0e-3
INNER_DIAM = 22.6e-3

# Axial distance from the bowl's rim - its plane of greatest aperture radius -
# to the geometric focus: 54.50 mm.  Everything is registered to that plane; see
# the Registration note in the module docstring.
RIM_TO_FOCUS = float(np.sqrt(RADIUS_CURV**2 - (OUTER_DIAM / 2) ** 2))

# z = 0 is the C-103 base plane, which coincides with the bowl rim and with the
# measured holography plane, so both source models drive the same geometry.
CONE_BASE_TO_FOCUS = RIM_TO_FOCUS

# Measured holography plane (data/rprofile_FF.mat): acquisition frequency and
# the radius the reference scripts truncate the profile at.
HOLOGRAM_FREQ = 0.3e6
HOLOGRAM_RPOS = 37.0e-3

# Water background and C-103 material (matching figure2/c103_comparison_steady.py).
C0, RHO0 = 1500.0, 1000.0
CONE_C, CONE_RHO = 2270.0, 1200.0

# Domain, quoted relative to the cone base plane (z = 0).  Fixed in metres so
# that every frequency sees the identical physical problem; only dx changes.
# Z_MIN clears the back of the bowl (its inner edge is at z = -7.7 mm) and
# R_MAX clears the cone's outer wall (radius 49.1 mm) plus the absorbing layer.
# Z_MAX leaves ~28 mm of usable field beyond the focus at z = +54.5 mm.
Z_MIN = -20.0e-3
Z_MAX = 90.0e-3
R_MAX = 64.0e-3

# Absorbing-layer thickness in metres rather than grid points, so it stays
# adequate at the longest wavelength (8 mm is 1.07 lambda at 0.2 MHz) instead of
# thinning out as the grid refines.
PML_METRES = 8.0e-3
MIN_PML_POINTS = 8

# Measured on the CPU this was developed on: padded point-updates per second.
# Only used by --dry-run to estimate wall-clock; pass --throughput to retune.
DEFAULT_THROUGHPUT = 39.8e6

# Radial half-extent of the field panels [mm].  Display only - the metrics are
# unaffected.  Cropping to 50 mm still clears the cone's outer wall (49.1 mm)
# and leaves the maps close enough to square that, at equal aspect, they match
# the line plots sharing their row.
DISPLAY_R_MAX = 50.0



def next_fft_size(n: int) -> int:
    """Padded transform length used by ``jaxisymmetric.utils`` (next power of 2)."""
    return 2 ** ((n - 1).bit_length())


# ---------------------------------------------------------------------------
# Measured holography plane
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def load_hologram(data_dir: str):
    """Load the measured H-117 plane and split its phase into delay + residual.

    ``data/rprofile_FF.mat`` is a complex radial profile measured at 0.3 MHz.
    Its phase cannot simply be scaled by ``f / f0``: the plane is a *diffracted*
    field, so it carries a nodal ring (here at r = 6.25 mm, where the amplitude
    drops to ~1 % of peak) across which the phase steps by ~0.78 pi.  A phase
    reversal at a null is a topological feature of the field, not a propagation
    delay - scaling it by 4 turns it into ~3.1 pi, which both aliases the 0.5 mm
    sampling and physically erases the reversal.

    So the phase is decomposed as

        phi_0(r) = omega_0 * tau(r) + psi(r),
        tau(r)   = (sqrt(d^2 + r^2) - d) / c0

    with ``d`` fitted to the measurement, amplitude-weighted so the reliable
    high-amplitude annulus sets it and the nulls do not.  Only the geometric
    delay ``tau`` is re-scaled with frequency; the residual ``psi`` - the
    measured aberration and the nodal structure - is carried unchanged.  At
    ``f = f0`` this reproduces the measured profile exactly.

    Returns:
        ``(r_centers, amp, tau, psi, d_fit)``, all numpy arrays but ``d_fit``.
    """
    from scipy.io import loadmat
    from scipy.optimize import least_squares

    data = loadmat(str(Path(data_dir) / "rprofile_FF.mat"))
    r = np.asarray(data["r_centers"]).squeeze().astype(float)
    prof = np.asarray(data["radial_prof"]).squeeze()
    amp = np.abs(prof)
    phi0 = np.unwrap(np.angle(prof))
    w0 = 2 * np.pi * HOLOGRAM_FREQ

    inside = r <= HOLOGRAM_RPOS
    weight = amp[inside] / amp[inside].max()

    def model(p, rr):
        return w0 * (np.hypot(p[0], rr) - p[0]) / C0 + p[1]

    fit = least_squares(
        lambda p: weight * (model(p, r[inside]) - phi0[inside]),
        x0=[RIM_TO_FOCUS, 0.0],
    )
    d_fit = float(fit.x[0])

    tau = (np.hypot(d_fit, r) - d_fit) / C0
    psi = phi0 - w0 * tau  # absorbs the fitted constant offset, which is global
    return r, amp, tau, psi, d_fit


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


class Case:
    """Everything that defines one frequency's simulation, before it is run."""

    def __init__(self, freq: float, args):
        self.freq = float(freq)
        self.dx = C0 / (self.freq * args.ppw)
        self.Nx = int(np.ceil((Z_MAX - Z_MIN) / self.dx))
        self.Nr = int(np.ceil(R_MAX / self.dx))
        self.pml = max(MIN_PML_POINTS, int(np.ceil(PML_METRES / self.dx)))
        self.dt = args.cfl * self.dx / C0

        # Enough time for the wave to cross the domain (and reverberate between
        # the hard source and the cone) before the recording window opens.
        diag = float(np.hypot(self.Nx * self.dx, self.Nr * self.dx))
        t_transit = args.transit_factor * diag / C0
        t_cycles = (args.ramp_cycles + args.settle_cycles + args.record_cycles) / self.freq
        self.Nt = int(np.ceil((t_transit + t_cycles) / self.dt))

        # Cost proxy: the solver pads every spectral derivative to a power of
        # two, so the padded shape - not the stored shape - sets the run time.
        self.fft_x = next_fft_size(self.Nx)
        self.fft_r = next_fft_size(2 * self.Nr - 1)
        self.work_per_step = self.fft_x * self.fft_r / 2.0

    def est_seconds(self, throughput: float) -> float:
        """Estimated wall-clock for *both* runs (free field and cone) [s]."""
        return 2.0 * self.Nt * self.work_per_step / throughput

    def describe(self, throughput: float) -> str:
        est = self.est_seconds(throughput)
        pad = self.work_per_step / (self.Nx * self.Nr)
        return (
            f"{self.freq / 1e6:5.2f} | {self.dx * 1e6:7.1f} | {self.Nx:5d} x {self.Nr:4d} |"
            f" {self.pml:4d} | {self.Nt:7d} | {self.fft_x:5d} x {self.fft_r:5d} |"
            f" {pad:5.2f} | {est / 60:8.1f}"
        )


def _bowl_mask(cfg, case):
    """Nominal H-117 shell, with its rim on the cone base plane."""
    from jaxisymmetric.sources import make_focused_bowl_source

    mask = make_focused_bowl_source(
        cfg,
        focus_pos=-Z_MIN + CONE_BASE_TO_FOCUS,
        radius_curvature=RADIUS_CURV,
        outer_diameter=OUTER_DIAM,
        inner_diameter=INNER_DIAM,
    )
    if float(np.sum(np.asarray(mask))) < 1.0:
        raise RuntimeError(f"empty bowl source mask at {case.freq / 1e6:.2f} MHz")
    return mask


def _hologram_mask(cfg, case, args):
    """Measured H-117 plane, re-phased to ``case.freq`` (see :func:`load_hologram`).

    This mirrors ``jaxisymmetric.sources.make_holography_source`` but takes an
    already-unwrapped phase instead of a complex profile.  That library function
    re-derives the phase with ``unwrap(angle(prof))``, and once the geometric
    term is scaled up the largest sample-to-sample step reaches ~2.7 rad - under
    pi, but close enough that a round trip through wrapping is a hazard not
    worth taking when the unwrapped phase is already in hand.
    """
    import jax.numpy as jnp

    r_centers, amp, tau, psi, _ = load_hologram(args.data_dir)
    phase = 2 * np.pi * case.freq * tau + psi

    step = float(np.max(np.abs(np.diff(phase[r_centers <= HOLOGRAM_RPOS]))))
    if step >= np.pi:
        raise RuntimeError(
            f"re-phased hologram aliases its own 0.5 mm sampling at "
            f"{case.freq / 1e6:.2f} MHz (max step {step:.2f} rad >= pi)"
        )

    plane_x = -Z_MIN
    axial = jnp.abs(cfg.X - plane_x) < (cfg.dx / 2)
    radial = cfg.R <= HOLOGRAM_RPOS
    a = jnp.interp(cfg.R, jnp.asarray(r_centers), jnp.asarray(amp))
    p = jnp.interp(cfg.R, jnp.asarray(r_centers), jnp.asarray(phase))
    # Nearest-neighbour on the axis, as make_holography_source does.
    a = a.at[:, 0].set(a[:, 1])
    p = p.at[:, 0].set(p[:, 1])

    mask = (axial & radial) * a * jnp.exp(1j * p)
    if float(np.sum(np.abs(np.asarray(mask)))) <= 0.0:
        raise RuntimeError(f"empty hologram source mask at {case.freq / 1e6:.2f} MHz")
    return mask


def build_config(case: Case, args):
    """Build the ``SimConfig``, source and cone geometry for one frequency."""
    from jaxisymmetric import SimConfig, Source
    from jaxisymmetric.geometry import FixedGeometry

    cfg = SimConfig(
        Nx=case.Nx,
        Nr=case.Nr,
        dx=case.dx,
        dr=case.dx,
        c0=C0,
        rho0=RHO0,
        cfl=args.cfl,
        pml_width=case.pml,
        Nt=case.Nt,
    )

    # Grid x = 0 is z = Z_MIN, so the cone base (z = 0) is at grid x = -Z_MIN.
    cone_base_x = -Z_MIN

    if args.source == "bowl":
        src_mask = _bowl_mask(cfg, case)
    else:
        src_mask = _hologram_mask(cfg, case, args)

    ramp_steps = max(1, int(round(args.ramp_cycles / case.freq / case.dt)))
    source = Source(mask=src_mask, freq=case.freq, ramp_steps=ramp_steps)

    cone = FixedGeometry.from_mat(
        cfg,
        mat_file_path=str(Path(args.data_dir) / "C103cone.mat"),
        c=CONE_C,
        rho=CONE_RHO,
        mask_key="C103array2D",
        x_key="x_vec",
        y_key="y_vec",
        source_zpos=cone_base_x,
    )
    if float(np.sum(np.asarray(cone.mask))) < 1.0:
        raise RuntimeError(f"empty cone mask at {case.freq / 1e6:.2f} MHz")

    return cfg, source, cone


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _radial_fwhm(prof: np.ndarray, dx: float) -> float:
    """Full width at half maximum of a radial profile that peaks on the axis.

    The half-maximum crossing is interpolated linearly between the two
    straddling samples, so the result is not quantised to the grid.  Returns
    twice the domain extent if the profile never falls to half.
    """
    half = 0.5 * prof[0]
    below = np.flatnonzero(prof < half)
    if prof[0] <= 0.0 or not below.size or below[0] == 0:
        return 2.0 * (prof.size - 1) * dx
    j = int(below[0])
    frac = (prof[j - 1] - half) / (prof[j - 1] - prof[j])
    return 2.0 * (j - 1 + frac) * dx


def measure(res: dict, args) -> dict:
    """Derive the focal analysis region and the cone-induced error metrics.

    The region is the *focal* one, not the whole domain:

    * axially it starts at the cone's own tip - the same plane in the free-field
      case, so both fields are compared over identical ground - and runs to the
      inner edge of the absorbing layer;
    * radially it extends to ``--focal-radial-factor`` times the free-field
      radial FWHM, measured on the free-field focal plane.

    Sizing the box from the *free field* matters: it makes the region a property
    of the reference rather than of the thing being measured, so the cone cannot
    move its own goalposts.  Any source or cone cells falling inside are still
    excluded, though with the box starting at the tip there are at most a few in
    the first column.

    This works purely from the stored fields, so it is re-derived every time a
    cached sweep is loaded and the metrics always match the current definition.
    """
    dx = float(res["dx"])
    crop = int(res["pml"])
    Nx, Nr = int(res["Nx"]), int(res["Nr"])
    p_free = np.asarray(res["p_free"], dtype=float)
    p_cone = np.asarray(res["p_cone"], dtype=float)
    src = np.abs(np.asarray(res["src_mask"], dtype=float))
    cone = np.asarray(res["cone_mask"], dtype=float)

    z = np.arange(Nx) * dx + Z_MIN
    r = np.arange(Nr) * dx
    z_hi_i = Nx - crop - 1
    r_hi_i = Nr - crop - 1

    # Axial start: the cone tip, as the grid actually resolves it.  An optional
    # ``args.box_z_start`` overrides it, for callers whose geometries end on a
    # common aperture plane but whose outermost wall cell does not - figure5's
    # dome family, where the wall is capped perpendicular to its own surface and
    # so reaches a little past the aperture by a curvature-dependent amount.
    box_start = getattr(args, "box_z_start", None)
    if box_start is not None:
        tip_i = int(np.searchsorted(z, float(box_start)))
    else:
        cols = np.flatnonzero((cone > 0.5).any(axis=1))
        tip_i = int(cols.max()) if cols.size else int(np.searchsorted(z, 0.0))

    # Free-field focal plane, searched downstream of that tip.
    axis = p_free[:, 0]
    foc_i = tip_i + int(np.argmax(axis[tip_i : z_hi_i + 1]))

    fwhm = _radial_fwhm(p_free[foc_i, : r_hi_i + 1], dx)
    r_box_i = int(min(r_hi_i, max(1, np.ceil(args.focal_radial_factor * fwhm / dx))))

    # Focal-width ratio.  Not measured at each field's own peak plane: with the
    # cone in place the on-axis profile is multi-lobed, so that plane hops
    # between lobes and the width jumps with it (0.881 -> 0.553 between 1.0 and
    # 1.1 MHz for the bowl).  Take the -6 dB width of the focal region's
    # envelope, max_z |p(z, r)|, which does not care which lobe wins and is
    # smooth across the band.  Both fields use it, so the ratio is consistent.
    # For the free field it exceeds the focal-plane FWHM above by 14 % at
    # 0.2 MHz, where the focus is long enough that off-plane contributions
    # broaden the envelope, falling below 2 % from 0.5 MHz up.
    band = (slice(tip_i, z_hi_i + 1), slice(0, r_hi_i + 1))
    fwhm_env_free = _radial_fwhm(p_free[band].max(axis=0), dx)
    fwhm_env_cone = _radial_fwhm(p_cone[band].max(axis=0), dx)

    sl = (slice(tip_i, z_hi_i + 1), slice(0, r_box_i + 1))
    keep = (src[sl] <= 0.0) & (cone[sl] <= 1e-6)

    res.update(compute_metrics(p_free[sl], p_cone[sl], keep, r[: r_box_i + 1]))
    res.update(
        box_z_lo=z[tip_i],
        box_z_hi=z[z_hi_i],
        box_r=r[r_box_i],
        fwhm=fwhm,
        z_focus=z[foc_i],
        fwhm_env_free=fwhm_env_free,
        fwhm_env_cone=fwhm_env_cone,
        peak_ratio=float(res["cone_peak"]) / float(res["free_peak"]),
        fwhm_ratio=fwhm_env_cone / fwhm_env_free,
    )
    return res


def compute_metrics(p_free, p_cone, keep, r_half):
    """Relative L2 / L-infinity difference between the cone and free-field fields.

    Args:
        p_free: Free-field steady-state amplitude on the cropped half-plane.
        p_cone: Same with the cone present.
        keep:   Boolean mask of water points (outside cone and source).
        r_half: Radial coordinate of each column [m], for the volume weighting.

    Returns:
        Dict of relative errors.  ``rel_l2``/``rel_linf`` are unweighted norms
        over the half-plane image, matching the convention already used by
        ``figure2/*_comparison_steady.py``.  ``rel_l2_vol`` weights each cell by
        ``r``, which is the true L2 over the revolved 3-D volume.
    """
    d = np.where(keep, p_cone - p_free, 0.0)
    f = np.where(keep, p_free, 0.0)
    w = np.broadcast_to(r_half[None, :], f.shape)

    return {
        "rel_l2": float(np.sqrt(np.sum(d**2) / np.sum(f**2))),
        "rel_linf": float(np.max(np.abs(d)) / np.max(np.abs(f))),
        "rel_l2_vol": float(np.sqrt(np.sum(w * d**2) / np.sum(w * f**2))),
        "free_peak": float(np.max(f)),
        "cone_peak": float(np.max(np.where(keep, p_cone, 0.0))),
    }


# ---------------------------------------------------------------------------
# One frequency
# ---------------------------------------------------------------------------


def _fmt_peak(value, res) -> str:
    """Format a peak amplitude in whichever units the source model implies."""
    if str(res["peak_unit"]) == "MPa":
        return f"peak {float(value) / 1e6:.3f} MPa"
    return f"peak gain {float(value):.1f}"


def cache_path(case: Case, args) -> Path:
    name = (
        f"sweep_{args.source}_f{round(case.freq / 1e3):04d}kHz_ppw{args.ppw:g}"
        f"_cfl{args.cfl:g}_tf{args.transit_factor:g}.npz"
    )
    return Path(args.cache_dir) / name


def run_case(case: Case, args) -> dict:
    """Run the free-field and cone simulations at one frequency."""
    import jax

    from jaxisymmetric import run_simulation_amplitude

    cfg, source, cone = build_config(case, args)
    # cfg and source are passed as arguments rather than captured, so the grid
    # and coordinate arrays they carry stay runtime inputs instead of being
    # constant-folded into the executable (~22 MB of them at 1.2 MHz, 20 PPW).
    sim = jax.jit(
        lambda medium, c, s: run_simulation_amplitude(
            medium, c, s, record_cycles=args.record_cycles
        )
    )

    t0 = time.perf_counter()
    p_free = np.asarray(
        sim(cfg.homogeneous_medium(), cfg, source).block_until_ready(), dtype=np.float32
    )
    t_free = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_cone = np.asarray(
        sim(cone.as_medium(cfg), cfg, source).block_until_ready(), dtype=np.float32
    )
    t_cone = time.perf_counter() - t0

    src_mask = np.abs(np.asarray(source.mask)).astype(np.float32)
    cone_mask = np.asarray(cone.mask, dtype=np.float32)

    out = dict(
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
        # The bowl is a unit-pressure shell, so its fields are dimensionless
        # gains; the hologram carries the measured amplitude, so its fields are
        # absolute pressures in Pa.
        peak_unit="gain" if args.source == "bowl" else "MPa",
        seconds=t_free + t_cone,
        p_free=p_free,
        p_cone=p_cone,
        src_mask=src_mask,
        cone_mask=cone_mask,
    )

    return out


def _report(res: dict, note: str) -> None:
    print(
        f"  {float(res['freq']) / 1e6:.2f} MHz: L2 = {100 * float(res['rel_l2']):6.2f} %,"
        f"  Linf = {100 * float(res['rel_linf']):6.2f} %,"
        f"  {_fmt_peak(res['free_peak'], res)} -> {_fmt_peak(res['cone_peak'], res)},"
        f"  FWHM {1e3 * float(res['fwhm']):5.2f} mm  ({note})",
        flush=True,
    )


def load_or_run(case: Case, args) -> dict:
    """Get one frequency's fields, from cache if possible, and measure them.

    The metrics depend on the analysis region but the *fields* do not, so they
    are always re-derived here rather than trusted from the cache.  Changing
    ``--focal-radial-factor`` therefore costs nothing: the sweep need not be
    re-simulated.
    """
    path = cache_path(case, args)
    if path.exists() and not args.force:
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
        _report(measure(out, args), "cached")
        return out
    if args.plot_only:
        raise FileNotFoundError(f"--plot-only but no cache for {case.freq / 1e6:.2f} MHz: {path}")

    out = run_case(case, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    _report(measure(out, args), f"{float(out['seconds']):.1f} s")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _mirror(field: np.ndarray) -> np.ndarray:
    """Reflect a half-plane field about r = 0 for display."""
    return np.concatenate([field[:, 1:][:, ::-1], field], axis=1)


def _axes_vectors(res: dict):
    """Physical axes [mm] of the full mirrored field, absorbing layer included."""
    dx = float(res["dx"])
    x = np.arange(int(res["Nx"])) * dx + Z_MIN
    r = np.arange(int(res["Nr"])) * dx
    r_full = np.concatenate([-r[1:][::-1], r])
    return 1e3 * x, 1e3 * r_full


def _analysis_box(res: dict):
    """The region the metrics are taken over, as ``((z_lo, z_hi), r_lim)`` in mm.

    Read straight back from what :func:`measure` sliced, so the drawn box and
    the reported numbers cannot drift apart.
    """
    return (
        (1e3 * float(res["box_z_lo"]), 1e3 * float(res["box_z_hi"])),
        1e3 * float(res["box_r"]),
    )


def _display_window(results):
    """Largest z/r window every panel can fill, in mm.

    The grid rounds up to a whole number of cells, so the domain overshoots
    ``Z_MAX``/``R_MAX`` by up to one cell and by a different amount at each
    frequency.  Taking the intersection of the extents gives all panels one
    identical window with no blank margin on the coarser grid.
    """
    axes = [_axes_vectors(res) for res in results]
    z_lo = max(float(x[0]) for x, _ in axes)
    z_hi = min(float(x[-1]) for x, _ in axes)
    r_lim = min(min(-float(r[0]), float(r[-1])) for _, r in axes)
    return (z_lo, z_hi), min(r_lim, DISPLAY_R_MAX)


def _panel(fig, ax, res, field_key, label, vmax_ref, window, vmax_disp):
    """Draw one field map, scaled to that frequency's free-field peak.

    Both panels of a column are divided by the same free-field peak, so 1.0
    always means "the focal pressure this transducer reaches in open water at
    this frequency" and the cone map can be read straight against the free-field
    map above it.  ``vmax_disp`` is stretched past 1.0 when the cone's peak
    exceeds the free field's, which it does at low frequency - clipping the
    brightest part of the focus would hide exactly what the panel is for.
    """
    from matplotlib.patches import Rectangle

    field = _mirror(np.asarray(res[field_key])) / vmax_ref

    outline = np.abs(np.asarray(res["src_mask"]))
    if field_key == "p_cone":
        outline = outline + np.asarray(res["cone_mask"])
    outline = _mirror(outline)

    x_vec, r_vec = _axes_vectors(res)
    im = ax.imshow(
        field.T,
        extent=[x_vec[0], x_vec[-1], r_vec[0], r_vec[-1]],
        origin="lower",
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_disp,
    )
    ax.contour(x_vec, r_vec, outline.T, levels=[0.5], colors="w", linewidths=0.8)

    # The metric region: everything but the absorbing layer, and within that,
    # everything but the source and cone cells the white contours enclose.
    (bz_lo, bz_hi), br = _analysis_box(res)
    box = Rectangle(
        (bz_lo, -br),
        bz_hi - bz_lo,
        2 * br,
        fill=False,
        edgecolor="#ff7f0e",
        linewidth=1.1,
        linestyle=(0, (5, 3)),
    )
    ax.add_patch(box)

    (z_lo, z_hi), r_lim = window
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Radial position [mm]")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("|p| / free-field peak", fontsize=8)
    return im


def make_figure(results: list[dict], args, out_dir: Path):
    import matplotlib as mpl

    if not args.show:
        mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    freqs = np.array([float(r["freq"]) for r in results])
    panel_res = []
    for want in args.panel_freqs:
        i = int(np.argmin(np.abs(freqs - want)))
        if abs(freqs[i] - want) > 1.0:
            print(
                f"NOTE: {want / 1e6:.2f} MHz is not in the sweep; "
                f"plotting the nearest, {freqs[i] / 1e6:.2f} MHz."
            )
        panel_res.append(results[i])

    window = _display_window(panel_res)

    # Two rows of three: free-field maps then cone maps, with the two line plots
    # filling the right-hand column so each row reads left to right as
    # "what the field looks like" then "what it does across the band".
    fig = plt.figure(figsize=(17.0, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.33)
    field_tags = (("a", "b"), ("d", "e"))

    for col, res in enumerate(panel_res):
        f_mhz = float(res["freq"]) / 1e6
        vmax_ref = float(res["free_peak"])
        cone_peak = float(res["cone_peak"])
        # Column-wise, so the tighter high-frequency focus keeps its contrast
        # instead of being dimmed by the low-frequency column's larger range.
        vmax_disp = max(1.0, cone_peak / vmax_ref)
        # The widths quoted here are the same envelope measure panel (f) ratios,
        # so dividing one caption by the other reproduces its purple curve.
        rows = (
            (
                0,
                "p_free",
                "Free field",
                _fmt_peak(vmax_ref, res),
                float(res["fwhm_env_free"]),
            ),
            (
                1,
                "p_cone",
                "With C-103 cone",
                f"{_fmt_peak(cone_peak, res)} "
                f"({100 * cone_peak / vmax_ref:.0f} % of free field)",
                float(res["fwhm_env_cone"]),
            ),
        )
        for row, key, what, sub, width in rows:
            ax = fig.add_subplot(gs[row, col])
            _panel(
                fig,
                ax,
                res,
                key,
                f"{field_tags[row][col]}) {what} - {f_mhz:.1f} MHz\n{sub}\n"
                f"focal $-6$ dB width {1e3 * width:.2f} mm",
                vmax_ref,
                window,
                vmax_disp,
            )
            if row == 1:
                ax.set_xlabel("Axial position from cone base [mm]")
            else:
                ax.tick_params(labelbottom=False)
            if col:
                # Same radial axis as the column to its left; dropping the
                # repeat also clears that column's colour-bar label.
                ax.set_ylabel("")

    order = np.argsort(freqs)
    fx = freqs[order] / 1e6

    def curve(res_key, scale=1.0):
        return scale * np.array([float(r[res_key]) for r in results])[order]

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(fx, curve("rel_l2", 100), "o-", color="#1f77b4", label=r"relative $L_2$")
    ax_c.plot(fx, curve("rel_linf", 100), "s-", color="#d62728", label=r"relative $L_\infty$")
    ax_c.set_ylabel("cone-induced field error [%]")
    ax_c.set_title("c) Aberration over the boxed focal region", fontsize=10)

    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(fx, curve("peak_ratio"), "o-", color="#2ca02c", label="focal peak")
    ax_f.plot(fx, curve("fwhm_ratio"), "^-", color="#9467bd", label=r"focal $-6$ dB width")
    ax_f.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax_f.set_ylabel("cone / free field")
    ax_f.set_title("f) What the cone does to the focus", fontsize=10)

    for ax in (ax_c, ax_f):
        for res in panel_res:
            ax.axvline(float(res["freq"]) / 1e6, color="0.6", ls=":", lw=1.0, zorder=0)
        ax.set_xlabel("Source frequency [MHz]")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(freqs.min() / 1e6 - 0.05, freqs.max() / 1e6 + 0.05)

    which = (
        "nominal focused bowl"
        if args.source == "bowl"
        else "measured holography plane, re-phased"
    )
    fig.suptitle(
        "H-117 with the C-103 coupling cone: frequency dependence of the focal aberration\n"
        f"source: {which}   ({args.ppw:g} points per wavelength, CFL {args.cfl:g})",
        fontsize=12,
        y=0.998,
    )
    # One key for all four maps - the box hugs the domain edge, so an in-axes
    # legend would sit on top of it whichever corner it went in.
    fig.legend(
        handles=[
            Line2D([], [], color="#ff7f0e", lw=1.1, ls=(0, (5, 3)),
                   label=r"region the $L_2$ / $L_\infty$ metrics are taken over"),
            Line2D([], [], color="0.35", lw=0.8, label="source and cone, excluded from it"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"frequency_sweep_{args.source}_ppw{args.ppw:g}"
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
        description="C-103 cone aberration versus H-117 drive frequency."
    )
    p.add_argument(
        "--source",
        choices=("bowl", "hologram"),
        default="hologram",
        help="source model: 'hologram' re-phases the measured H-117 plane "
        "(experimentally grounded, default), 'bowl' uses the nominal shell",
    )
    p.add_argument("--ppw", type=float, default=20.0, help="points per wavelength in water")
    p.add_argument("--cfl", type=float, default=0.1, help="CFL number (dt = cfl*dx/c0)")
    p.add_argument("--fmin", type=float, default=0.2e6)
    p.add_argument("--fmax", type=float, default=1.2e6)
    p.add_argument("--df", type=float, default=0.1e6)
    p.add_argument("--freqs", type=float, nargs="+", default=None, help="explicit frequency list")
    p.add_argument(
        "--panel-freqs",
        type=float,
        nargs=2,
        default=[0.2e6, 1.0e6],
        help="the two frequencies shown as 2-D field maps",
    )
    p.add_argument(
        "--focal-radial-factor",
        type=float,
        default=2.0,
        help="radial half-extent of the metric region, in multiples of the "
        "free-field focal FWHM (default 2, i.e. the box spans +/-2 FWHM)",
    )
    p.add_argument("--ramp-cycles", type=float, default=3.0)
    p.add_argument("--settle-cycles", type=float, default=3.0)
    p.add_argument("--record-cycles", type=float, default=3.0)
    p.add_argument(
        "--transit-factor",
        type=float,
        default=3.0,
        help="domain diagonals of propagation time before the recording window; "
        "3.0 is converged, 1.5 halves the cost for about 2 %% error",
    )
    p.add_argument("--data-dir", default=str(here.parent / "data"))
    p.add_argument("--out-dir", default=str(here))
    p.add_argument("--cache-dir", default=str(here / "sweep_cache"))
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--plot-only", action="store_true", help="draw from cache, simulate nothing")
    p.add_argument("--dry-run", action="store_true", help="print the cost table and stop")
    p.add_argument("--throughput", type=float, default=DEFAULT_THROUGHPUT)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    if args.freqs is None:
        n = int(round((args.fmax - args.fmin) / args.df)) + 1
        args.freqs = [args.fmin + i * args.df for i in range(n)]
    return args


def main(argv=None):
    args = parse_args(argv)
    cases = [Case(f, args) for f in args.freqs]

    print(
        f"C-103 frequency sweep - source '{args.source}', {args.ppw:g} points per "
        f"wavelength, CFL {args.cfl:g}"
    )
    print(
        f"domain z in [{Z_MIN * 1e3:.0f}, {Z_MAX * 1e3:.0f}] mm, r <= {R_MAX * 1e3:.0f} mm, "
        f"absorbing layer {PML_METRES * 1e3:.0f} mm; cone base at z = 0, "
        f"focus at z = {CONE_BASE_TO_FOCUS * 1e3:.2f} mm"
    )
    if args.source == "hologram":
        r_centers, _, _, _, d_fit = load_hologram(args.data_dir)
        dr_src = float(np.diff(r_centers)[0])
        print(
            f"hologram: measured at {HOLOGRAM_FREQ / 1e6:.1f} MHz, {dr_src * 1e3:.2f} mm "
            f"sampling, fitted convergence distance {d_fit * 1e3:.2f} mm "
            f"(bowl rim-to-focus {RIM_TO_FOCUS * 1e3:.2f} mm)"
        )
        thin = [f for f in args.freqs if C0 / f / dr_src < 4.0]
        if thin:
            print(
                f"  WARNING: the measured plane is sampled at under 4 points per "
                f"wavelength above {min(thin) / 1e6:.2f} MHz "
                f"({C0 / max(args.freqs) / dr_src:.1f} pts/lambda at "
                f"{max(args.freqs) / 1e6:.2f} MHz) - the source itself, not the grid, "
                f"is the accuracy limit there.  Cross-check with --source bowl."
            )
    print(
        "\n  f/MHz |   dx/um |     grid     |  pml |      Nt |  padded FFT   |"
        "   pad |  est/min"
    )
    print("  " + "-" * 82)
    for c in cases:
        print("  " + c.describe(args.throughput))
    total = sum(c.est_seconds(args.throughput) for c in cases)
    print("  " + "-" * 82)
    print(f"  estimated total: {total / 3600:.2f} h at {args.throughput / 1e6:.1f} Mpt/s\n")

    if args.dry_run:
        return

    results = [load_or_run(c, args) for c in cases]

    mpa = str(results[0]["peak_unit"]) == "MPa"
    scale, head = (1e-6, "free/MPa | cone/MPa") if mpa else (1.0, "free gain | cone gain")
    print(f"\n  f/MHz |  rel L2 % | rel Linf % | rel L2 (vol) % | {head} | peak ratio"
          " | width free/cone mm | width ratio")
    print("  " + "-" * 124)
    for r in results:
        free = scale * float(r["free_peak"])
        cone = scale * float(r["cone_peak"])
        print(
            f"  {float(r['freq']) / 1e6:5.2f} | {100 * float(r['rel_l2']):9.2f} |"
            f" {100 * float(r['rel_linf']):10.2f} | {100 * float(r['rel_l2_vol']):14.2f} |"
            f" {free:8.3f} | {cone:8.3f} | {float(r['peak_ratio']):10.3f}"
            f" | {1e3 * float(r['fwhm_env_free']):8.2f} /{1e3 * float(r['fwhm_env_cone']):8.2f}"
            f" | {float(r['fwhm_ratio']):11.3f}"
        )
    print(
        f"  metric region: z from the cone tip "
        f"({1e3 * float(results[0]['box_z_lo']):.2f} mm) to "
        f"{1e3 * float(results[0]['box_z_hi']):.1f} mm, "
        f"|r| <= {args.focal_radial_factor:g} x FWHM"
    )

    make_figure(results, args, Path(args.out_dir))


if __name__ == "__main__":
    main()
