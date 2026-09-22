"""
paper/figure2/freefield_comparison_steady.py
============================================

Steady-state-amplitude variant of ``freefield_comparison.py``.

The original script compares the JAXisymmetric *peak* pressure (``max|p|`` over
all time) against k-Wave for the H-117 transducer in free field.  Peak pressure
is biased by the transient ramp-up/overshoot, so it is not a like-for-like match
to a steady-state CW experiment.  This version instead extracts the
**steady-state pressure amplitude** at the drive frequency over the final few
cycles, via :func:`jaxisymmetric.run_simulation_amplitude` (a single-frequency
lock-in / FFT-bin estimate accumulated online).

  a) JAXisymmetric steady-state pressure amplitude (gain)
  b) k-Wave steady-state pressure amplitude (gain)
  c) Absolute difference (%)

k-Wave reference: this script prefers ``H117_FreeFieldkWave_steady.mat``
(variable ``p_amp_full``) produced by ``H117comparisonJAXisymmetric.m``, which
extracts the k-Wave steady-state amplitude with the *same* lock-in DFT — giving
a true steady-vs-steady comparison.  If that file is not present yet it falls
back to the legacy peak field ``p_max_full`` (panel/label flag this), in which
case panel (b) and the difference map mix JAX-amplitude with k-Wave-peak.

Requires a build of ``jaxisymmetric`` that provides ``run_simulation_amplitude``.

Run with::

    python paper/figure2/freefield_comparison_steady.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
from jax import device_put
from jaxisymmetric import SimConfig, Source, run_simulation_amplitude
from jaxisymmetric.sources import make_focused_bowl_source
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# H-117 transducer parameters (nominal)
# ---------------------------------------------------------------------------
FOCUS_POS = 75e-3
RADIUS_CURV = 63.2e-3
OUTER_DIAM = 64e-3
INNER_DIAM = 22.6e-3

# ---------------------------------------------------------------------------
# Grid and simulation config
# ---------------------------------------------------------------------------
Nx, Nr = 256, 81
dx = dr = 0.5e-3
c0, rho0 = 1500.0, 1000.0

source_freq = 0.3e6
ramp_cycles = 3
cfl = 0.3
PML_WIDTH = 5
dt = cfl * dx / c0
ramp_steps = round(ramp_cycles * (1 / source_freq) / dt)

# Number of final periods over which the steady-state amplitude is averaged.
RECORD_CYCLES = 3

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl, pml_width=PML_WIDTH)

x = jnp.arange(Nx) * dx
r = jnp.arange(Nr) * dr

source_mask = make_focused_bowl_source(
    cfg,
    focus_pos=FOCUS_POS,
    radius_curvature=RADIUS_CURV,
    outer_diameter=OUTER_DIAM,
    inner_diameter=INNER_DIAM,
)
source = Source(mask=source_mask, freq=source_freq, ramp_steps=ramp_steps)

# ---------------------------------------------------------------------------
# Run JAXisymmetric simulation — steady-state amplitude instead of peak pressure
# ---------------------------------------------------------------------------
print(f"Running free-field simulation (steady-state amplitude, last {RECORD_CYCLES} cycles)...")
jax_data = jax.jit(
    lambda m, c, s: run_simulation_amplitude(m, c, s, record_cycles=RECORD_CYCLES)
)(cfg.homogeneous_medium(), cfg, source)
jax_data.block_until_ready()
print("  done.")

# ---------------------------------------------------------------------------
# Load k-Wave reference.
# Prefer the steady-state amplitude field (p_amp_full) produced by
# H117comparisonJAXisymmetric.m, giving a true steady-vs-steady comparison.
# Fall back to the legacy peak field (p_max_full) if that file isn't there yet.
# ---------------------------------------------------------------------------
try:
    kwave_raw = loadmat("data/H117_FreeFieldkWave_steady.mat")
    kwave_field = kwave_raw["p_amp_full"]
    kwave_label = "b) k-Wave Steady-State Amplitude [ref. 1]"
    print("Using k-Wave steady-state amplitude (p_amp_full).")
except (FileNotFoundError, KeyError):
    kwave_raw = loadmat("data/H117_FreeFieldkWave.mat")
    kwave_field = kwave_raw["p_max_full"]
    kwave_label = "b) k-Wave Gain [ref. 1] (PEAK — run the .m for steady amplitude)"
    print("WARNING: steady k-Wave file not found; falling back to PEAK p_max_full.")

# Pre-processing: mirror and crop PML
crop = PML_WIDTH

jax_data = jnp.concatenate([jax_data[:, 1:][:, ::-1], jax_data], axis=1)
jax_sim = jax_data[crop:-crop, crop:-crop]

kwave_sim = device_put(kwave_field)[crop:-crop, crop:-crop]

# z = 0 at the bowl rim (aperture) plane — the greatest-radius / highest-axial
# point of the shell, computed from the geometry: x_rim = focus - sqrt(RoC^2 - r_out^2)
x_rim = FOCUS_POS - jnp.sqrt(RADIUS_CURV**2 - (OUTER_DIAM / 2) ** 2)
x_vec = 1e3 * (x[crop:-crop] - x_rim)
r_vec_full = jnp.concatenate([-jnp.flip(r[1:]), r])
r_vec = 1e3 * r_vec_full[crop:-crop]

mask_full = jnp.concatenate([jnp.flip(source_mask[:, 1:], axis=1), source_mask], axis=1)
mask_full = mask_full[crop:-crop, crop:-crop]

# Error metrics (excluding source region)
difference = jnp.abs(jax_sim - kwave_sim) / jnp.max(kwave_sim)
difference = jnp.where(mask_full > 0, 0.0, difference)

jax_clean = jnp.where(mask_full > 0, 0.0, jax_sim)
kwave_clean = jnp.where(mask_full > 0, 0.0, kwave_sim)

rel_l2 = jnp.sqrt(jnp.sum(jnp.square(kwave_clean - jax_clean)) / jnp.sum(jnp.square(kwave_clean)))
rel_linf = jnp.max(jnp.abs(kwave_clean - jax_clean)) / jnp.max(jnp.abs(kwave_clean))
rmse = 100 * jnp.sqrt(jnp.mean(difference**2))
max_diff = 100 * jnp.max(difference)

print(f"RMSE: {rmse:.4f}%")
print(f"Max Absolute Difference: {max_diff:.4f}%")
print(f"Relative L2 Error: {rel_l2:.4f}")
print(f"Relative Linf Error: {rel_linf:.4f}")

# ---------------------------------------------------------------------------
# Plot panels a-c
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)


def plot_sim(ax, data, title, cmap='viridis', is_diff=False):
    # We transpose 'data' to match MATLAB's imagesc behavior with (x, r) axes
    val = 100 * data if is_diff else data

    # Use the 'cmap' variable here instead of the hardcoded string
    im = ax.imshow(
        val.T,
        extent=[x_vec[0], x_vec[-1], r_vec[0], r_vec[-1]],
        origin='lower',
        aspect='equal',
        cmap=cmap)

    # Overlay Contour (Mask)
    ax.contour(x_vec, r_vec, mask_full.T, levels=[0.5], colors='w', linewidths=1)

    ax.set_title(title)
    ax.set_ylabel('Radial Position [mm]')

    # Note: Ensure 'fig' is defined in your outer scope or passed in
    fig.colorbar(im, ax=ax)

plot_sim(axes[0], jax_sim, 'a) JAXisymmetric Steady-State Amplitude [ref. 1]', cmap='viridis')
plot_sim(axes[1], kwave_sim, kwave_label, cmap='viridis')
plot_sim(axes[2], difference, 'c) Absolute Difference [%]', cmap='gist_gray', is_diff=True)



axes[2].set_xlabel("Axial Position [mm]")

OUT = Path(__file__).resolve().parent

plt.tight_layout()
plt.savefig(str(OUT / "freefield_comparison_steady.pdf"), format="pdf")
plt.savefig(str(OUT / "freefield_comparison_steady.svg"), format="svg")
plt.show()
print(f"Figure saved to {OUT / 'freefield_comparison_steady.pdf'}")
