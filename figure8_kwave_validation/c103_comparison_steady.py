"""
paper/figure2/c103_comparison_steady.py
=======================================

Steady-state-amplitude variant of ``c103_comparison.py``.

The original script compares the JAXisymmetric *peak* pressure (``max|p|`` over
all time) against k-Wave for the H-117 transducer with the C-103 cone.  Peak
pressure is biased by the transient ramp-up/overshoot, so it is not a like-for-
like match to a steady-state CW experiment.  This version instead extracts the
**steady-state pressure amplitude** at the drive frequency over the final few
cycles, via :func:`jaxisymmetric.run_simulation_amplitude` (a single-frequency
lock-in / FFT-bin estimate accumulated online).

  d) JAXisymmetric steady-state pressure amplitude [MPa]
  e) k-Wave pressure [MPa]
  f) Normalised difference [%]

k-Wave reference: this script prefers ``EXPH117_C103_kWave_steady.mat``
(variable ``p_amp_full``) produced by ``EXPH117_comparisonJAXisymmetric.m``,
which extracts the k-Wave steady-state amplitude with the *same* lock-in DFT —
giving a true steady-vs-steady comparison.  If that file is not present yet it
falls back to the legacy peak field ``p_max_full`` (panel/label flag this), in
which case panel (e) and the difference map mix JAX-amplitude with k-Wave-peak.
Either way the headline number is the experimental max-pressure comparison
against ``EXPmax`` (a steady-state measurement).

Requires a build of ``jaxisymmetric`` that provides ``run_simulation_amplitude``.

Run with::

    python paper/figure2/c103_comparison_steady.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
from jax import device_put
from jaxisymmetric import SimConfig, Source, run_simulation_amplitude
from jaxisymmetric.geometry import FixedGeometry
from jaxisymmetric.sources import make_holography_source
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Grid and simulation config
# ---------------------------------------------------------------------------
Nx, Nr = 512, 256
dx = dr = 0.2e-3
c0, rho0 = 1500.0, 1000.0

source_freq = 0.3e6
ramp_cycles = 3
cfl = 0.1
PML_WIDTH = 10
dt = cfl * dx / c0
ramp_steps = round(ramp_cycles * (1 / source_freq) / dt)

# Number of final periods over which the steady-state amplitude is averaged.
RECORD_CYCLES = 3

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl, pml_width=PML_WIDTH)

x = jnp.arange(Nx) * dx
r = jnp.arange(Nr) * dr

# ---------------------------------------------------------------------------
# Transducer source from measured radial profile (acoustic holography)
# ---------------------------------------------------------------------------
rdata = loadmat("data/rprofile_FF.mat")
r_centers = jnp.array(rdata["r_centers"]).squeeze()
radial_prof = jnp.array(rdata["radial_prof"]).squeeze()

zpos = (Nx / 2) * dx - 45e-3

src_mask = make_holography_source(
    cfg,
    source_zpos=zpos,
    source_rpos=37e-3,
    r_centers=r_centers,
    radial_prof=radial_prof,
)
source = Source(mask=src_mask, freq=source_freq, ramp_steps=ramp_steps)

# ---------------------------------------------------------------------------
# Load C-103 cone geometry and build material fields
# ---------------------------------------------------------------------------
cone = FixedGeometry.from_mat(
    cfg,
    mat_file_path=str("data/C103cone.mat"),
    c=2270.0,
    rho=1200.0,
    mask_key="C103array2D",
    x_key="x_vec",
    y_key="y_vec",
    source_zpos=zpos,
)
# ---------------------------------------------------------------------------
# Run simulation — steady-state amplitude instead of peak pressure
# ---------------------------------------------------------------------------
print(f"Running C-103 cone simulation (steady-state amplitude, last {RECORD_CYCLES} cycles)...")
jax_data = jax.jit(
    lambda m, c, s: run_simulation_amplitude(m, c, s, record_cycles=RECORD_CYCLES)
)(cone.as_medium(cfg), cfg, source)
jax_data.block_until_ready()
print("  done.")

# ---------------------------------------------------------------------------
# Load k-Wave reference.
# Prefer the steady-state amplitude field (p_amp_full) produced by
# EXPH117_comparisonJAXisymmetric.m, giving a true steady-vs-steady comparison.
# Fall back to the legacy peak field (p_max_full) if that file isn't there yet.
# ---------------------------------------------------------------------------
try:
    kwave_raw = loadmat("data/EXPH117_C103_kWave_steady.mat")
    kwave_field = kwave_raw["p_amp_full"]
    kwave_label = "e) k-Wave Steady-State Amplitude [MPa]"
    print("Using k-Wave steady-state amplitude (p_amp_full).")
except (FileNotFoundError, KeyError):
    kwave_raw = loadmat("data/EXPH117_C103_kWave.mat")
    kwave_field = kwave_raw["p_max_full"]
    kwave_label = "e) k-Wave Pressure [MPa] (PEAK — run the .m for steady amplitude)"
    print("WARNING: steady k-Wave file not found; falling back to PEAK p_max_full.")

crop = PML_WIDTH

jax_data = jnp.concatenate([jax_data[:, 1:][:, ::-1], jax_data], axis=1)
jax_sim = jax_data[crop:-crop, crop:-crop]

kwave_sim = device_put(kwave_field)
kwave_sim = kwave_sim[crop:-crop, crop:-crop]

x_vec = 1e3 * (x[crop:-crop] - zpos)   # z = 0 at the source plane
r_vec_full = jnp.concatenate([-jnp.flip(r[1:]), r])
r_vec = 1e3 * r_vec_full[crop:-crop]

# Combined mask (source + cone)
mask_full = jnp.concatenate([jnp.flip(jnp.abs(src_mask[:, 1:]), axis=1), jnp.abs(src_mask)], axis=1)
mask_full = mask_full[crop:-crop, crop:-crop]
cone_mask = cone.mask
mask_full_cone = jnp.concatenate(
    [jnp.flip(jnp.abs(cone_mask[:, 1:]), axis=1), jnp.abs(cone_mask)], axis=1
)
mask_full_cone = mask_full_cone[crop:-crop, crop:-crop]
mask_full = mask_full + mask_full_cone

# Error metrics (excluding source and cone region)
jax_clean = jnp.where(mask_full > 0, 0.0, jax_sim)
kwave_clean = jnp.where(mask_full > 0, 0.0, kwave_sim)

difference_norm = jnp.abs((jax_clean / jnp.max(jax_clean)) - (kwave_clean / jnp.max(kwave_clean)))

rel_l2 = jnp.sqrt(jnp.sum(jnp.square(kwave_clean - jax_clean)) / jnp.sum(jnp.square(kwave_clean)))
rel_linf = jnp.max(jnp.abs(kwave_clean - jax_clean)) / jnp.max(jnp.abs(kwave_clean))

print(f"Relative L2 Error: {rel_l2:.4f}")
print(f"Relative Linf Error: {rel_linf:.4f}")

# Max pressure comparison with experimental reference (steady-state measurement)
EXPmax = 1019681.1
JAXmaxerror = (jnp.max(jax_clean) - EXPmax) / EXPmax
kWavemaxerror = (jnp.max(kwave_clean) - EXPmax) / EXPmax
print(f"Max Pressure Error: JAXisymmetric (steady-state amplitude): {JAXmaxerror:.4f}, k-Wave: {kWavemaxerror:.4f}")
print(f"Max Pressure: JAXisymmetric: {jnp.max(jax_clean):.2f}, k-Wave: {jnp.max(kwave_clean):.2f}")

# ---------------------------------------------------------------------------
# Plot panels d-f
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

plot_sim(axes[0], 1e-6 * jax_sim, 'd) JAXisymmetric Steady-State Amplitude [MPa]', cmap='viridis')
plot_sim(axes[1], 1e-6 * kwave_sim, kwave_label, cmap='viridis')
plot_sim(axes[2], difference_norm, 'f) Normalised Difference [%]', cmap='gist_gray', is_diff=True)



axes[2].set_xlabel("Axial Position [mm]")

OUT = Path(__file__).resolve().parent

plt.tight_layout()
plt.savefig(str(OUT / "c103_comparison_steady.pdf"), format="pdf")


plt.savefig(str(OUT / "c103_comparison_steady.svg"), format="svg")
plt.show()
print(f"Figure saved to {OUT / 'c103_comparison_steady.pdf'}")
