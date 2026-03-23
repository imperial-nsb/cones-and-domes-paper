"""
paper/figure2/c103_comparison.py
=================================

Figure 2, panels d-f: Comparison of JAXisymmetric against k-Wave for the
H-117 transducer defined using acoustic holography with the C-103 cone
embedded into the simulation domain.

  d) JAXisymmetric pressure [MPa]
  e) k-Wave pressure [MPa]
  f) Normalised difference [%]

Produces: paper/figure2/c103_comparison.pdf

Run with::

    python paper/figure2/c103_comparison.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import device_put
from scipy.io import loadmat

from jaxisymmetric import SimConfig, Source, run_simulation
from jaxisymmetric.geometry import FixedGeometry
from jaxisymmetric.sources import make_holography_source

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
OUT = Path(__file__).resolve().parent

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

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl, pml_width=PML_WIDTH)

x = jnp.arange(Nx) * dx
r = jnp.arange(Nr) * dr

# ---------------------------------------------------------------------------
# Transducer source from measured radial profile (acoustic holography)
# ---------------------------------------------------------------------------
rdata = loadmat(str(DATA / "rprofile_FF.mat"))
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
    mat_file_path=str(DATA / "C103cone.mat"),
    c=2750.0,
    rho=1190.0,
    mask_key="C103array2D",
    x_key="x_vec",
    y_key="y_vec",
    source_zpos=zpos,
)
# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print("Running C-103 cone simulation...")
jax_data = jax.jit(run_simulation)(cone.as_medium(cfg), cfg, source)
jax_data.block_until_ready()
print("  done.")

# ---------------------------------------------------------------------------
# Load k-Wave reference
# ---------------------------------------------------------------------------
kwave_raw = loadmat(str(DATA / "EXPH117_C103_kWave.mat"))

crop = PML_WIDTH

jax_data = jnp.concatenate([jax_data[:, 1:][:, ::-1], jax_data], axis=1)
jax_sim = jax_data[crop:-crop, crop:-crop]

kwave_sim = device_put(kwave_raw["p_max_full"])
kwave_sim = kwave_sim[crop:-crop, crop:-crop]

x_vec = 1e3 * x[crop:-crop]
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

# Max pressure comparison with experimental reference
EXPmax = 1048796.8
JAXmaxerror = (jnp.max(jax_clean) - EXPmax) / EXPmax
kWavemaxerror = (jnp.max(kwave_clean) - EXPmax) / EXPmax
print(f"Max Pressure Error: JAXisymmetric: {JAXmaxerror:.4f}, k-Wave: {kWavemaxerror:.4f}")
print(f"Max Pressure: JAXisymmetric: {jnp.max(jax_clean):.2f}, k-Wave: {jnp.max(kwave_clean):.2f}")

# ---------------------------------------------------------------------------
# Plot panels d-f
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)


def plot_panel(ax, data, title, is_diff=False):
    val = 100 * data if is_diff else data
    im = ax.imshow(
        val.T,
        extent=[x_vec[0], x_vec[-1], r_vec[0], r_vec[-1]],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    ax.contour(x_vec, r_vec, mask_full.T, levels=[0.5], colors="k", linewidths=1)
    ax.set_title(title)
    ax.set_ylabel("Radial Position [mm]")
    fig.colorbar(im, ax=ax)


plot_panel(axes[0], jax_sim, "d) JAXisymmetric Pressure [MPa]")
plot_panel(axes[1], kwave_sim, "e) k-Wave Pressure [MPa]")
plot_panel(axes[2], difference_norm, "f) Normalised Difference [%]", is_diff=True)

axes[2].set_xlabel("Axial Position [mm]")

plt.tight_layout()
plt.savefig(str(OUT / "c103_comparison.pdf"), format="pdf")
plt.savefig(str(OUT / "c103_comparison.svg"), format="svg")
plt.show()
print(f"Figure saved to {OUT / 'c103_comparison.pdf'}")
