"""
paper/figure3/plot_bezier.py
=====================================

Optimisation of a Bézier-curve lens geometry to match a reference (free-field)
acoustic pressure distribution at the focus of an H-117-style transducer.

Produces snapshots and results used for Figure 03 panels (a-d).
"""

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxisymmetric import BoundedParam, SimConfig, Source, run_simulation
from jaxisymmetric.geometry import BezierGeometry
from jaxisymmetric.loss import IntersectionPenalty, RoiMseLoss, rectangular_roi
from jaxisymmetric.sources import make_holography_source
from jaxisymmetric.train import run_optimization
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# 1.  Grid and simulation config
# ---------------------------------------------------------------------------
Nx, Nr = 256, 128
dx = dr = 0.5e-3
c0, rho0 = 1500.0, 1000.0
source_freq = 0.3e6

cfl = 0.1
dt = cfl * dx / c0
ramp_steps = round(3 * (1 / source_freq) / dt)

# Transducer params (H-117 nominal focal area configuration)
H117_FOCUS_X = 80e-3

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

ROOT = Path(__file__).resolve().parents[1]
rdata = loadmat(str(ROOT / "data" / "rprofile_FF.mat"))
r_centers = jnp.array(rdata["r_centers"]).squeeze()
radial_prof = jnp.array(rdata["radial_prof"]).squeeze()

zpos = (Nx / 2) * dx - 40e-3
rpos = 37e-3

src_mask = make_holography_source(
    cfg,
    source_zpos=zpos,
    source_rpos=rpos,
    r_centers=r_centers,
    radial_prof=radial_prof,
)
source_binary = jnp.abs(src_mask) > 0
source = Source(mask=src_mask, freq=source_freq, ramp_steps=ramp_steps)

# ---------------------------------------------------------------------------
# 2.  Bézier definition
# ---------------------------------------------------------------------------
P1 = (zpos - 2e-3, rpos + 2e-3)
P2 = (70e-3, 14.5e-3)

initial_cp = jnp.array([(P1[0] + P2[0]) / 2, (P2[1] + 2 * P1[1]) / 2])
lower = jnp.array([P1[0], P2[1]])
upper = jnp.array([P2[0], 2 * P1[1]])

geometry = BezierGeometry(
    c=2500.0,
    rho=1200.0,
    control_point=BoundedParam.from_physical(
        initial_cp,
        lower=lower,
        upper=upper,
    ),
    P1=P1,
    P2=P2,
    thickness=1.0e-3,
)

# ---------------------------------------------------------------------------
# 3.  Baseline (free-field) target simulation
# ---------------------------------------------------------------------------
print("Running baseline simulation (target field)…")
target_field = jax.jit(run_simulation)(cfg.homogeneous_medium(), cfg, source)
target_field.block_until_ready()
print("  baseline done.\n")

roi_mask = rectangular_roi(
    cfg.X,
    cfg.R,
    focus_x=H117_FOCUS_X,
    focus_r=0.0,
    size_x=30e-3,
    size_r=10e-3,
)

# ---------------------------------------------------------------------------
# 4.  Loss function
# ---------------------------------------------------------------------------
loss_obj = RoiMseLoss(target_field, roi_mask) + 10.0 * IntersectionPenalty(source_binary)


def loss_fn(geom):
    p_max = run_simulation(geom.as_medium(cfg), cfg, source)
    mask = geom(cfg.X, cfg.R)
    return loss_obj(p_max, mask)


# ---------------------------------------------------------------------------
# 5.  Optimisation
# ---------------------------------------------------------------------------
result = run_optimization(
    loss_fn,
    geometry,
    n_steps=100,
    opt=optax.adam(0.1),
    verbose=True,
    log_every=5,
)

final_geometry = cast(BezierGeometry, result.model)
print(f"\nOptimal control point: {final_geometry.control_point.value * 1e3} mm")

# ---------------------------------------------------------------------------
# 6.  Final simulation
# ---------------------------------------------------------------------------
final_shape = final_geometry(cfg.X, cfg.R)
final_field = jax.jit(run_simulation)(final_geometry.as_medium(cfg), cfg, source)

# ---------------------------------------------------------------------------
# 7.  Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
OUT = Path(__file__).resolve().parent

# a) Final acoustic field with geometry
ax = axes[0]
r = cfg.r
x = jnp.arange(Nx) * dx
full_r = jnp.concatenate([-r[1:][::-1], r])
extent = [x[0]*1e3, x[-1]*1e3, full_r[-1]*1e3, full_r[0]*1e3]
field_full = jnp.concatenate([final_field[:, 1:][:, ::-1], final_field], axis=1)

im = ax.imshow(field_full.T / 1e6, extent=extent, cmap="magma", origin="upper", aspect="auto")
ax.contour(x*1e3, r*1e3, final_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, final_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.plot(
    final_geometry.control_point.value[0] * 1e3,
    final_geometry.control_point.value[1] * 1e3,
    "wx",
)
ax.plot(
    final_geometry.control_point.value[0] * 1e3,
    -final_geometry.control_point.value[1] * 1e3,
    "wx",
)
fig.colorbar(im, ax=ax, label="Peak Pressure [MPa]")
ax.set_title("a)")
ax.set_xlabel("Axial Position [mm]")
ax.set_ylabel("Radial Position [mm]")

# b) Axial profile comparison
ax = axes[1]
ax.plot(cfg.X[:, 0]*1e3, target_field[:, 0] / 1e6, label="Free-Field", color="C0")
ax.plot(cfg.X[:, 0]*1e3, final_field[:, 0] / 1e6, label="Optimised", color="C1", linestyle="--")
ax.set_title("b)")
ax.set_xlabel("Axial Position [mm]")
ax.set_ylabel("Peak Pressure [MPa]")
ax.legend()

# c) Control point trajectory
ax = axes[2]
cps = jnp.stack([cast(BezierGeometry, m).control_point.value for m in result.model_history]) * 1e3
sc = ax.scatter(cps[:, 0], cps[:, 1], c=jnp.arange(len(cps)), cmap="plasma", s=10)
ax.plot(cps[0, 0], cps[0, 1], 'gs', label="Initial")
ax.plot(cps[-1, 0], cps[-1, 1], 'r*', label="Final")
fig.colorbar(sc, ax=ax, label="Optimisation Step")
ax.set_title("c)")
ax.set_xlabel("Control Point Axial Position [mm]")
ax.set_ylabel("Control Point Radial Position [mm]")
ax.legend()

# d) Loss curve
ax = axes[3]
ax.plot(result.loss_history)
ax.set_yscale('log')
ax.set_title("d)")
ax.set_xlabel("Optimisation Step")
ax.set_ylabel("Loss (log scale)")

plt.tight_layout()
plt.savefig(str(OUT / "fig3_bezier.pdf"), format="pdf")
plt.savefig(str(OUT / "fig3_bezier.png"), format="png", dpi=300)
print(f"Saved plots to {OUT}")
