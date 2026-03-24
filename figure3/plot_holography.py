"""
paper/figure3/plot_holography.py
=====================================

Optimisation of RBF and Spline lens geometries to maximise focal
pressure at the focus of an H-117-style transducer.

Produces snapshots and results used for Figure 03 panels (e-h).
"""

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxisymmetric import BoundedParam, SimConfig, Source, run_simulation
from jaxisymmetric.geometry import RbfGeometry, SplineGeometry
from jaxisymmetric.loss import FocalPressureLoss, IntersectionPenalty, rectangular_roi
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

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

# Load transducer profile
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
# 2.  Baseline (free-field) simulation
# ---------------------------------------------------------------------------
print("Running baseline simulation (target field)…")
target_field = jax.jit(run_simulation)(cfg.homogeneous_medium(), cfg, source)
target_field.block_until_ready()
print("  baseline done.\n")

roi_mask = rectangular_roi(
    cfg.X,
    cfg.R,
    focus_x=85e-3,
    focus_r=0.0,
    size_x=5e-3,
    size_r=2e-3,
)

loss_obj = FocalPressureLoss(roi_mask) + 10.0 * IntersectionPenalty(source_binary)

P1 = jnp.array([zpos - 0.5e-3, rpos + 2e-3])
P6_x_fixed = zpos + 48e-3

# Degenerate bounds (lower == upper) pin fixed coordinates with zero gradient:
#   row 0      → P1 fully fixed
#   row 5 col 0 → P6_x_fixed (x only); r is learnable
spline_lower = jnp.array(
    [
        P1,
        [zpos + 5e-3, 2e-3],
        [zpos + 16e-3, 2e-3],
        [zpos + 26e-3, 2e-3],
        [zpos + 35e-3, 2e-3],
        [P6_x_fixed, 2e-3],
    ]
)
spline_upper = jnp.array(
    [
        P1,
        [zpos + 15e-3, 60e-3],
        [zpos + 25e-3, 60e-3],
        [zpos + 35e-3, 60e-3],
        [zpos + 46e-3, 60e-3],
        [P6_x_fixed, 20e-3],
    ]
)

initial_cps_physical = jnp.array(
    [
        P1,
        [zpos + 10e-3, 40e-3],
        [zpos + 20e-3, 30e-3],
        [zpos + 30e-3, 20e-3],
        [zpos + 40e-3, 10e-3],
        [P6_x_fixed, 10e-3],
    ]
)

geometry_spline = SplineGeometry(
    c=2500.0,
    rho=1178.0,
    control_points=BoundedParam.from_physical(
        initial_cps_physical,
        lower=spline_lower,
        upper=spline_upper,
    ),
    thickness=2.0e-3,
)


def loss_fn_spline(geom):
    p_max = run_simulation(geom.as_medium(cfg), cfg, source)
    mask = geom(cfg.X, cfg.R)
    return loss_obj(p_max, mask)


print("Running Spline optimisation...")
result_spline = run_optimization(
    loss_fn_spline,
    geometry_spline,
    n_steps=50,
    opt=optax.adam(0.2),
    verbose=True,
    log_every=5,
)
final_geometry_spline = cast(SplineGeometry, result_spline.model)
final_shape_spline = final_geometry_spline(cfg.X, cfg.R)
final_field_spline = jax.jit(run_simulation)(
    final_geometry_spline.as_medium(cfg), cfg, source
)

# ---------------------------------------------------------------------------
# 4.  RBF Optimisation
# ---------------------------------------------------------------------------
x_start = zpos - 0.5e-3
x_end = zpos + 48e-3
r_start = rpos + 2e-3

lower_rbf = jnp.array([0e-3, 2e-3, 2e-3, 2e-3, 2e-3])
upper_rbf = jnp.array([40e-3, 40e-3, 40e-3, 40e-3, 30e-3])
lower_r_end = 0.5e-3
upper_r_end = 20e-3

initial_rbf_weights = jnp.array([30e-3, 25e-3, 20e-3, 15e-3, 10e-3])
initial_r_end = 5.0e-3

geometry_rbf = RbfGeometry(
    c=2500.0,
    rho=1200.0,
    rbf_weights=BoundedParam.from_physical(
        initial_rbf_weights, lower=lower_rbf, upper=upper_rbf
    ),
    r_end=BoundedParam.from_physical(
        jnp.array(initial_r_end),
        lower=jnp.array(lower_r_end),
        upper=jnp.array(upper_r_end),
    ),
    x_start=x_start,
    x_end=x_end,
    r_start=r_start,
    thickness=2.0e-3,
    bandwidth=0.15,
)

def loss_fn_rbf(geom):
    p_max = run_simulation(geom.as_medium(cfg), cfg, source)
    mask = geom(cfg.X, cfg.R)
    return loss_obj(p_max, mask)

print("Running RBF optimisation...")
result_rbf = run_optimization(
    loss_fn_rbf,
    geometry_rbf,
    n_steps=50,
    opt=optax.adam(0.3),
    verbose=True,
    log_every=5,
)
final_geometry_rbf = cast(RbfGeometry, result_rbf.model)
final_shape_rbf = final_geometry_rbf(cfg.X, cfg.R)
final_field_rbf = jax.jit(run_simulation)(final_geometry_rbf.as_medium(cfg), cfg, source)

# ---------------------------------------------------------------------------
# 5.  Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
OUT = Path(__file__).resolve().parent
extent = [cfg.X[0,0]*1e3, cfg.X[-1,0]*1e3, cfg.R[0,-1]*1e3, cfg.R[0,0]*1e3]

# e) Final acoustic field with RBF
ax = axes[0]
r = cfg.r
x = jnp.arange(Nx) * dx
full_r = jnp.concatenate([-r[1:][::-1], r])
extent = [x[0]*1e3, x[-1]*1e3, full_r[-1]*1e3, full_r[0]*1e3]

field_full_rbf = jnp.concatenate([final_field_rbf[:, 1:][:, ::-1], final_field_rbf], axis=1)
im = ax.imshow(field_full_rbf.T / 1e6, extent=extent, cmap="magma", origin="upper", aspect="auto")
ax.contour(x*1e3, r*1e3, final_shape_rbf.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, final_shape_rbf.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
fig.colorbar(im, ax=ax, label="Peak Pressure [MPa]")
ax.set_title("e)")
ax.set_xlabel("Axial Position [mm]")
ax.set_ylabel("Radial Position [mm]")

# f) Final acoustic field with Spline
ax = axes[1]
field_full_spline = jnp.concatenate([final_field_spline[:, 1:][:, ::-1], final_field_spline], axis=1)
im2 = ax.imshow(field_full_spline.T / 1e6, extent=extent, cmap="magma", origin="upper", aspect="auto")
ax.contour(x*1e3, r*1e3, final_shape_spline.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, final_shape_spline.T, levels=[0.5], colors="white", linewidths=1.5)
ax.contour(x*1e3, r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
fig.colorbar(im2, ax=ax, label="Peak Pressure [MPa]")
ax.set_title("f)")
ax.set_xlabel("Axial Position [mm]")
ax.set_ylabel("Radial Position [mm]")

# g) Axial profile comparison
ax = axes[2]
ax.plot(cfg.X[:, 0]*1e3, target_field[:, 0] / 1e6, label="Free-Field", color="k")
ax.plot(cfg.X[:, 0]*1e3, final_field_spline[:, 0] / 1e6, label="Spline", color="b", linestyle="--")
ax.plot(cfg.X[:, 0]*1e3, final_field_rbf[:, 0] / 1e6, label="RBF", color="r", linestyle="-.")
ax.axvline(82.5, color='g', linestyle=':', label="ROI")
ax.axvline(87.5, color='g', linestyle=':')
ax.set_title("g)")
ax.set_xlabel("Axial Position [mm]")
ax.set_ylabel("Peak Pressure [MPa]")
ax.legend()

# h) Mean ROI peak pressure histories
ax = axes[3]
loss_spline = -jnp.array(result_spline.loss_history) / 1e6
loss_rbf = -jnp.array(result_rbf.loss_history) / 1e6
ax.plot(loss_spline, label=f"Spline (final = {loss_spline[-1]:.2f})", color="b")
ax.plot(loss_rbf, label=f"RBF (final = {loss_rbf[-1]:.2f})", color="r")
ax.set_title("h)")
ax.set_xlabel("Optimisation Step")
ax.set_ylabel("Mean ROI peak pressure [MPa]")
ax.legend()

plt.tight_layout()
plt.savefig(str(OUT / "fig3_holography.pdf"), format="pdf")
plt.savefig(str(OUT / "fig3_holography.png"), format="png", dpi=300)
print(f"Saved plots to {OUT}")
