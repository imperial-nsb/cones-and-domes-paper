"""
paper/figure3/plot_bezier.py
=====================================

Optimisation of a Bézier-curve lens geometry to match a reference (free-field)
acoustic pressure distribution at the focus of an H-117-style transducer.

Produces snapshots and results used for Figure 03 panels (a-d).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import equinox as eqx

from jaxisymmetric import SimConfig, Source, run_simulation
from jaxisymmetric.geometry import BezierGeometry
from jaxisymmetric.loss import IntersectionPenalty, RoiMseLoss, rectangular_roi
from jaxisymmetric.sources import make_focused_bowl_source
from jaxisymmetric.train import run_optimization

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

# Transducer params (H-117 nominal)
H117_ROC = 60e-3
H117_OUTER = 60e-3
H117_INNER = 20e-3
H117_FOCUS_X = 80e-3

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

source_mask = make_focused_bowl_source(
    cfg,
    focus_pos=H117_FOCUS_X,
    radius_curvature=H117_ROC,
    outer_diameter=H117_OUTER,
    inner_diameter=H117_INNER,
)
source = Source(mask=source_mask, freq=source_freq, ramp_steps=ramp_steps)

# ---------------------------------------------------------------------------
# 2.  Bézier definition
# ---------------------------------------------------------------------------
P1 = jnp.array([20e-3, 35e-3])
P2 = jnp.array([70e-3, 10e-3])

initial_cp = jnp.array([(P1[0] + P2[0]) / 2, (P2[1] + 1.5 * P1[1]) / 2])
lower = jnp.array([P1[0], P2[1]])
upper = jnp.array([P2[0], 1.5 * P1[1]])

def physical_to_latent(cp_phys, l, u):
    norm = jnp.clip((cp_phys - l) / (u - l + 1e-12), 1e-5, 1.0 - 1e-5)
    return jnp.log(norm / (1.0 - norm))

latent_cp = physical_to_latent(initial_cp, lower, upper)

geometry = BezierGeometry(
    c=2500.0,
    rho=1200.0,
    control_point=latent_cp,
    P1=P1,
    P2=P2,
    thickness=2.0e-3,
)

def to_physical(geom: BezierGeometry) -> BezierGeometry:
    cp_phys = lower + (upper - lower) * jax.nn.sigmoid(geom.control_point)
    geom = eqx.tree_at(lambda g: g.control_point, geom, cp_phys)
    geom = eqx.tree_at(lambda g: (g.P1, g.P2), geom, (P1, P2))
    return geom

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
loss_obj = RoiMseLoss(target_field, roi_mask) + 10.0 * IntersectionPenalty(source_mask)


def loss_fn(latent_geom: BezierGeometry):
    geom = to_physical(latent_geom)
    p_max = run_simulation(geom.as_medium(cfg), cfg, source)
    mask = geom(cfg.X, cfg.R)
    return loss_obj(p_max, mask)


# ---------------------------------------------------------------------------
# 5.  Optimisation
# ---------------------------------------------------------------------------
result = run_optimization(
    loss_fn,
    geometry,
    n_steps=20,
    opt=optax.adam(0.1),
    verbose=True,
    log_every=5,
)

final_geometry = to_physical(result.model)
print(f"\nOptimal control point: {final_geometry.control_point * 1e3} mm")

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
ax.contour(x*1e3, r*1e3, source_mask.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.contour(x*1e3, -r*1e3, source_mask.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax.plot(final_geometry.control_point[0]*1e3, final_geometry.control_point[1]*1e3, 'wx')
ax.plot(final_geometry.control_point[0]*1e3, -final_geometry.control_point[1]*1e3, 'wx')
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
physical_history = [to_physical(m) for m in result.model_history]
cps = jnp.stack([m.control_point for m in physical_history]) * 1e3
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
