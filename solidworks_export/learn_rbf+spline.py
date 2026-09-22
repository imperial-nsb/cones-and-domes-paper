"""
Optimisation of RBF and Spline lens geometries to maximise focal
pressure at the focus of an H-117-style transducer.

Produces snapshots and results used for Figure 03 panels (e-h).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
import optax
from jaxisymmetric import BoundedParam, SimConfig, Source, run_simulation
from jaxisymmetric.geometry import RbfGeometry, SplineGeometry
from jaxisymmetric.loss import FocalPressureLoss, IntersectionPenalty, rectangular_roi
from jaxisymmetric.sources import make_holography_source
from jaxisymmetric.train import run_optimization
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# 1.  Grid and simulation config
# ---------------------------------------------------------------------------
Nx, Nr = 512, 256
dx = dr = 0.25e-3
c0, rho0 = 1500.0, 1000.0
source_freq = 0.3e6

cfl = 0.1
dt = cfl * dx / c0
ramp_steps = round(3 * (1 / source_freq) / dt)

cfg = SimConfig(Nx=Nx, Nr=Nr, dx=dx, dr=dr, c0=c0, rho0=rho0, cfl=cfl)

# Load transducer profile
rdata = loadmat("data/rprofile_FF.mat")
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

focus_x = zpos + 60e-3
roi_mask = rectangular_roi(
    cfg.X,
    cfg.R,
    focus_x=focus_x,
    focus_r=0.0,
    size_x=5e-3,
    size_r=2e-3,
)

loss_obj = FocalPressureLoss(roi_mask) + 10.0 * IntersectionPenalty(source_binary)

P1 = jnp.array([zpos - 0.5e-3, rpos + 2e-3])
P6_x_fixed = zpos + 38e-3

# Degenerate bounds (lower == upper) pin fixed coordinates with zero gradient:
#   row 0      → P1 fully fixed
#   row 5 col 0 → P6_x_fixed (x only); r is learnable
spline_lower = jnp.array(
    [
        P1,
        [zpos + 2e-3, 2e-3],
        [zpos + 10e-3, 2e-3],
        [zpos + 18e-3, 2e-3],
        [zpos + 28e-3, 2e-3],
        [P6_x_fixed, 2e-3],
    ]
)
spline_upper = jnp.array(
    [
        P1,
        [zpos + 8e-3, 60e-3],
        [zpos + 17e-3, 60e-3],
        [zpos + 27e-3, 60e-3],
        [zpos + 36e-3, 60e-3],
        [P6_x_fixed, 20e-3],
    ]
)

initial_cps_physical = jnp.array(
    [
        P1,
        [zpos + 8e-3, 40e-3],
        [zpos + 17e-3, 30e-3],
        [zpos + 27e-3, 20e-3],
        [zpos + 36e-3, 10e-3],
        [P6_x_fixed, 10e-3],
    ]
)

geometry_spline = SplineGeometry(
    c=2500.0,
    rho=1200.0,
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
    return loss_obj(p_max, mask), p_max


print("Running Spline optimisation...")
result_spline = run_optimization(
    loss_fn_spline,
    geometry_spline,
    n_steps=50,
    opt=optax.adam(0.2),
    verbose=True,
    log_every=5,
    has_aux=True,
)
final_geometry_spline = result_spline.geometry
final_shape_spline = final_geometry_spline(cfg.X, cfg.R)
final_field_spline = jax.jit(run_simulation)(
    final_geometry_spline.as_medium(cfg), cfg, source
)

# ---------------------------------------------------------------------------
# 4.  RBF Optimisation
# ---------------------------------------------------------------------------
x_start = zpos - 0.5e-3
x_end = zpos + 38e-3
r_start = rpos + 2e-3

lower_rbf = jnp.array([0e-3, 2e-3, 2e-3, 2e-3, 2e-3])
upper_rbf = jnp.array([40e-3, 40e-3, 40e-3, 40e-3, 30e-3])
lower_r_end = 0.5e-3
upper_r_end = 20e-3

initial_rbf_weights = jnp.array([35e-3, 20e-3, 25e-3, 20e-3, 15e-3])
initial_r_end = 10.0e-3

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
    return loss_obj(p_max, mask), p_max

print("Running RBF optimisation...")
result_rbf = run_optimization(
    loss_fn_rbf,
    geometry_rbf,
    n_steps=50,
    opt=optax.adam(0.3),
    verbose=True,
    log_every=5,
    has_aux=True,
)
final_geometry_rbf = result_rbf.geometry
final_shape_rbf = final_geometry_rbf(cfg.X, cfg.R)
final_field_rbf = jax.jit(run_simulation)(final_geometry_rbf.as_medium(cfg), cfg, source)

# ---------------------------------------------------------------------------
# 5.  Animation
# ---------------------------------------------------------------------------
print("Collecting fields for animation...")

# Reuse fields recorded during training — no recomputation needed
assert result_spline.aux_history is not None
assert result_rbf.aux_history is not None
frame_indices = list(range(len(result_spline.geometry_history)))
field_hist_spline = result_spline.aux_history
field_hist_rbf = result_rbf.aux_history

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
OUT = Path(__file__).resolve().parent
r = cfg.r
x = jnp.arange(Nx) * dx - zpos
full_r = jnp.concatenate([-r[1:][::-1], r])
extent = [x[0]*1e3, x[-1]*1e3, full_r[-1]*1e3, full_r[0]*1e3]

# Initial frames setup with scaling to final frame
vmax = max(jnp.max(field_hist_rbf[-1]), jnp.max(field_hist_spline[-1])) / 1e6
pmax_axial = (
    max(jnp.max(field_hist_rbf[-1][:, 0]), jnp.max(field_hist_spline[-1][:, 0])) / 1e6
)

# e) RBF field
ax_rbf = axes[0]
field_full_rbf_0 = jnp.concatenate(
    [field_hist_rbf[0][:, 1:][:, ::-1], field_hist_rbf[0]], axis=1
)
im_rbf = ax_rbf.imshow(
    field_full_rbf_0.T / 1e6,
    extent=extent,
    cmap="magma",
    origin="upper",
    aspect="equal",
    vmax=vmax,
)
fig.colorbar(im_rbf, ax=ax_rbf, label="Peak Pressure [MPa]")
cnt_rbf_up = [None]
cnt_rbf_lo = [None]
ax_rbf.contour(
    x * 1e3, r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
ax_rbf.contour(
    x * 1e3, -r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
ax_rbf.set_title("e) RBF")
ax_rbf.set_xlabel("Axial Position [mm]")
ax_rbf.set_ylabel("Radial Position [mm]")

# f) Spline field
ax_spline = axes[1]
field_full_spline_0 = jnp.concatenate(
    [field_hist_spline[0][:, 1:][:, ::-1], field_hist_spline[0]], axis=1
)
im_spline = ax_spline.imshow(
    field_full_spline_0.T / 1e6,
    extent=extent,
    cmap="magma",
    origin="upper",
    aspect="equal",
    vmax=vmax,
)
fig.colorbar(im_spline, ax=ax_spline, label="Peak Pressure [MPa]")
cnt_spline_up = [None]
cnt_spline_lo = [None]
ax_spline.contour(
    x * 1e3, r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
ax_spline.contour(
    x * 1e3, -r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
ax_spline.set_title("f) Spline")
ax_spline.set_xlabel("Axial Position [mm]")
ax_spline.set_ylabel("Radial Position [mm]")

# g) Axial profile
ax_p = axes[2]
ax_p.plot(x * 1e3, target_field[:, 0] / 1e6, label="Free-Field", color="k")
(line_spline,) = ax_p.plot([], [], label="Spline", color="b", linestyle="--")
(line_rbf,) = ax_p.plot([], [], label="RBF", color="r", linestyle="-.")
ax_p.axvline((focus_x - 2.5e-3 - zpos) * 1e3, color="g", linestyle=":", label="ROI")
ax_p.axvline((focus_x + 2.5e-3 - zpos) * 1e3, color="g", linestyle=":")
ax_p.set_title("g)")
ax_p.set_xlabel("Axial Position [mm]")
ax_p.set_ylabel("Peak Pressure [MPa]")
ax_p.set_ylim(0, pmax_axial * 1.1)
ax_p.legend()

# h) Pressure histories
ax_h = axes[3]
hist_spline_full = -jnp.array(result_spline.loss_history) / 1e6
hist_rbf_full = -jnp.array(result_rbf.loss_history) / 1e6
ax_h.plot(hist_spline_full, color="b", alpha=0.3)
ax_h.plot(hist_rbf_full, color="r", alpha=0.3)
(line_h_spline,) = ax_h.plot([], [], label="Spline", color="b")
(line_h_rbf,) = ax_h.plot([], [], label="RBF", color="r")
ax_h.set_title("h)")
ax_h.set_xlabel("Optimisation Step")
ax_h.set_ylabel("Mean ROI Peak Pressure [MPa]")
ax_h.legend()

plt.tight_layout()


def update(frame_idx):
    f_spline = field_hist_spline[frame_idx]
    f_rbf = field_hist_rbf[frame_idx]
    g_spline = result_spline.geometry_history[frame_idx]
    g_rbf = result_rbf.geometry_history[frame_idx]

    # Update fields
    field_full_rbf = jnp.concatenate([f_rbf[:, 1:][:, ::-1], f_rbf], axis=1)
    im_rbf.set_data(field_full_rbf.T / 1e6)

    field_full_spline = jnp.concatenate([f_spline[:, 1:][:, ::-1], f_spline], axis=1)
    im_spline.set_data(field_full_spline.T / 1e6)

    # Update contours
    for c in [cnt_rbf_up[0], cnt_rbf_lo[0], cnt_spline_up[0], cnt_spline_lo[0]]:
        if c is not None:
            c.remove()

    shape_rbf = g_rbf(cfg.X, cfg.R)
    cnt_rbf_up[0] = ax_rbf.contour(
        x * 1e3, r * 1e3, shape_rbf.T, levels=[0.5], colors="white", linewidths=1.5
    )
    cnt_rbf_lo[0] = ax_rbf.contour(
        x * 1e3, -r * 1e3, shape_rbf.T, levels=[0.5], colors="white", linewidths=1.5
    )

    shape_spline = g_spline(cfg.X, cfg.R)
    cnt_spline_up[0] = ax_spline.contour(
        x * 1e3, r * 1e3, shape_spline.T, levels=[0.5], colors="white", linewidths=1.5
    )
    cnt_spline_lo[0] = ax_spline.contour(
        x * 1e3, -r * 1e3, shape_spline.T, levels=[0.5], colors="white", linewidths=1.5
    )

    # Update axial profiles
    line_spline.set_data(x * 1e3, f_spline[:, 0] / 1e6)
    line_rbf.set_data(x * 1e3, f_rbf[:, 0] / 1e6)

    # Update history lines
    line_h_spline.set_data(jnp.arange(frame_idx + 1), hist_spline_full[: frame_idx + 1])
    line_h_rbf.set_data(jnp.arange(frame_idx + 1), hist_rbf_full[: frame_idx + 1])

    return [im_rbf, im_spline, line_spline, line_rbf, line_h_spline, line_h_rbf]


filename = "rbf+spline_learning"

ani = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
writer = FFMpegWriter(fps=10)
ani_path = str(OUT / f"{filename}.mp4")
print(f"Saving animation to {ani_path}...")
ani.save(ani_path, writer=writer)

ani_path_gif = str(OUT / f"{filename}.gif")
print(f"Saving animation to {ani_path_gif}...")
ani.save(ani_path_gif, writer="pillow", fps=10)

# Final save
update(len(frame_indices) - 1)
plt.savefig(str(OUT / f"{filename}.pdf"), format="pdf")
plt.savefig(str(OUT / f"{filename}.png"), format="png", dpi=300)
plt.savefig(str(OUT / f"{filename}.svg"), format="svg")
print(f"Saved plots to {OUT}")


# Save results to .npz file
import numpy as np
from datetime import datetime

_OUT_DIR = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = str(_OUT_DIR / f"optimize_maxP_spline_{timestamp}.npz")

# Create results directory if it doesn't exist
import os
os.makedirs(_OUT_DIR, exist_ok=True)

np.savez(
    output_file,
    x=x,
    dx=dx,
    dr=dr,
    final_mask=np.array(final_shape_spline > 0.5, dtype=np.float32),  
)


# At end of examples/optimize_maxP.py, after `np.savez(...)`

from scipy.io import savemat

mat_output_file = str(_OUT_DIR / f"optimize_maxP_spline_{timestamp}.mat")

savemat(mat_output_file, {
    "x": np.array(x),                      # axial coordinate
    "r": np.array(r),                      # radial coordinate
    "final_mask": np.array(final_shape_spline > 0.5, dtype=np.float32),  # final geometry mask
    "final_field_spline": np.array(final_field_spline),       # pressure field with spline mask
})
print(f"Saved MATLAB file: {mat_output_file}")