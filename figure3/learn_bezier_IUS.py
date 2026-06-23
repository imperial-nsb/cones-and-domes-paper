"""
Optimisation of a Bézier-curve lens geometry to match a reference (free-field)
acoustic pressure distribution at the focus of an H-117-style transducer.

Produces snapshots and results used for Figure 03 panels (a-d).
"""

from jax import config
config.update("jax_enable_x64", False)


from pathlib import Path


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxisymmetric import BoundedParam, SimConfig, Source, run_simulation
from jaxisymmetric.geometry import BezierGeometry
from jaxisymmetric.loss import IntersectionPenalty, RoiMseLoss, rectangular_roi
from jaxisymmetric.sources import make_holography_source
from jaxisymmetric.train import run_optimization
from matplotlib.animation import FFMpegWriter, FuncAnimation
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

rdata = loadmat("data/rprofile_FF.mat")
r_centers = jnp.array(rdata["r_centers"]).squeeze()
radial_prof = jnp.array(rdata["radial_prof"]).squeeze()

zpos = (Nx / 2) * dx - 40e-3
rpos = 37e-3

# Transducer params (H-117 nominal focal area configuration)
H117_FOCUS_X = zpos + 60e-3; # focal length


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
P2 = (zpos + 46e-3, 15e-3)

initial_cp = jnp.array([(P1[0] + P2[0]) / 2, (P2[1] + P1[1]) / 2])
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
    return loss_obj(p_max, mask), p_max


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
    has_aux=True,
)

final_geometry = result.geometry

# ---------------------------------------------------------------------------
# 6.  Final simulation
# ---------------------------------------------------------------------------
final_shape = final_geometry(cfg.X, cfg.R)
final_field = jax.jit(run_simulation)(final_geometry.as_medium(cfg), cfg, source)

# ---------------------------------------------------------------------------
# 7.  Animation
# ---------------------------------------------------------------------------
print("Collecting fields for animation...")

# We only animate every 5th step to keep it snappy, plus the final one
frame_indices = list(range(0, len(result.geometry_history), 2))
if (len(result.geometry_history) - 1) not in frame_indices:
    frame_indices.append(len(result.geometry_history) - 1)

assert result.aux_history is not None
field_history = [result.aux_history[i] for i in frame_indices]
geom_history = [result.geometry_history[i] for i in frame_indices]
loss_history_subset = [result.loss_history[i] for i in frame_indices]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
OUT = Path(__file__).resolve().parent

# Set up constant stuff
r = cfg.r
x = jnp.arange(Nx) * dx - zpos
full_r = jnp.concatenate([-r[1:][::-1], r])
extent = [x[0] * 1e3, x[-1] * 1e3, full_r[-1] * 1e3, full_r[0] * 1e3]

# Initial frame setup with scaling to final frame
final_field_max = jnp.max(field_history[-1]) / 1e6
axial_max = jnp.max(field_history[-1][:, 0]) / 1e6

ax_f = axes[0]
field_full_0 = jnp.concatenate(
    [field_history[0][:, 1:][:, ::-1], field_history[0]], axis=1
)
im = ax_f.imshow(
    field_full_0.T / 1e6,
    extent=extent,
    cmap="magma",
    origin="upper",
    aspect="equal",
    vmax=final_field_max,
)
cbar = fig.colorbar(im, ax=ax_f, label="Peak Pressure [MPa]")

# Prepare contours and plots for animation update
cnt_up = [None] * 2
cnt_lo = [None] * 2
src_up = ax_f.contour(
    x * 1e3, r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
src_lo = ax_f.contour(
    x * 1e3, -r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5
)
pt_w = ax_f.plot([], [], "wx")[0]
pt_w_mirr = ax_f.plot([], [], "wx")[0]

ax_f.set_title("a)")
ax_f.set_xlabel("Axial Position [mm]")
ax_f.set_ylabel("Radial Position [mm]")

# b) Axial profile
ax_p = axes[1]
ax_p.plot(x * 1e3, target_field[:, 0] / 1e6, label="Free-Field", color="C0")
(line_opt,) = ax_p.plot([], [], label="Optimised", color="C1", linestyle="--")
ax_p.set_title("b)")
ax_p.set_xlabel("Axial Position [mm]")
ax_p.set_ylabel("Peak Pressure [MPa]")
ax_p.set_ylim(0, axial_max * 1.1)
ax_p.legend()

# c) Control point trajectory
ax_t = axes[2]
cps_all = (jnp.stack([m.control_point.value for m in result.geometry_history]) - jnp.array([zpos, 0])) * 1e3
sc = ax_t.scatter(
    cps_all[:, 0],
    cps_all[:, 1],
    c=jnp.arange(len(cps_all)),
    cmap="plasma",
    s=10,
    alpha=0.3,
)
ax_t.plot(cps_all[0, 0], cps_all[0, 1], "gs", label="Initial")
(pt_current,) = ax_t.plot([], [], "r*", label="Current")
fig.colorbar(sc, ax=ax_t, label="Optimisation Step")
ax_t.set_title("c)")
ax_t.set_xlabel("Axial Position [mm]")
ax_t.set_ylabel("Radial Position [mm]")
ax_t.legend()

# d) Loss curve
ax_l = axes[3]
ax_l.plot(result.loss_history, alpha=0.3)
(line_loss,) = ax_l.plot([], [], color="C0")
ax_l.set_yscale("log")
ax_l.set_title("d)")
ax_l.set_xlabel("Optimisation Step")
ax_l.set_ylabel("Loss")

plt.tight_layout()


def update(frame_idx):
    f = field_history[frame_idx]
    g = geom_history[frame_idx]
    step_idx = frame_indices[frame_idx]

    # Update field
    field_full = jnp.concatenate([f[:, 1:][:, ::-1], f], axis=1)
    im.set_data(field_full.T / 1e6)

    # Update geometry contours (clear and redraw)
    for c in cnt_up + cnt_lo:
        if c is not None:
            c.remove()

    shape = g(cfg.X, cfg.R)
    cnt_up[0] = ax_f.contour(
        x * 1e3, r * 1e3, shape.T, levels=[0.5], colors="white", linewidths=1.5
    )
    cnt_lo[0] = ax_f.contour(
        x * 1e3, -r * 1e3, shape.T, levels=[0.5], colors="white", linewidths=1.5
    )

    # Update points
    cp = (g.control_point.value - jnp.array([zpos, 0])) * 1e3
    pt_w.set_data([cp[0]], [cp[1]])
    pt_w_mirr.set_data([cp[0]], [-cp[1]])

    # Update axial profile
    line_opt.set_data(x * 1e3, f[:, 0] / 1e6)

    # Update trajectory current point
    pt_current.set_data([cp[0]], [cp[1]])

    # Update loss curve progress
    line_loss.set_data(jnp.arange(step_idx + 1), result.loss_history[: step_idx + 1])

    return [im, pt_w, pt_w_mirr, line_opt, pt_current, line_loss]


filename = "bezier_learning_IUS"

ani = FuncAnimation(fig, update, frames=len(field_history), blit=False)
writer = FFMpegWriter(fps=10)
ani_path = str(OUT / f"{filename}.mp4")
print(f"Saving animation to {ani_path}...")
ani.save(ani_path, writer=writer)

ani_path_gif = str(OUT / f"{filename}.gif")
print(f"Saving animation to {ani_path_gif}...")
ani.save(ani_path_gif, writer="pillow", fps=10)

# Final save as before
# ---------------------------------------------------------------------------
# 8.  Final comparison figure
# ---------------------------------------------------------------------------
fig_final, axes_final = plt.subplots(2, 2, figsize=(12, 10))

# Get initial and final geometries/fields
first_geom = geom_history[0]
final_geom = geom_history[-1]
first_field = field_history[0]
final_field = field_history[-1]

# a) First geometry and field
ax_a = axes_final[0, 0]
first_shape = first_geom(cfg.X, cfg.R)
field_full_first = jnp.concatenate([first_field[:, 1:][:, ::-1], first_field], axis=1)
im_a = ax_a.imshow(
    field_full_first.T / 1e6,
    extent=extent,
    cmap="magma",
    origin="upper",
    aspect="equal",
    vmax=0.8,
)
ax_a.contour(x * 1e3, r * 1e3, first_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax_a.contour(x * 1e3, -r * 1e3, first_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax_a.contour(x * 1e3, r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax_a.contour(x * 1e3, -r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)

# Plot fixed endpoints (P1 and P2) as white 'x' marks
P1_rel = ((P1[0] - zpos) * 1e3, P1[1] * 1e3)
P2_rel = ((P2[0] - zpos) * 1e3, P2[1] * 1e3)
ax_a.plot([P1_rel[0]], [P1_rel[1]], "wx", markersize=6, markeredgewidth=2, label="fixed")
ax_a.plot([P2_rel[0]], [P2_rel[1]], "wx", markersize=6, markeredgewidth=2)
ax_a.plot([P1_rel[0]], [-P1_rel[1]], "wx", markersize=6, markeredgewidth=2)
ax_a.plot([P2_rel[0]], [-P2_rel[1]], "wx", markersize=6, markeredgewidth=2)

# Plot initial control point as green square
first_cp = (first_geom.control_point.value - jnp.array([zpos, 0])) * 1e3
ax_a.plot([first_cp[0]], [first_cp[1]], "gs", markersize=6, label="control")
ax_a.plot([first_cp[0]], [-first_cp[1]], "gs", markersize=6)


ax_a.set_title("a) Initial")
ax_a.set_xlabel("Axial Position [mm]")
ax_a.set_ylabel("Radial Position [mm]")
fig_final.colorbar(im_a, ax=ax_a, label="Peak Pressure [MPa]")

# b) Optimized geometry and field
ax_b = axes_final[0, 1]
final_shape = final_geom(cfg.X, cfg.R)
field_full_final = jnp.concatenate([final_field[:, 1:][:, ::-1], final_field], axis=1)
im_b = ax_b.imshow(
    field_full_final.T / 1e6,
    extent=extent,
    cmap="magma",
    origin="upper",
    aspect="equal",
    vmax=0.8,
)
ax_b.contour(x * 1e3, r * 1e3, final_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax_b.contour(x * 1e3, -r * 1e3, final_shape.T, levels=[0.5], colors="white", linewidths=1.5)
ax_b.contour(x * 1e3, r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)
ax_b.contour(x * 1e3, -r * 1e3, source_binary.T, levels=[0.5], colors="cyan", linewidths=1.5)

P1_rel = ((P1[0] - zpos) * 1e3, P1[1] * 1e3)
P2_rel = ((P2[0] - zpos) * 1e3, P2[1] * 1e3)
ax_b.plot([P1_rel[0]], [P1_rel[1]], "wx", markersize=6, markeredgewidth=2)
ax_b.plot([P2_rel[0]], [P2_rel[1]], "wx", markersize=6, markeredgewidth=2)
ax_b.plot([P1_rel[0]], [-P1_rel[1]], "wx", markersize=6, markeredgewidth=2)
ax_b.plot([P2_rel[0]], [-P2_rel[1]], "wx", markersize=6, markeredgewidth=2)

# Plot final control point as green square
final_cp = (final_geom.control_point.value - jnp.array([zpos, 0])) * 1e3
ax_b.plot([final_cp[0]], [final_cp[1]], "gs", markersize=6, label="Control Point")
ax_b.plot([final_cp[0]], [-final_cp[1]], "gs", markersize=6)

ax_b.set_title("b) Optimized")
ax_b.set_xlabel("Axial Position [mm]")
ax_b.set_ylabel("Radial Position [mm]")
fig_final.colorbar(im_b, ax=ax_b, label="Peak Pressure [MPa]")

# c) Axial profiles comparison
ax_c = axes_final[1, 0]
ax_c.plot(x * 1e3, target_field[:, 0] / 1e6, label="Free-Field", color="C0", linewidth=2)
ax_c.plot(x * 1e3, first_field[:, 0] / 1e6, label="Initial", color="C2", linestyle="--", linewidth=2)
ax_c.plot(x * 1e3, final_field[:, 0] / 1e6, label="Optimized", color="C1", linestyle="-.", linewidth=2)
ax_c.set_title("c) Axial Profiles")
ax_c.set_xlabel("Axial Position [mm]")
ax_c.set_ylabel("Peak Pressure [MPa]")
ax_c.set_ylim(0, 1.0)
ax_c.legend()
ax_c.grid(True, alpha=0.3)

# d) Control point trajectory
ax_d = axes_final[1, 1]
sc = ax_d.scatter(
    cps_all[:, 0],
    cps_all[:, 1],
    c=jnp.arange(len(cps_all)),
    cmap="plasma",
    s=20,
    alpha=0.6,
)
ax_d.plot(cps_all[0, 0], cps_all[0, 1], "gs", markersize=10, label="Initial")
ax_d.plot(cps_all[-1, 0], cps_all[-1, 1], "r*", markersize=15, label="Final")
ax_d.plot(cps_all[:, 0], cps_all[:, 1], "k-", alpha=0.2, linewidth=1)
fig_final.colorbar(sc, ax=ax_d, label="Optimisation Step")
ax_d.set_title("d) Control Point Evolution")
ax_d.set_xlabel("Axial Position [mm]")
ax_d.set_ylabel("Radial Position [mm]")
ax_d.legend()
ax_d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT / f"{filename}.pdf"), format="pdf")
plt.savefig(str(OUT / f"{filename}.png"), format="png", dpi=300)
plt.savefig(str(OUT / f"{filename}.svg"), format="svg")

print(f"Saved final comparison figure to {OUT}")