"""
Export optimized spline to CSV

Loads the final_mask from optimize_maxP_spline_*.npz and generates
a csv with the exact same dimensions as the simulation.
"""

import numpy as np
from pathlib import Path
import os


# Find the latest spline result file
results_dir = Path("results")
spline_files = sorted(results_dir.glob("optimize_maxP_spline_*.npz"))

if not spline_files:
    print("Error: No spline result files found in results/ directory")
    print("Run optimize_maxP.py first and save final_mask in the .npz file")
    exit(1)

# Load the latest file
npz_file = spline_files[-1]
data = np.load(npz_file)

print(f"Loaded: {npz_file.name}")

# Check if final_mask exists
if "final_mask" not in data:
    print("Error: final_mask not found in .npz file")
    print("Please update optimize_maxP.py to save final_mask")
    exit(1)

final_mask = data["final_mask"]
x_grid = data["x"]
control_points = data["final_control_points"]

print(f"Mask shape: {final_mask.shape}")
print(f"X grid length: {len(x_grid)}")

# Grid parameters (must match optimize_maxP.py)
dx = 0.5e-3  # 0.5 mm
dr = 0.5e-3  # 0.5 mm

# Extract radius profile directly from the mask
# At each x position, find the maximum r where mask > 0.5
x_curve = []
r_curve = []

for x_idx in range(final_mask.shape[0]):
    mask_at_x = final_mask[x_idx, :]
    r_indices = np.where(mask_at_x > 0.5)[0]

    if len(r_indices) > 0:
        max_r_idx = np.min(r_indices)
        x_curve.append(x_grid[x_idx])
        r_curve.append(max_r_idx * dr)  # Convert index to physical radius

x_curve = np.array(x_curve)
r_curve = np.array(r_curve)

print(f"\nExtracted curve from mask:")
print(f"  X range: {x_curve.min() * 1e3:.2f} to {x_curve.max() * 1e3:.2f} mm")
print(f"  R range: {r_curve.min() * 1e3:.2f} to {r_curve.max() * 1e3:.2f} mm")
print(f"  Number of points: {len(x_curve)}")

# Also save the CSV for reference
with open("solidworks_export/spline_curve_points.csv", "w") as f:
    f.write("X_mm,R_mm\n")
    for x, r in zip(x_curve, r_curve):
        f.write(f"{x * 1e3:.6f},{r * 1e3:.6f}\n")

print(f"✓ Curve points also saved to: solidworks_export/spline_curve_points.csv")

# Print control points
print(f"\nControl points (for reference):")
for i, pt in enumerate(control_points):
    print(f"  P{i + 1}: X={pt[0] * 1e3:7.2f} mm, R={pt[1] * 1e3:7.2f} mm")
