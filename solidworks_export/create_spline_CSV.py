"""
Export optimized spline to CSV

Loads the final_mask from optimize_maxP_spline_*.npz and generates
a csv with the exact same dimensions as the simulation.
"""

import numpy as np
import pandas as pd
from skimage import measure
from pathlib import Path
import os


# Find the latest spline result file
results_dir = Path("figure3")
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

print(f"Mask shape: {final_mask.shape}")
print(f"X grid length: {len(x_grid)}")

# Grid parameters (must match optimize_maxP.py)
dx = data["dx"] 
dr = data["dr"]

# Extract radius profile directly from the mask
# At each x position, find the maximum r where mask > 0.5
x_curve = []
rmin_curve = []
rmax_curve = []


for x_idx in range(final_mask.shape[0]):
    mask_at_x = final_mask[x_idx, :]
    r_indices = np.where(mask_at_x > 0.5)[0]

    if len(r_indices) > 0:
        min_r_idx = np.min(r_indices)
        max_r_idx = np.max(r_indices)
        x_curve.append(x_grid[x_idx] - x_grid[0])  # Convert to relative x (starting from 0)
        rmin_curve.append(min_r_idx * dr)  # Convert index to physical radius
        rmax_curve.append(max_r_idx * dr)  # Convert index to physical radius

x_curve = np.array(x_curve)
rmin_curve = np.array(rmin_curve)
rmax_curve = np.array(rmax_curve)

print(f"\nExtracted curve from mask:")
print(f"  X range: {x_curve.min() * 1e3:.2f} to {x_curve.max() * 1e3:.2f} mm")
print(f"  R range: {rmin_curve.min() * 1e3:.2f} to {rmin_curve.max() * 1e3:.2f} mm")
print(f"  Number of points: {len(x_curve)}")

# 1. Prepare the Forward points (x, r_min, 0)
forward_points = []
for x, rmin in zip(x_curve, rmin_curve):
    forward_points.append([x * 1e3, rmin * 1e3, 0.0])

# 2. Prepare the Backward points (flipped x, r_max, 0)
# [::-1] reverses the arrays
backward_points = []
for x, rmax in zip(reversed(x_curve), reversed(rmax_curve)):
    backward_points.append([x * 1e3, rmax * 1e3, 0.0])

# 3. Combine them into one continuous path
full_path = forward_points + backward_points

# 4. Write to file
with open("solidworks_export/spline_curve_points.txt", "w") as f:
    for p in full_path:
        f.write(f"{p[1]:.6f}\t{p[0]:.6f}\t{p[2]:.6f}\n")

print(f"✓ Curve points saved: solidworks_export/spline_curve_points.txt")

# 2. Try using marching squares
# Level 0.5 finds the midpoint between 0 and 1
contours = measure.find_contours(final_mask, 0.5)

# Usually, the longest contour is the surface you want
main_surface = max(contours, key=len)

# 3. Physical Scaling
# Grid indices (row, col) to physical units (r, z)
z_coords = main_surface[:, 0] * dx * 1e3  # Convert to mm
r_coords = main_surface[:, 1] * dr * 1e3  # Convert to mm

# 4. Prepare for SolidWorks (X, Y, Z format)
# SolidWorks Curve through XYZ points requires 3 columns
# We map R -> X, Z -> Y, and set Z-axis -> 0
df = pd.DataFrame({
    'X': r_coords, 
    'Y': z_coords,
    'Z': np.zeros_like(r_coords)
})

# 5. Export to CSV (SolidWorks likes tab or space delimited, but CSV works)
# Export to .txt with Tab Separation
# index=False and header=False are CRITICAL for SolidWorks
df.to_csv("solidworks_export/MC_surface_curve.txt", 
          sep='\t', 
          index=False, 
          header=False)

print(f"✓ Marching Squares curve saved to: solidworks_export/MC_surface_curve.txt")