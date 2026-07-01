from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# =========================
# Settings
# code used to generate figure 5.7 matrix
# Code created by Albaraa Adel Farouk Khalil
# Last updated: 27.06.2026
# LLM ChatGPT have been used as an assist to develope this code
# =========================
folder = Path(r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging\intensity")

PREBLEACH_N = 15
POSTBLEACH_N = 3

# columns and rows of matrix
laser_order = [4, 3, 2, 1]   # columns
frame_order = [4, 3, 2, 1]   # rows

# only include raw condition files, not summary outputs
csv_files = [
    f for f in sorted(folder.glob("*.csv"))
    if re.match(r"^\d+_L_\d+_F(?:_\d+)?\.csv$", f.name)
]

if len(csv_files) == 0:
    raise ValueError("No matching raw condition CSV files found.")

# =========================
# FILENAME PARSER
# =========================
def parse_condition(filename):
    """
    Expected filenames:
    1_L_3_F.csv
    3_L_3_F_50.csv
    3_L_3_F_75.csv
    3_L_3_F_80.csv
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_L_(\d+)_F(?:_(\d+))?$", stem)

    if not m:
        raise ValueError(f"Filename format not recognized: {filename}")

    lasers = int(m.group(1))
    frames = int(m.group(2))
    laser3_percent = int(m.group(3)) if m.group(3) is not None else np.nan

    return lasers, frames, laser3_percent


# =========================
# ANALYZE ALL FILES
# =========================
results = []

for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    if "Gray_Value" not in df.columns:
        print(f"Skipping {file.name}: no Gray_Value column")
        continue

    intensity = df["Gray_Value"].dropna().to_numpy()

    if len(intensity) < PREBLEACH_N + POSTBLEACH_N:
        print(f"Skipping {file.name}: not enough values")
        continue

    prebleach = intensity[:PREBLEACH_N]
    postbleach = intensity[PREBLEACH_N:PREBLEACH_N + POSTBLEACH_N]

    mean_pre = np.mean(prebleach)
    mean_post = np.mean(postbleach)
    norm_post = mean_post / mean_pre

    lasers, frames, laser3_percent = parse_condition(file.name)

    results.append({
        "file": file.name,
        "lasers": lasers,
        "frames": frames,
        "laser3_percent": laser3_percent,
        "mean_prebleach": mean_pre,
        "mean_postbleach": mean_post,
        "normalized_postbleach": norm_post,
        "bleaching_depth": 1 - norm_post
    })

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No usable files were analyzed.")

results_df = results_df.sort_values(
    by=["frames", "lasers", "laser3_percent"],
    ascending=[False, False, True],
    na_position="last"
).reset_index(drop=True)

summary_path = folder / "postbleach_summary_all_files.csv"
results_df.to_csv(summary_path, index=False)

print("Per-file results:")
print(results_df)
print(f"\nSaved summary table to:\n{summary_path}")

# =========================
# COLOR SCALE
# =========================
vmin = results_df["normalized_postbleach"].min()
vmax = results_df["normalized_postbleach"].max()

cmap = plt.cm.viridis_r
norm = Normalize(vmin=vmin, vmax=vmax)

# =========================
# PLOT MANUAL HEATMAP
# =========================
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

fig, ax = plt.subplots(figsize=(8, 7))

# Draw cell borders for whole 4x4 matrix
for i, frame in enumerate(frame_order):
    for j, laser in enumerate(laser_order):
        outer = Rectangle(
            (j, i), 1, 1,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2
        )
        ax.add_patch(outer)

        # Select data for this cell
        cell_data = results_df[
            (results_df["frames"] == frame) &
            (results_df["lasers"] == laser)
        ].copy()

        if cell_data.empty:
            # leave transparent
            continue

        # Sort mini-sections by intensity percentage if present
        cell_data = cell_data.sort_values("laser3_percent", na_position="last").reset_index(drop=True)

        n = len(cell_data)

        if n == 1:
            # One file only: fill full box
            value = cell_data.loc[0, "normalized_postbleach"]
            color = cmap(norm(value))

            rect = Rectangle(
                (j, i), 1, 1,
                facecolor=color,
                edgecolor="black",
                linewidth=1.2
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5, i + 0.5,
                f"{value:.2f}",
                ha="center", va="center",
                color="white",
                fontsize=16,
                fontweight="bold"
            )

        else:
            # Multiple files: split into stacked mini-sections
            sub_h = 1 / n

            for k, (_, row) in enumerate(cell_data.iterrows()):
                value = row["normalized_postbleach"]
                color = cmap(norm(value))

                y0 = i + k * sub_h

                rect = Rectangle(
                    (j, y0), 1, sub_h,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1
                )
                ax.add_patch(rect)

                # label
                if pd.notna(row["laser3_percent"]):
                    label = f"{int(row['laser3_percent'])}% - {value:.2f}"
                else:
                    label = f"{value:.2f}"

                ax.text(
                    j + 0.5, y0 + sub_h / 2,
                    label,
                    ha="center", va="center",
                    color="white",
                    fontsize=16,
                    fontweight="bold"
                )

# =========================
# AXES FORMATTING
# =========================
ax.set_xlim(0, len(laser_order))
ax.set_ylim(0, len(frame_order))
ax.invert_yaxis()  # so row 4 is at the top

ax.set_xticks(np.arange(len(laser_order)) + 0.5)
ax.set_yticks(np.arange(len(frame_order)) + 0.5)

ax.set_xticklabels(laser_order, fontsize=16, fontweight="bold")
ax.set_yticklabels(frame_order, fontsize=16, fontweight="bold")

ax.set_xlabel(
    "Number of laser lines",
    fontsize=16,
    fontweight="bold"
)
ax.set_ylabel(
    "Number of bleaching frames",
    fontsize=16,
    fontweight="bold"
)
ax.set_title(
    "Normalized post-bleach intensity matrix",
    fontsize=16,
    fontweight="bold"
)

# remove default frame
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(length=0)

# =========================
# COLORBAR
# =========================
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(
    "Normalized post-bleach intensity",
    fontsize=12
)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

plot_path = folder / "normalized_postbleach_matrix_heatmap_subdivided.png"

plt.savefig(
    plot_path,
    dpi=1200,
    bbox_inches="tight",
    transparent=True
)

plt.show()

print(f"\nSaved heatmap to:\n{plot_path}")

