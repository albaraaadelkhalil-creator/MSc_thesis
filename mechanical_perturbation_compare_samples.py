#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare selected GUV mechanical-deformation outputs across experiments.

Compatible with the ablation analysis outputs:
- global_metrics_relative_to_prebleach.csv
- regional_metrics_relative_to_prebleach.csv

Generated plots:
1. Global contour solidity.
2. Area and perimeter change relative to prebleach.
3. Mean absolute radial displacement: bleached vs non-bleached.
4. Mean absolute local strain: bleached vs non-bleached.

All plots are saved without being shown.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# The path should be the output of the final_ablation_analysis.py: guv_deformation_analysis
# update the names of the experiemnt per file to add the name in the image legend.
# This code loops over the experiments provided so you can add more paths in the experiment array and the code will process them!
# Finally, check the output folder
# =============================================================================

EXPERIMENTS = [
    {
        "name": "4 lasers, 4 bleaching frames",
        "folder": Path(
            r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging"
            r"\FRAP 4 lasers\guv_deformation_analysis"
        ),
    },
    {
        "name": "2 lasers, 3 bleaching frames",
        "folder": Path(
            r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging"
            r"\FRAP_2_Lasers_Line_3_frames\guv_deformation_analysis"
        ),
    },
    {
        "name": "3 lasers, 2 bleaching frames",
        "folder": Path(
            r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging"
            r"\FRAP 3 lasers 2 frames_2\guv_deformation_analysis"
        ),
    },
]

OUTPUT_FOLDER = Path(
    r"C:\Master End Project\Images\FRAP\ablation_comparison_3_GUVs"
)

PREFERRED_LENGTH_UNIT = "um"  # "um" or "px"
CROP_TO_SHORTEST_EXPERIMENT = True
SAVE_PDF = True
SAVE_LEGENDS_SEPARATELY = True

DPI = 1200
FONT_SIZE = 16
FIGSIZE_SINGLE = (8.0, 6.0)
FIGSIZE_WIDE = (10.5, 6.5)
LINEWIDTH = 2.5
MARKER_SIZE = 6
LEGEND_NCOL = 3
LEGEND_FIGSIZE = (11.0, 1.8)

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)


# =============================================================================
# FILE LOADING
# =============================================================================

def ensure_output_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_file(folder: Path, candidates: list[str]) -> Optional[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    lookup = {normalize_name(file.stem): file for file in files}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]

    for candidate in candidates:
        key = normalize_name(candidate)
        for file_key, file in lookup.items():
            if key in file_key:
                return file

    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table type: {path.suffix}")


def load_experiment(experiment: dict) -> dict:
    folder = Path(experiment["folder"])

    global_path = find_file(
        folder,
        [
            "global_metrics_relative_to_prebleach",
            "global_metrics",
        ],
    )
    regional_path = find_file(
        folder,
        [
            "regional_metrics_relative_to_prebleach",
            "local_bleached_region_metrics",
            "local_metrics",
        ],
    )

    if global_path is None:
        raise FileNotFoundError(f"Global metrics file not found in {folder}")
    if regional_path is None:
        raise FileNotFoundError(f"Regional metrics file not found in {folder}")

    global_df = read_table(global_path).copy()
    regional_df = read_table(regional_path).copy()

    global_df["experiment"] = experiment["name"]
    regional_df["experiment"] = experiment["name"]

    return {
        "name": experiment["name"],
        "global": global_df,
        "regional": regional_df,
    }


# =============================================================================
# COLUMN AND FRAME HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        "None of the expected columns were found:\n"
        + "\n".join(f"  - {column}" for column in candidates)
        + "\n\nAvailable columns:\n"
        + "\n".join(f"  - {column}" for column in dataframe.columns)
    )


def x_column(dataframe: pd.DataFrame) -> str:
    if "time_s" in dataframe.columns:
        return "time_s"
    if "frame" in dataframe.columns:
        return "frame"
    raise KeyError("Expected either 'time_s' or 'frame'.")


def x_label(dataframe: pd.DataFrame) -> str:
    return "Time (s)" if x_column(dataframe) == "time_s" else "Frame"


def crop_to_shortest(experiments: list[dict]) -> list[dict]:
    if not CROP_TO_SHORTEST_EXPERIMENT:
        return experiments

    if all(
        "frame" in experiment[key].columns
        for experiment in experiments
        for key in ("global", "regional")
    ):
        last_frames = []
        for experiment in experiments:
            global_last = int(pd.to_numeric(experiment["global"]["frame"], errors="coerce").max())
            regional_last = int(pd.to_numeric(experiment["regional"]["frame"], errors="coerce").max())
            last_frames.append(min(global_last, regional_last))

        common_last = min(last_frames)
        cropped = []

        for experiment in experiments:
            updated = experiment.copy()
            for key in ("global", "regional"):
                frame_values = pd.to_numeric(experiment[key]["frame"], errors="coerce")
                updated[key] = experiment[key].loc[frame_values <= common_last].copy().reset_index(drop=True)
            cropped.append(updated)

        print(f"Cropped all experiments to frames 0-{common_last}.")
        return cropped

    common_rows = min(
        len(experiment[key])
        for experiment in experiments
        for key in ("global", "regional")
    )

    cropped = []
    for experiment in experiments:
        updated = experiment.copy()
        updated["global"] = experiment["global"].head(common_rows).copy().reset_index(drop=True)
        updated["regional"] = experiment["regional"].head(common_rows).copy().reset_index(drop=True)
        cropped.append(updated)

    print(f"Cropped all experiments to the first {common_rows} rows.")
    return cropped


# =============================================================================
# PLOT FORMATTING
# =============================================================================

def bold_axis_text(axis: plt.Axes) -> None:
    axis.title.set_fontsize(FONT_SIZE)
    axis.title.set_fontweight("bold")
    axis.xaxis.label.set_size(FONT_SIZE)
    axis.xaxis.label.set_weight("bold")
    axis.yaxis.label.set_size(FONT_SIZE)
    axis.yaxis.label.set_weight("bold")
    axis.tick_params(axis="both", labelsize=FONT_SIZE)

    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")


def add_bleach_line(axis: plt.Axes, experiments: list[dict]) -> None:
    times = []

    for experiment in experiments:
        dataframe = experiment["global"]
        if "phase" not in dataframe.columns or "time_s" not in dataframe.columns:
            continue

        postbleach = dataframe[
            dataframe["phase"].astype(str).str.lower().eq("postbleach")
        ]
        if not postbleach.empty:
            times.append(float(postbleach["time_s"].iloc[0]))

    if times:
        axis.axvline(
            min(times),
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            label="First postbleach frame",
        )


def save_figure(figure: plt.Figure, filename_stem: str) -> None:
    png_path = OUTPUT_FOLDER / f"{filename_stem}.png"
    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        transparent=True,
    )

    if SAVE_PDF:
        figure.savefig(
            OUTPUT_FOLDER / f"{filename_stem}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    plt.close(figure)
    print(f"Saved: {png_path}")


def save_legend(handles, labels, filename_stem: str) -> None:
    unique_handles = []
    unique_labels = []
    seen = set()

    for handle, label in zip(handles, labels):
        if label in seen or str(label).startswith("_"):
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
        seen.add(label)

    if not unique_handles:
        return

    ncol = min(LEGEND_NCOL, len(unique_handles))
    nrows = int(np.ceil(len(unique_handles) / ncol))
    height = max(LEGEND_FIGSIZE[1], 0.5 + 0.45 * nrows)

    figure = plt.figure(figsize=(LEGEND_FIGSIZE[0], height))
    legend = figure.legend(
        unique_handles,
        unique_labels,
        loc="center",
        frameon=False,
        ncol=ncol,
        handlelength=2.5,
        columnspacing=1.6,
    )

    for text in legend.get_texts():
        text.set_fontsize(FONT_SIZE)
        text.set_fontweight("bold")

    figure.tight_layout()
    save_figure(figure, f"{filename_stem}_legend")


def finish_plot(figure: plt.Figure, axis: plt.Axes, filename_stem: str) -> None:
    handles, labels = axis.get_legend_handles_labels()

    if SAVE_LEGENDS_SEPARATELY:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    bold_axis_text(axis)
    figure.tight_layout()
    save_figure(figure, filename_stem)

    if SAVE_LEGENDS_SEPARATELY:
        save_legend(handles, labels, filename_stem)


# =============================================================================
# REQUESTED PLOTS
# =============================================================================

def plot_solidity(experiments: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    all_values = []

    for experiment in experiments:
        dataframe = experiment["global"]
        xcol = x_column(dataframe)
        ycol = first_existing_column(
            dataframe,
            ["solidity", "solidity_polygon", "solidity_regionprops"],
        )
        values = pd.to_numeric(dataframe[ycol], errors="coerce")
        all_values.extend(values.dropna().tolist())

        axis.plot(
            dataframe[xcol],
            values,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            label=experiment["name"],
        )

    add_bleach_line(axis, experiments)
    axis.set_xlabel(x_label(experiments[0]["global"]))
    axis.set_ylabel("Solidity")
    axis.set_title("Global contour solidity")
    axis.grid(alpha=0.25)

    if all_values:
        lower = float(np.nanmin(all_values))
        upper = float(np.nanmax(all_values))
        margin = max(0.15 * (upper - lower), 0.005)
        axis.set_ylim(lower - margin, upper + margin)

    if not SAVE_LEGENDS_SEPARATELY:
        axis.legend(frameon=False)

    finish_plot(figure, axis, "comparison_global_contour_solidity")


def plot_area_perimeter_change(experiments: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=FIGSIZE_WIDE)

    for experiment in experiments:
        dataframe = experiment["global"]
        xcol = x_column(dataframe)
        area_col = first_existing_column(
            dataframe,
            ["area_change_vs_prebleach", "area_strain_vs_prebleach"],
        )
        perimeter_col = first_existing_column(
            dataframe,
            ["perimeter_change_vs_prebleach", "perimeter_strain_vs_prebleach"],
        )

        axis.plot(
            dataframe[xcol],
            100 * pd.to_numeric(dataframe[area_col], errors="coerce"),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            label=f"{experiment['name']} area",
        )
        axis.plot(
            dataframe[xcol],
            100 * pd.to_numeric(dataframe[perimeter_col], errors="coerce"),
            marker="s",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=f"{experiment['name']} perimeter",
        )

    add_bleach_line(axis, experiments)
    axis.axhline(0, linewidth=1.3, alpha=0.75)
    axis.set_xlabel(x_label(experiments[0]["global"]))
    axis.set_ylabel("Change relative to prebleach (%)")
    axis.set_title("Area and perimeter change relative to prebleach")
    axis.grid(alpha=0.25)

    if not SAVE_LEGENDS_SEPARATELY:
        axis.legend(frameon=False, ncol=2)

    finish_plot(
        figure,
        axis,
        "comparison_area_perimeter_change_relative_to_prebleach",
    )


def plot_radial_displacement(experiments: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=FIGSIZE_WIDE)
    preferred = "um" if PREFERRED_LENGTH_UNIT.lower() == "um" else "px"
    plotted_unit = "µm" if preferred == "um" else "px"

    for experiment in experiments:
        dataframe = experiment["regional"]
        xcol = x_column(dataframe)

        bleached_col = first_existing_column(
            dataframe,
            [
                f"bleached_mean_abs_radial_displacement_{preferred}",
                f"bleached_abs_mean_radial_displacement_{preferred}",
                "bleached_mean_abs_radial_displacement_um",
                "bleached_abs_mean_radial_displacement_um",
                "bleached_mean_abs_radial_displacement_px",
                "bleached_abs_mean_radial_displacement_px",
            ],
        )
        nonbleached_col = first_existing_column(
            dataframe,
            [
                f"nonbleached_mean_abs_radial_displacement_{preferred}",
                f"nonbleached_abs_mean_radial_displacement_{preferred}",
                "nonbleached_mean_abs_radial_displacement_um",
                "nonbleached_abs_mean_radial_displacement_um",
                "nonbleached_mean_abs_radial_displacement_px",
                "nonbleached_abs_mean_radial_displacement_px",
            ],
        )

        plotted_unit = "µm" if bleached_col.endswith("_um") else "px"

        axis.plot(
            dataframe[xcol],
            pd.to_numeric(dataframe[bleached_col], errors="coerce"),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            label=f"{experiment['name']} bleached",
        )
        axis.plot(
            dataframe[xcol],
            pd.to_numeric(dataframe[nonbleached_col], errors="coerce"),
            marker="s",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=f"{experiment['name']} non-bleached",
        )

    add_bleach_line(axis, experiments)
    axis.set_xlabel(x_label(experiments[0]["regional"]))
    axis.set_ylabel(f"Mean absolute radial displacement ({plotted_unit})")
    axis.set_title("Mean absolute radial displacement: bleached vs non-bleached")
    axis.grid(alpha=0.25)

    if not SAVE_LEGENDS_SEPARATELY:
        axis.legend(frameon=False, ncol=2)

    finish_plot(
        figure,
        axis,
        "comparison_mean_absolute_radial_displacement",
    )


def plot_local_strain(experiments: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=FIGSIZE_WIDE)

    for experiment in experiments:
        dataframe = experiment["regional"]
        xcol = x_column(dataframe)

        bleached_col = first_existing_column(
            dataframe,
            ["bleached_mean_abs_local_strain", "bleached_abs_mean_segment_strain"],
        )
        nonbleached_col = first_existing_column(
            dataframe,
            ["nonbleached_mean_abs_local_strain", "nonbleached_abs_mean_segment_strain"],
        )

        axis.plot(
            dataframe[xcol],
            100 * pd.to_numeric(dataframe[bleached_col], errors="coerce"),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            label=f"{experiment['name']} bleached",
        )
        axis.plot(
            dataframe[xcol],
            100 * pd.to_numeric(dataframe[nonbleached_col], errors="coerce"),
            marker="s",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=f"{experiment['name']} non-bleached",
        )

    add_bleach_line(axis, experiments)
    axis.set_xlabel(x_label(experiments[0]["regional"]))
    axis.set_ylabel("Mean absolute local strain (%)")
    axis.set_title("Mean absolute local strain: bleached vs non-bleached")
    axis.grid(alpha=0.25)

    if not SAVE_LEGENDS_SEPARATELY:
        axis.legend(frameon=False, ncol=2)

    finish_plot(
        figure,
        axis,
        "comparison_mean_absolute_local_strain",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_output_folder(OUTPUT_FOLDER)

    experiments = [load_experiment(experiment) for experiment in EXPERIMENTS]
    experiments = crop_to_shortest(experiments)

    plot_solidity(experiments)
    plot_area_perimeter_change(experiments)
    plot_radial_displacement(experiments)
    plot_local_strain(experiments)

    print("\nFinished requested comparison plots.")
    print(f"Output folder:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
