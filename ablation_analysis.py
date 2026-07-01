#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code created by Albaraa Adel Farouk Khalil
LLM ChatGPT have been used as an assist to develope this code

Last updated: 27.06.2026

GUV photobleaching-deformation analysis.

Calculated in this version
1. Displacement-vector quiver plots relative to the mean prebleach contour.
2. Area relative to the mean prebleach area.
3. Perimeter relative to the mean prebleach perimeter.
4. Solidity.
5. Circularity.
6. Mean absolute radial displacement relative to the mean prebleach contour.
7. Mean absolute local segment strain relative to the mean prebleach contour.

The bleached region is detected from the first postbleach frame. Contours are
then divided into bleached and non-bleached arcs and resampled with a fixed
number of points in each region so that contour indices remain comparable
between frames.

All generated plots use 16-point bold text and are saved at 600 dpi.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from skimage.draw import disk
from PIL import Image, ImageDraw


# =============================================================================
# USER SETTINGS
# Add your tif file of a GUV corrected for translation and rotation drift using stackreg in imagej
# Add snakes of the tif file extracted via Jfilament in imageJ 
# update Pixel_size_um according to your tif file properties. 
# =============================================================================

TIFF_PATH = Path(
    r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging\FRAP_3_Lasers_line_4_frames\33_frames_trans_rot_corr.tif"
)

CONTOUR_PATH = Path(
    r"C:\Master End Project\Images\FRAP\26.03.06_optimizing_frap_imaging\FRAP_3_Lasers_line_4_frames\33_frames_trans_rot_corr.snakes"
)

OUTPUT_FOLDER = TIFF_PATH.parent / "guv_deformation_analysis"

# Set to None to retain pixel units only.
PIXEL_SIZE_UM = 0.2083089

# Number of frames acquired before bleaching.
PREBLEACH_FRAMES = 5

# -------------------------------------------------------------------------
# Time axis
# -------------------------------------------------------------------------
# Allowed values:
# "constant_post": constant postbleach frame interval
# "blocks":        multiple postbleach acquisition blocks
# "explicit":      supply one time value per frame
TIME_MODE = "blocks"

PREBLEACH_DT_S = 0.653
POSTBLEACH_DT_S = 0.653

POSTBLEACH_BLOCKS = [
    ("Pb1", 15, 0.653),
    ("Pb2", 10, 5.0),
    ("Pb3", 30, 10.0),
]

EXPLICIT_TIME_AXIS_S = None

# -------------------------------------------------------------------------
# Contour preprocessing
# -------------------------------------------------------------------------
N_CONTOUR_POINTS = 720
START_AT_TOP = True
FORCE_CLOCKWISE = False

# -------------------------------------------------------------------------
# Bleached-region detection
# -------------------------------------------------------------------------
BLEACHED_REFERENCE_FRAME = PREBLEACH_FRAMES

# "per_frame": use the fixed reference-frame intensity threshold to detect the
# bleached arc independently in every postbleach frame.
# "fixed": use the arc detected in the reference frame for all frames.
BLEACHED_REGION_MODE = "per_frame"
BLEACHED_PERCENTILE = 10
INTENSITY_BAND_RADIUS_PX = 2
BLEACHED_SMOOTHING_WINDOW = 21
MIN_BLEACHED_ARC_POINTS = 20
BLEACHED_FILL_GAPS_POINTS = 10
BLEACHED_REMOVE_SMALL_OBJECTS_POINTS = 8
EXPAND_BLEACHED_ARC_POINTS = 0

# Set True to bypass automatic detection.
USE_MANUAL_BLEACHED_ARC = False
MANUAL_ARC_START_INDEX = 500
MANUAL_ARC_END_INDEX = 550

# -------------------------------------------------------------------------
# Output and plot settings
# -------------------------------------------------------------------------
SHOW_PLOTS = False
PLOT_DPI = 1200
FONT_SIZE = 16

QUIVER_EVERY_N_POINTS = 20
QUIVER_ARROW_DISPLAY_SCALE = 1.0

# Quiver appearance
QUIVER_CONTOUR_LINEWIDTH = 5
QUIVER_CONTOUR_ALPHA = 0.55
QUIVER_ARROW_WIDTH = 0.012
QUIVER_ARROW_HEADWIDTH = 7.0
QUIVER_ARROW_HEADLENGTH = 8.0
QUIVER_ARROW_HEADAXISLENGTH = 7.0
QUIVER_ARROW_COLOR = "crimson"
QUIVER_ARROW_EDGE_COLOR = "black"
QUIVER_BLEACHED_POINT_COLOR = "blue"

# Retained quality-control outputs
SAVE_BLEACHED_DETECTION_PROFILES_ALL_FRAMES = True
SAVE_MAPPING_POINTS_OVERLAY = True
MAPPING_POINT_RADIUS = 1
MAPPING_LABEL_EVERY_N_POINTS = 30

# Plot one early postbleach frame and the last available frame.
QUIVER_FIRST_FRAME = PREBLEACH_FRAMES
QUIVER_LAST_FRAME = None

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
# GENERAL UTILITIES
# =============================================================================

def ensure_output_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_tiff_stack(path: Path) -> np.ndarray:
    """Load a 2D TIFF or 3D TIFF stack as [frame, y, x]."""
    image = np.squeeze(tiff.imread(str(path)))

    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a 3D [frame, y, x] TIFF stack, got {image.shape}."
        )

    return image.astype(float)


def build_time_axis(n_frames: int) -> np.ndarray:
    """Construct the acquisition time for every frame."""
    if PREBLEACH_FRAMES < 1:
        raise ValueError("PREBLEACH_FRAMES must be at least 1.")

    if TIME_MODE == "explicit":
        if EXPLICIT_TIME_AXIS_S is None:
            raise ValueError(
                "EXPLICIT_TIME_AXIS_S must be provided when TIME_MODE='explicit'."
            )
        time = np.asarray(EXPLICIT_TIME_AXIS_S, dtype=float)
        if len(time) != n_frames:
            raise ValueError(
                f"Explicit time axis has {len(time)} values but the TIFF has "
                f"{n_frames} frames."
            )
        return time

    time = np.zeros(n_frames, dtype=float)

    for frame in range(1, min(PREBLEACH_FRAMES, n_frames)):
        time[frame] = time[frame - 1] + PREBLEACH_DT_S

    if TIME_MODE == "constant_post":
        for frame in range(PREBLEACH_FRAMES, n_frames):
            time[frame] = time[frame - 1] + POSTBLEACH_DT_S
        return time

    if TIME_MODE == "blocks":
        frame = PREBLEACH_FRAMES
        current_time = time[PREBLEACH_FRAMES - 1]

        for _, block_count, block_dt in POSTBLEACH_BLOCKS:
            for _ in range(block_count):
                if frame >= n_frames:
                    break
                current_time += block_dt
                time[frame] = current_time
                frame += 1

        if frame < n_frames:
            raise ValueError(
                "POSTBLEACH_BLOCKS do not cover all TIFF frames. "
                f"Covered {frame} of {n_frames} frames."
            )

        return time

    raise ValueError(
        "TIME_MODE must be 'constant_post', 'blocks', or 'explicit'."
    )


# =============================================================================
# CONTOUR LOADING AND STANDARDIZATION
# =============================================================================

def parse_single_contour_txt(path: Path) -> dict[int, np.ndarray]:
    """
    Parse a JFilament-like contour file.

    Expected numeric rows:
        frame_id point_id x y z

    Returns
    -------
    dict
        Zero-based frame index -> array with columns [x, y].
    """
    numeric_rows: list[tuple[int, int, float, float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            try:
                frame_id = int(float(parts[0]))
                point_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue

            numeric_rows.append((frame_id, point_id, x, y))

    if not numeric_rows:
        raise ValueError(f"No contour points were found in {path}.")

    dataframe = pd.DataFrame(
        numeric_rows,
        columns=["frame_id", "point_id", "x", "y"],
    )

    contours: dict[int, np.ndarray] = {}
    for frame_id, subset in dataframe.groupby("frame_id"):
        subset = subset.sort_values("point_id")
        contours[int(frame_id) - 1] = subset[["x", "y"]].to_numpy(float)

    return contours


def load_contours(path: Path) -> dict[int, np.ndarray]:
    """Load one multi-frame contour file or a folder of contour text files."""
    path = Path(path)

    if path.is_file():
        return parse_single_contour_txt(path)

    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt contour files found in {path}.")

        contours: dict[int, np.ndarray] = {}
        for file_index, file in enumerate(files):
            parsed = parse_single_contour_txt(file)

            if len(parsed) == 1:
                contours[file_index] = next(iter(parsed.values()))
            else:
                contours.update(parsed)

        return contours

    raise FileNotFoundError(f"Contour path does not exist: {path}")


def close_contour(contour: np.ndarray) -> np.ndarray:
    contour = np.asarray(contour, dtype=float)

    if len(contour) < 3:
        raise ValueError("A contour requires at least three points.")

    if np.linalg.norm(contour[0] - contour[-1]) > 1e-6:
        contour = np.vstack([contour, contour[0]])

    return contour


def polygon_signed_area(contour: np.ndarray) -> float:
    closed = close_contour(contour)
    x = closed[:, 0]
    y = closed[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def resample_closed_contour(
    contour: np.ndarray,
    n_points: int = N_CONTOUR_POINTS,
) -> np.ndarray:
    """Resample a closed contour to equally spaced arc-length positions."""
    closed = close_contour(contour)

    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length <= 0:
        raise ValueError("Contour length is zero.")

    x_interpolator = interp1d(cumulative_length, closed[:, 0], kind="linear")
    y_interpolator = interp1d(cumulative_length, closed[:, 1], kind="linear")

    new_positions = np.linspace(0.0, total_length, n_points + 1)[:-1]

    return np.column_stack(
        [
            x_interpolator(new_positions),
            y_interpolator(new_positions),
        ]
    )


def orient_and_start_contour(contour: np.ndarray) -> np.ndarray:
    """Enforce a consistent direction and start the contour at its top."""
    contour = np.asarray(contour, dtype=float)
    clockwise = polygon_signed_area(contour) > 0

    if FORCE_CLOCKWISE and not clockwise:
        contour = contour[::-1]
    elif not FORCE_CLOCKWISE and clockwise:
        contour = contour[::-1]

    if START_AT_TOP:
        minimum_y = np.min(contour[:, 1])
        candidates = np.where(np.isclose(contour[:, 1], minimum_y, atol=1.0))[0]

        if len(candidates) > 1:
            median_x = np.median(contour[:, 0])
            start_index = candidates[
                np.argmin(np.abs(contour[candidates, 0] - median_x))
            ]
        else:
            start_index = candidates[0]

        contour = np.roll(contour, -int(start_index), axis=0)

    return contour


def preprocess_contours(
    contours: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Resample and orient every loaded contour."""
    processed: dict[int, np.ndarray] = {}

    for frame, contour in contours.items():
        processed[frame] = orient_and_start_contour(
            resample_closed_contour(contour, N_CONTOUR_POINTS)
        )

    return processed


# =============================================================================
# BLEACHED-REGION DETECTION AND REGION-AWARE RESAMPLING
# =============================================================================

def sample_image_at_points(
    image: np.ndarray,
    points: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Sample the mean image intensity around each contour point."""
    x = points[:, 0]
    y = points[:, 1]

    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx**2 + dy**2 <= radius**2:
                offsets.append((dy, dx))

    sampled = []
    for dy, dx in offsets:
        coordinates = np.vstack([y + dy, x + dx])
        sampled.append(
            map_coordinates(
                image,
                coordinates,
                order=1,
                mode="nearest",
            )
        )

    return np.mean(np.vstack(sampled), axis=0)


def circular_smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a circular moving-average filter."""
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    if window >= len(values):
        window = len(values) - 1 if len(values) % 2 == 0 else len(values) - 2

    window = max(window, 3)
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window, dtype=float) / window

    return np.convolve(padded, kernel, mode="valid")


def fill_small_false_gaps_circular(
    mask: np.ndarray,
    max_gap_size: int,
) -> np.ndarray:
    """Fill short False gaps located between True circular segments."""
    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0 or np.all(mask) or not np.any(mask):
        return mask.copy()

    filled = mask.copy()
    doubled = np.concatenate([mask, mask])
    n = len(mask)

    index = 0
    while index < 2 * n:
        if doubled[index]:
            index += 1
            continue

        end = index
        while end < 2 * n and not doubled[end]:
            end += 1

        gap_length = end - index
        left_true = doubled[index - 1] if index > 0 else doubled[-1]
        right_true = doubled[end] if end < 2 * n else doubled[0]

        if gap_length <= max_gap_size and left_true and right_true:
            filled[np.arange(index, end) % n] = True

        index = end

    return filled


def remove_small_true_segments_circular(
    mask: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Remove short isolated True segments from a circular Boolean mask."""
    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0 or np.all(mask) or not np.any(mask):
        return mask.copy()

    cleaned = mask.copy()
    doubled = np.concatenate([mask, mask])
    n = len(mask)
    processed: set[int] = set()

    index = 0
    while index < 2 * n:
        if not doubled[index]:
            index += 1
            continue

        end = index
        while end < 2 * n and doubled[end]:
            end += 1

        segment_indices = np.arange(index, end) % n
        segment_key = int(segment_indices[0])

        if segment_key not in processed and len(segment_indices) < min_size:
            cleaned[segment_indices] = False

        processed.update(int(item) for item in segment_indices)
        index = end

    return cleaned


def find_longest_circular_segment(mask: np.ndarray) -> np.ndarray:
    """Return indices belonging to the longest contiguous True circular segment."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)

    if np.all(mask):
        return np.arange(n)

    if not np.any(mask):
        return np.array([], dtype=int)

    doubled = np.concatenate([mask, mask])
    best_start = 0
    best_length = 0
    index = 0

    while index < 2 * n:
        if not doubled[index]:
            index += 1
            continue

        end = index
        while end < 2 * n and doubled[end]:
            end += 1

        segment_length = end - index
        if index < n and segment_length > best_length:
            best_start = index
            best_length = min(segment_length, n)

        index = end

    return np.arange(best_start, best_start + best_length) % n


def sort_circular_arc_indices(indices: np.ndarray, n_points: int) -> np.ndarray:
    """Order selected indices continuously around a circular contour."""
    indices = np.sort(np.unique(np.asarray(indices, dtype=int)))

    if len(indices) <= 1:
        return indices

    gaps = np.diff(np.concatenate([indices, [indices[0] + n_points]]))
    largest_gap_position = int(np.argmax(gaps))
    start = (largest_gap_position + 1) % len(indices)

    return np.concatenate([indices[start:], indices[:start]])


def expand_circular_indices(
    indices: np.ndarray,
    n_points: int,
    expansion: int,
) -> np.ndarray:
    """Expand selected circular indices by a fixed number of points."""
    expanded = set(int(index) for index in indices)

    for index in indices:
        for offset in range(1, expansion + 1):
            expanded.add((int(index) - offset) % n_points)
            expanded.add((int(index) + offset) % n_points)

    return sort_circular_arc_indices(
        np.asarray(sorted(expanded), dtype=int),
        n_points,
    )


def manual_bleached_arc(n_points: int) -> np.ndarray:
    """Return the manually defined bleached arc."""
    if MANUAL_ARC_START_INDEX <= MANUAL_ARC_END_INDEX:
        indices = np.arange(
            MANUAL_ARC_START_INDEX,
            MANUAL_ARC_END_INDEX + 1,
        )
    else:
        indices = np.concatenate(
            [
                np.arange(MANUAL_ARC_START_INDEX, n_points),
                np.arange(0, MANUAL_ARC_END_INDEX + 1),
            ]
        )

    return sort_circular_arc_indices(indices, n_points)


def detect_bleached_arc(
    image: np.ndarray,
    contour: np.ndarray,
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Detect the darkest continuous contour segment in one postbleach frame.

    Returns the selected arc, raw profile, smoothed profile, and threshold.
    """
    if USE_MANUAL_BLEACHED_ARC:
        arc = manual_bleached_arc(len(contour))
        raw_profile = sample_image_at_points(
            image,
            contour,
            radius=INTENSITY_BAND_RADIUS_PX,
        )
        smoothed_profile = circular_smooth_1d(
            raw_profile,
            BLEACHED_SMOOTHING_WINDOW,
        )
        threshold = float(np.percentile(smoothed_profile, BLEACHED_PERCENTILE))
        return arc, raw_profile, smoothed_profile, threshold

    raw_profile = sample_image_at_points(
        image,
        contour,
        radius=INTENSITY_BAND_RADIUS_PX,
    )
    smoothed_profile = circular_smooth_1d(
        raw_profile,
        BLEACHED_SMOOTHING_WINDOW,
    )

    if fixed_threshold is None:
        threshold = float(
            np.percentile(smoothed_profile, BLEACHED_PERCENTILE)
        )
    else:
        threshold = float(fixed_threshold)

    low_mask = smoothed_profile <= threshold

    low_mask = fill_small_false_gaps_circular(
        low_mask,
        BLEACHED_FILL_GAPS_POINTS,
    )
    low_mask = remove_small_true_segments_circular(
        low_mask,
        BLEACHED_REMOVE_SMALL_OBJECTS_POINTS,
    )

    arc = find_longest_circular_segment(low_mask)

    if len(arc) < MIN_BLEACHED_ARC_POINTS:
        centre = int(np.argmin(smoothed_profile))
        half_width = MIN_BLEACHED_ARC_POINTS // 2
        arc = (
            np.arange(centre - half_width, centre + half_width + 1)
            % len(contour)
        )

    arc = expand_circular_indices(
        arc,
        len(contour),
        EXPAND_BLEACHED_ARC_POINTS,
    )

    return arc, raw_profile, smoothed_profile, threshold


def complement_arc_indices(
    arc_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Return the ordered complement of one circular arc."""
    selected = set(int(index) for index in arc_indices)

    if not selected:
        return np.arange(n_points)

    if len(selected) == n_points:
        return np.array([], dtype=int)

    ordered_arc = sort_circular_arc_indices(arc_indices, n_points)
    start = (int(ordered_arc[-1]) + 1) % n_points

    complement = []
    for offset in range(n_points):
        index = (start + offset) % n_points
        if index not in selected:
            complement.append(index)

    return np.asarray(complement, dtype=int)


def resample_open_curve(
    points: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Resample an open curve to equally spaced arc-length positions."""
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        raise ValueError("Cannot resample an empty curve.")

    if len(points) == 1:
        return np.repeat(points, n_points, axis=0)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length <= 0:
        return np.repeat(points[:1], n_points, axis=0)

    x_interpolator = interp1d(cumulative_length, points[:, 0], kind="linear")
    y_interpolator = interp1d(cumulative_length, points[:, 1], kind="linear")
    new_positions = np.linspace(0.0, total_length, n_points)

    return np.column_stack(
        [
            x_interpolator(new_positions),
            y_interpolator(new_positions),
        ]
    )


def region_aware_resample(
    contour: np.ndarray,
    bleached_indices: np.ndarray,
    n_bleached_points: int,
    n_nonbleached_points: int,
) -> np.ndarray:
    """
    Resample bleached and non-bleached arcs separately.

    Output order:
        0:n_bleached_points                   bleached region
        n_bleached_points:N_CONTOUR_POINTS   non-bleached region
    """
    bleached_indices = sort_circular_arc_indices(
        bleached_indices,
        len(contour),
    )
    nonbleached_indices = complement_arc_indices(
        bleached_indices,
        len(contour),
    )

    bleached_points = contour[bleached_indices]
    nonbleached_points = contour[nonbleached_indices]

    return np.vstack(
        [
            resample_open_curve(bleached_points, n_bleached_points),
            resample_open_curve(nonbleached_points, n_nonbleached_points),
        ]
    )


# =============================================================================
# METRICS RELATIVE TO THE MEAN PREBLEACH REFERENCE
# =============================================================================

def polygon_area(contour: np.ndarray) -> float:
    return abs(polygon_signed_area(contour))


def polygon_perimeter(contour: np.ndarray) -> float:
    closed = close_contour(contour)
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return np.nan
    return float(4.0 * np.pi * area / perimeter**2)


def solidity(contour: np.ndarray, area: float) -> float:
    """Contour area divided by convex-hull area."""
    if len(contour) < 3:
        return np.nan

    hull = ConvexHull(contour)
    hull_area = float(hull.volume)

    if hull_area <= 0:
        return np.nan

    return float(area / hull_area)


def contour_centroid(contour: np.ndarray) -> np.ndarray:
    return np.mean(contour, axis=0)


def radial_unit_vectors(reference_contour: np.ndarray) -> np.ndarray:
    """Unit vectors from the reference-contour centroid to each point."""
    centre = contour_centroid(reference_contour)
    vectors = reference_contour - centre
    magnitudes = np.linalg.norm(vectors, axis=1)
    magnitudes[magnitudes == 0] = 1.0

    return vectors / magnitudes[:, None]


def signed_radial_displacement(
    contour: np.ndarray,
    reference_contour: np.ndarray,
    radial_units: np.ndarray,
) -> np.ndarray:
    """Project pointwise contour displacement onto reference radial vectors."""
    displacement = contour - reference_contour
    return np.sum(displacement * radial_units, axis=1)


def contour_segment_lengths(
    contour: np.ndarray,
    split_index: int,
) -> np.ndarray:
    """
    Calculate local segment lengths without joining the bleached and
    non-bleached blocks across their artificial resampling boundary.
    """
    contour = np.asarray(contour, dtype=float)
    n_points = len(contour)
    lengths = np.full(n_points, np.nan, dtype=float)

    for start, end in [(0, split_index), (split_index, n_points)]:
        region = contour[start:end]

        if len(region) < 2:
            continue

        region_lengths = np.linalg.norm(np.diff(region, axis=0), axis=1)
        lengths[start : end - 1] = region_lengths

        # The last point of each open region has no forward internal segment.
        lengths[end - 1] = np.nan

    return lengths


def unit_convert(value: float | np.ndarray, dimension: str):
    """Convert pixel lengths or areas to micrometres when calibrated."""
    if PIXEL_SIZE_UM is None:
        return value

    if dimension == "length":
        return value * PIXEL_SIZE_UM

    if dimension == "area":
        return value * PIXEL_SIZE_UM**2

    raise ValueError("dimension must be 'length' or 'area'.")


def region_absolute_mean(
    values: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Mean magnitude within one contour region."""
    return float(np.nanmean(np.abs(values[indices])))


# =============================================================================
# PLOT HELPERS
# =============================================================================

def make_text_bold(ax: plt.Axes) -> None:
    """Enforce 16-point bold text on one Matplotlib axis."""
    ax.title.set_fontsize(FONT_SIZE)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_size(FONT_SIZE)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(FONT_SIZE)
    ax.yaxis.label.set_weight("bold")

    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZE)
            text.set_fontweight("bold")


def add_bleach_marker(ax: plt.Axes, first_postbleach_time: float) -> None:
    """Mark the start of the postbleach period."""
    ax.axvline(
        first_postbleach_time,
        linestyle="--",
        linewidth=1.5,
        label="First postbleach frame",
    )


def save_figure(
    figure: plt.Figure,
    output_name: str,
) -> None:
    """Save and then show or close a figure."""
    figure.savefig(
        OUTPUT_FOLDER / output_name,
        dpi=PLOT_DPI,
        bbox_inches="tight",
        transparent=True,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)


def plot_single_metric(
    dataframe: pd.DataFrame,
    y_column: str,
    ylabel: str,
    title: str,
    output_name: str,
    first_postbleach_time: float,
    horizontal_zero: bool = False,
) -> None:
    """Create one time-series plot with consistent formatting."""
    figure, axis = plt.subplots(figsize=(8, 6))

    axis.plot(
        dataframe["time_s"],
        dataframe[y_column],
        marker="o",
        linewidth=2,
    )

    if horizontal_zero:
        axis.axhline(0.0, linestyle="--", linewidth=1.2)

    add_bleach_marker(axis, first_postbleach_time)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    make_text_bold(axis)
    figure.tight_layout()

    save_figure(figure, output_name)


def plot_region_comparison(
    dataframe: pd.DataFrame,
    bleached_column: str,
    nonbleached_column: str,
    ylabel: str,
    title: str,
    output_name: str,
    first_postbleach_time: float,
) -> None:
    """Plot bleached and non-bleached values on the same time axis."""
    figure, axis = plt.subplots(figsize=(9, 6))

    axis.plot(
        dataframe["time_s"],
        dataframe[bleached_column],
        marker="o",
        linewidth=2,
        label="Bleached region",
    )
    axis.plot(
        dataframe["time_s"],
        dataframe[nonbleached_column],
        marker="s",
        linewidth=2,
        label="Non-bleached region",
    )

    add_bleach_marker(axis, first_postbleach_time)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    make_text_bold(axis)
    figure.tight_layout()

    save_figure(figure, output_name)


def make_displacement_vector_quiver_plot(
    profile_df: pd.DataFrame,
    first_frame: int,
    last_frame: int,
) -> None:
    """
    Compare displacement vectors in an early postbleach frame and the last frame.

    Vectors are calculated relative to the mean prebleach contour.
    """
    selected_frames = [first_frame]
    if last_frame != first_frame:
        selected_frames.append(last_frame)

    frame_data = []
    all_x = []
    all_y = []

    for frame in selected_frames:
        subset = (
            profile_df[profile_df["frame"] == frame]
            .sort_values("contour_index")
            .reset_index(drop=True)
        )

        if subset.empty:
            continue

        x = subset["x_px"].to_numpy(float)
        y = subset["y_px"].to_numpy(float)
        dx = (
            subset["displacement_vector_x_px"].to_numpy(float)
            * QUIVER_ARROW_DISPLAY_SCALE
        )
        dy = (
            subset["displacement_vector_y_px"].to_numpy(float)
            * QUIVER_ARROW_DISPLAY_SCALE
        )
        is_bleached = subset["is_bleached_arc"].to_numpy(bool)

        keep = np.arange(0, len(subset), QUIVER_EVERY_N_POINTS)

        frame_data.append((frame, x, y, dx, dy, is_bleached, keep))
        all_x.append(np.concatenate([x, x[keep] + dx[keep]]))
        all_y.append(np.concatenate([y, y[keep] + dy[keep]]))

    if not frame_data:
        raise ValueError("No requested frames were available for quiver plotting.")

    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    finite = np.isfinite(combined_x) & np.isfinite(combined_y)

    x_min, x_max = np.nanmin(combined_x[finite]), np.nanmax(combined_x[finite])
    y_min, y_max = np.nanmin(combined_y[finite]), np.nanmax(combined_y[finite])

    span = max(x_max - x_min, y_max - y_min)
    padding = 0.18 * span if span > 0 else 10.0

    common_xlim = (x_min - padding, x_max + padding)
    common_ylim = (y_max + padding, y_min - padding)

    figure, axes = plt.subplots(
        1,
        len(frame_data),
        figsize=(7 * len(frame_data), 7),
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, (frame, x, y, dx, dy, is_bleached, keep) in zip(
        axes,
        frame_data,
    ):
        axis.plot(
            x,
            y,
            linewidth=QUIVER_CONTOUR_LINEWIDTH,
            color="black",
            alpha=QUIVER_CONTOUR_ALPHA,
            label="Current contour",
        )

        axis.scatter(
            x[is_bleached],
            y[is_bleached],
            s=18,
            color=QUIVER_BLEACHED_POINT_COLOR,
            edgecolors=QUIVER_BLEACHED_POINT_COLOR,
            linewidths=0.5,
            label="Bleached region",
            zorder=3,
        )

        axis.quiver(
            x[keep],
            y[keep],
            dx[keep],
            dy[keep],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=QUIVER_ARROW_WIDTH,
            headwidth=QUIVER_ARROW_HEADWIDTH,
            headlength=QUIVER_ARROW_HEADLENGTH,
            headaxislength=QUIVER_ARROW_HEADAXISLENGTH,
            color=QUIVER_ARROW_COLOR,
            edgecolor=QUIVER_ARROW_EDGE_COLOR,
            linewidth=0.5,
            minlength=0.4,
            zorder=4,
        )

        axis.set_xlim(*common_xlim)
        axis.set_ylim(*common_ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")

        frame_label = (
            "first postbleach"
            if frame == first_frame
            else "last frame"
        )
        axis.set_title(f"Frame {frame} ({frame_label})")
        make_text_bold(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(FONT_SIZE)
        text.set_fontweight("bold")

    figure.suptitle(
        "Displacement vectors relative to the mean prebleach contour",
        fontsize=FONT_SIZE,
        fontweight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.91])

    save_figure(
        figure,
        f"displacement_vectors_frame_{first_frame:03d}_and_"
        f"frame_{last_frame:03d}.png",
    )


def plot_bleached_detection_profile(
    raw_profile: np.ndarray,
    smoothed_profile: np.ndarray,
    threshold: float,
    bleached_indices: np.ndarray,
    frame: int,
) -> None:
    """Save one quality-control plot for bleached-region detection."""
    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(
        raw_profile,
        linewidth=1.5,
        alpha=0.45,
        label="Raw contour intensity",
    )
    axis.plot(
        smoothed_profile,
        linewidth=2.5,
        label="Smoothed contour intensity",
    )
    axis.scatter(
        bleached_indices,
        smoothed_profile[bleached_indices],
        s=20,
        label="Detected bleached region",
        zorder=3,
    )
    axis.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"{BLEACHED_PERCENTILE}th percentile threshold",
    )

    axis.set_xlabel("Contour index")
    axis.set_ylabel("Fluorescence intensity")
    axis.set_title(f"Bleached-region detection, frame {frame}")
    axis.legend(frameon=False)
    make_text_bold(axis)
    figure.tight_layout()

    save_figure(
        figure,
        f"bleached_region_detection_frame_{frame:03d}.png",
    )



def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Percentile-normalize one frame for visualization only.

    This conversion is not used for quantitative measurements.
    """
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image, dtype=np.uint8)

    lower, upper = np.percentile(image[finite], [1, 99])

    if upper <= lower:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = np.clip((image - lower) / (upper - lower), 0, 1)
    return (255 * normalized).astype(np.uint8)


def save_bleached_detection_profiles_all_frames(
    stack: np.ndarray,
    contours: dict[int, np.ndarray],
    detected_bleached_indices_by_frame: dict[int, np.ndarray],
    fixed_threshold: float,
) -> None:
    """
    Save one contour-intensity detection profile for every available frame.

    For postbleach frames in per-frame mode, the displayed bleached indices are
    those obtained by applying the fixed reference-frame intensity threshold to
    that frame. Prebleach frames use the reference-frame spatial sector because
    no genuinely bleached dark region exists before bleaching.
    """
    profile_folder = OUTPUT_FOLDER / "bleached_detection_profiles_all_frames"
    profile_folder.mkdir(parents=True, exist_ok=True)

    for frame in sorted(contours):
        if frame >= stack.shape[0]:
            continue

        if frame not in detected_bleached_indices_by_frame:
            continue

        raw_profile = sample_image_at_points(
            stack[frame],
            contours[frame],
            radius=INTENSITY_BAND_RADIUS_PX,
        )
        smoothed_profile = circular_smooth_1d(
            raw_profile,
            BLEACHED_SMOOTHING_WINDOW,
        )

        arc_indices = np.asarray(
            detected_bleached_indices_by_frame[frame],
            dtype=int,
        )

        figure, axis = plt.subplots(figsize=(10, 5))

        axis.plot(
            raw_profile,
            linewidth=1.8,
            alpha=0.45,
            label="Raw contour intensity",
        )
        axis.plot(
            smoothed_profile,
            linewidth=3.0,
            label="Smoothed contour intensity",
        )
        axis.scatter(
            arc_indices,
            smoothed_profile[arc_indices],
            s=28,
            color="blue",
            edgecolors="blue",
            linewidths=0.5,
            label="Detected bleached region",
            zorder=3,
        )
        axis.axhline(
            fixed_threshold,
            linestyle="--",
            linewidth=2.0,
            color="black",
            label="Fixed reference-frame threshold",
        )

        axis.set_xlabel("Contour index")
        axis.set_ylabel("Fluorescence intensity")
        axis.set_title(f"Bleached-region detection profile, frame {frame}")
        axis.legend(frameon=False)
        make_text_bold(axis)
        figure.tight_layout()

        figure.savefig(
            profile_folder / f"bleached_detection_profile_frame_{frame:03d}.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
            transparent=True,
        )

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(figure)

    print(f"Saved all-frame bleached-region profiles to:\n{profile_folder}")


def save_mapping_points_overlay_tiff(
    stack: np.ndarray,
    contours: dict[int, np.ndarray],
    bleached_indices: np.ndarray,
    output_path: Path,
    point_radius: int = 1,
    label_every_n: int | None = 30,
) -> None:
    """
    Save an RGB TIFF stack showing contour-point correspondence over time.

    Cyan:
        non-bleached contour points.

    Yellow:
        bleached-region contour points.

    Red:
        contour index 0, used to verify the common contour starting position.
    """
    bleached_set = set(int(index) for index in bleached_indices)
    rgb_frames: list[np.ndarray] = []

    for frame in sorted(contours):
        if frame >= stack.shape[0]:
            continue

        image_8bit = normalize_to_uint8(stack[frame])
        rgb = np.stack([image_8bit, image_8bit, image_8bit], axis=-1)
        contour = contours[frame]

        for contour_index, (x, y) in enumerate(contour):
            rr, cc = disk(
                (int(round(y)), int(round(x))),
                radius=point_radius,
                shape=image_8bit.shape,
            )

            if contour_index in bleached_set:
                rgb[rr, cc] = (255, 255, 0)
            else:
                rgb[rr, cc] = (0, 255, 255)

        # Highlight the common contour start point.
        x_zero, y_zero = contour[0]
        rr, cc = disk(
            (int(round(y_zero)), int(round(x_zero))),
            radius=point_radius + 1,
            shape=image_8bit.shape,
        )
        rgb[rr, cc] = (255, 0, 0)

        if label_every_n is not None and label_every_n > 0:
            pil_image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_image)

            for contour_index in range(0, len(contour), label_every_n):
                x, y = contour[contour_index]
                text_color = (
                    (255, 255, 0)
                    if contour_index in bleached_set
                    else (0, 255, 255)
                )
                draw.text(
                    (float(x) + 3, float(y) + 3),
                    str(contour_index),
                    fill=text_color,
                )

            draw.text(
                (5, 5),
                f"Frame {frame}",
                fill=(255, 255, 255),
            )
            rgb = np.asarray(pil_image)

        rgb_frames.append(rgb)

    if not rgb_frames:
        raise ValueError("No frames were available for the mapping-points overlay.")

    tiff.imwrite(
        output_path,
        np.stack(rgb_frames, axis=0),
        photometric="rgb",
    )

    print(f"Saved mapping-points overlay TIFF to:\n{output_path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_folder(OUTPUT_FOLDER)

    stack = load_tiff_stack(TIFF_PATH)
    n_frames = stack.shape[0]
    time_s = build_time_axis(n_frames)

    raw_contours = load_contours(CONTOUR_PATH)
    base_contours = preprocess_contours(raw_contours)

    available_frames = sorted(
        frame
        for frame in base_contours
        if 0 <= frame < n_frames
    )

    if not available_frames:
        raise ValueError("No contour frames overlap the TIFF frame range.")

    prebleach_frames = [
        frame
        for frame in available_frames
        if frame < PREBLEACH_FRAMES
    ]

    if not prebleach_frames:
        raise ValueError("No prebleach contour frames were found.")

    if BLEACHED_REFERENCE_FRAME not in base_contours:
        raise ValueError(
            f"BLEACHED_REFERENCE_FRAME={BLEACHED_REFERENCE_FRAME} "
            "was not found in the contour file."
        )

    # Detect the bleached region in the reference frame and retain its
    # percentile-derived absolute intensity threshold.
    (
        budget_bleached_indices,
        detection_raw_profile,
        detection_smoothed_profile,
        detection_threshold,
    ) = detect_bleached_arc(
        stack[BLEACHED_REFERENCE_FRAME],
        base_contours[BLEACHED_REFERENCE_FRAME],
        fixed_threshold=None,
    )

    n_bleached_points = len(budget_bleached_indices)
    n_nonbleached_points = N_CONTOUR_POINTS - n_bleached_points

    if n_nonbleached_points < 10:
        raise ValueError(
            "Automatic detection assigned too much of the contour to the "
            "bleached region. Inspect the detection profile or use a manual arc."
        )

    print(
        f"Fixed bleached-region intensity threshold: "
        f"{detection_threshold:.3f}"
    )
    print(
        f"Region-aware point budget: {n_bleached_points} bleached and "
        f"{n_nonbleached_points} non-bleached points."
    )

    contours: dict[int, np.ndarray] = {}
    detected_bleached_indices_by_frame: dict[int, np.ndarray] = {}

    for frame in available_frames:
        # Before bleaching, use the same spatial sector detected in the first
        # postbleach frame. There is no dark bleached region to detect yet.
        if frame < PREBLEACH_FRAMES:
            detected_indices = budget_bleached_indices

        # In fixed mode, retain the reference-frame arc in every frame.
        elif BLEACHED_REGION_MODE == "fixed":
            detected_indices = budget_bleached_indices

        # In per-frame mode, apply the SAME absolute threshold obtained from
        # the reference frame to each postbleach frame and detect the longest
        # threshold-qualified dark arc.
        elif BLEACHED_REGION_MODE == "per_frame":
            (
                detected_indices,
                _,
                _,
                _,
            ) = detect_bleached_arc(
                stack[frame],
                base_contours[frame],
                fixed_threshold=detection_threshold,
            )

        else:
            raise ValueError(
                "BLEACHED_REGION_MODE must be 'per_frame' or 'fixed'."
            )

        detected_bleached_indices_by_frame[frame] = np.asarray(
            detected_indices,
            dtype=int,
        )

        contours[frame] = region_aware_resample(
            base_contours[frame],
            detected_indices,
            n_bleached_points,
            n_nonbleached_points,
        )

    # In the region-aware contours, the first fixed-size block always
    # represents the bleached region regardless of its original contour indices.
    bleached_indices = np.arange(0, n_bleached_points)
    nonbleached_indices = np.arange(
        n_bleached_points,
        N_CONTOUR_POINTS,
    )

    prebleach_reference = np.mean(
        np.stack([contours[frame] for frame in prebleach_frames]),
        axis=0,
    )

    reference_areas = np.asarray(
        [polygon_area(contours[frame]) for frame in prebleach_frames],
        dtype=float,
    )
    reference_perimeters = np.asarray(
        [polygon_perimeter(contours[frame]) for frame in prebleach_frames],
        dtype=float,
    )

    reference_area = float(np.mean(reference_areas))
    reference_perimeter = float(np.mean(reference_perimeters))

    reference_segment_lengths = np.nanmean(
        np.stack(
            [
                contour_segment_lengths(
                    contours[frame],
                    n_bleached_points,
                )
                for frame in prebleach_frames
            ]
        ),
        axis=0,
    )

    radial_units = radial_unit_vectors(prebleach_reference)

    global_rows = []
    local_rows = []
    profile_rows = []

    for frame in available_frames:
        contour = contours[frame]
        area_px2 = polygon_area(contour)
        perimeter_px = polygon_perimeter(contour)
        contour_circularity = circularity(area_px2, perimeter_px)
        contour_solidity = solidity(contour, area_px2)

        radial_displacement_px = signed_radial_displacement(
            contour,
            prebleach_reference,
            radial_units,
        )

        segment_lengths = contour_segment_lengths(
            contour,
            n_bleached_points,
        )
        local_strain = (
            segment_lengths - reference_segment_lengths
        ) / reference_segment_lengths

        global_rows.append(
            {
                "frame": frame,
                "time_s": time_s[frame],
                "phase": (
                    "prebleach"
                    if frame < PREBLEACH_FRAMES
                    else "postbleach"
                ),
                "area_px2": area_px2,
                "area_um2": unit_convert(area_px2, "area"),
                "area_change_vs_prebleach": (
                    area_px2 - reference_area
                ) / reference_area,
                "perimeter_px": perimeter_px,
                "perimeter_um": unit_convert(
                    perimeter_px,
                    "length",
                ),
                "perimeter_change_vs_prebleach": (
                    perimeter_px - reference_perimeter
                ) / reference_perimeter,
                "solidity": contour_solidity,
                "circularity": contour_circularity,
                "global_mean_abs_radial_displacement_px": float(
                    np.nanmean(np.abs(radial_displacement_px))
                ),
                "global_mean_abs_radial_displacement_um": float(
                    unit_convert(
                        np.nanmean(np.abs(radial_displacement_px)),
                        "length",
                    )
                ),
                "global_mean_abs_local_strain": float(
                    np.nanmean(np.abs(local_strain))
                ),
            }
        )

        local_rows.append(
            {
                "frame": frame,
                "time_s": time_s[frame],
                "phase": (
                    "prebleach"
                    if frame < PREBLEACH_FRAMES
                    else "postbleach"
                ),
                "bleached_mean_abs_radial_displacement_px": (
                    region_absolute_mean(
                        radial_displacement_px,
                        bleached_indices,
                    )
                ),
                "bleached_mean_abs_radial_displacement_um": (
                    unit_convert(
                        region_absolute_mean(
                            radial_displacement_px,
                            bleached_indices,
                        ),
                        "length",
                    )
                ),
                "nonbleached_mean_abs_radial_displacement_px": (
                    region_absolute_mean(
                        radial_displacement_px,
                        nonbleached_indices,
                    )
                ),
                "nonbleached_mean_abs_radial_displacement_um": (
                    unit_convert(
                        region_absolute_mean(
                            radial_displacement_px,
                            nonbleached_indices,
                        ),
                        "length",
                    )
                ),
                "bleached_mean_abs_local_strain": region_absolute_mean(
                    local_strain,
                    bleached_indices,
                ),
                "nonbleached_mean_abs_local_strain": region_absolute_mean(
                    local_strain,
                    nonbleached_indices,
                ),
            }
        )

        displacement_vectors = contour - prebleach_reference
        bleached_set = set(int(index) for index in bleached_indices)

        for contour_index in range(N_CONTOUR_POINTS):
            profile_rows.append(
                {
                    "frame": frame,
                    "time_s": time_s[frame],
                    "contour_index": contour_index,
                    "x_px": contour[contour_index, 0],
                    "y_px": contour[contour_index, 1],
                    "is_bleached_arc": contour_index in bleached_set,
                    "radial_displacement_px": radial_displacement_px[
                        contour_index
                    ],
                    "radial_displacement_um": unit_convert(
                        radial_displacement_px[contour_index],
                        "length",
                    ),
                    "local_segment_strain_vs_prebleach": local_strain[
                        contour_index
                    ],
                    "displacement_vector_x_px": displacement_vectors[
                        contour_index,
                        0,
                    ],
                    "displacement_vector_y_px": displacement_vectors[
                        contour_index,
                        1,
                    ],
                }
            )

    global_df = pd.DataFrame(global_rows)
    local_df = pd.DataFrame(local_rows)
    profile_df = pd.DataFrame(profile_rows)

    global_df.to_csv(
        OUTPUT_FOLDER / "global_metrics_relative_to_prebleach.csv",
        index=False,
    )
    local_df.to_csv(
        OUTPUT_FOLDER / "regional_metrics_relative_to_prebleach.csv",
        index=False,
    )
    profile_df.to_csv(
        OUTPUT_FOLDER / "pointwise_metrics_relative_to_prebleach.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "bleached_contour_index": bleached_indices,
        }
    ).to_csv(
        OUTPUT_FOLDER / "bleached_contour_indices.csv",
        index=False,
    )

    first_postbleach_time = float(
        time_s[min(PREBLEACH_FRAMES, n_frames - 1)]
    )

    distance_unit = "µm" if PIXEL_SIZE_UM is not None else "px"
    area_column = (
        "area_um2"
        if PIXEL_SIZE_UM is not None
        else "area_px2"
    )
    perimeter_column = (
        "perimeter_um"
        if PIXEL_SIZE_UM is not None
        else "perimeter_px"
    )
    global_displacement_column = (
        "global_mean_abs_radial_displacement_um"
        if PIXEL_SIZE_UM is not None
        else "global_mean_abs_radial_displacement_px"
    )
    bleached_displacement_column = (
        "bleached_mean_abs_radial_displacement_um"
        if PIXEL_SIZE_UM is not None
        else "bleached_mean_abs_radial_displacement_px"
    )
    nonbleached_displacement_column = (
        "nonbleached_mean_abs_radial_displacement_um"
        if PIXEL_SIZE_UM is not None
        else "nonbleached_mean_abs_radial_displacement_px"
    )

    # Global shape plots.
    plot_single_metric(
        global_df,
        area_column,
        f"Area ({'µm²' if PIXEL_SIZE_UM is not None else 'px²'})",
        "GUV area",
        "plot_area.png",
        first_postbleach_time,
    )
    plot_single_metric(
        global_df,
        "area_change_vs_prebleach",
        "Relative area change",
        "GUV area relative to the mean prebleach area",
        "plot_area_change_vs_prebleach.png",
        first_postbleach_time,
        horizontal_zero=True,
    )
    plot_single_metric(
        global_df,
        perimeter_column,
        f"Perimeter ({distance_unit})",
        "GUV perimeter",
        "plot_perimeter.png",
        first_postbleach_time,
    )
    plot_single_metric(
        global_df,
        "perimeter_change_vs_prebleach",
        "Relative perimeter change",
        "GUV perimeter relative to the mean prebleach perimeter",
        "plot_perimeter_change_vs_prebleach.png",
        first_postbleach_time,
        horizontal_zero=True,
    )
    plot_single_metric(
        global_df,
        "solidity",
        "Solidity",
        "GUV contour solidity",
        "plot_solidity.png",
        first_postbleach_time,
    )
    plot_single_metric(
        global_df,
        "circularity",
        "Circularity",
        "GUV contour circularity",
        "plot_circularity.png",
        first_postbleach_time,
    )

    # Displacement and local-strain plots, all relative to prebleach.
    plot_single_metric(
        global_df,
        global_displacement_column,
        f"Mean absolute radial displacement ({distance_unit})",
        "Global radial displacement relative to prebleach",
        "plot_global_mean_absolute_radial_displacement_vs_prebleach.png",
        first_postbleach_time,
    )
    plot_region_comparison(
        local_df,
        bleached_displacement_column,
        nonbleached_displacement_column,
        f"Mean absolute radial displacement ({distance_unit})",
        "Radial displacement relative to prebleach",
        "plot_bleached_nonbleached_mean_absolute_radial_displacement_vs_prebleach.png",
        first_postbleach_time,
    )
    plot_single_metric(
        global_df,
        "global_mean_abs_local_strain",
        "Mean absolute local segment strain",
        "Global local strain relative to prebleach",
        "plot_global_mean_absolute_local_strain_vs_prebleach.png",
        first_postbleach_time,
    )
    plot_region_comparison(
        local_df,
        "bleached_mean_abs_local_strain",
        "nonbleached_mean_abs_local_strain",
        "Mean absolute local segment strain",
        "Local strain relative to prebleach",
        "plot_bleached_nonbleached_mean_absolute_local_strain_vs_prebleach.png",
        first_postbleach_time,
    )

    # Quiver comparison.
    quiver_first = (
        QUIVER_FIRST_FRAME
        if QUIVER_FIRST_FRAME in available_frames
        else next(
            (
                frame
                for frame in available_frames
                if frame >= PREBLEACH_FRAMES
            ),
            available_frames[0],
        )
    )
    quiver_last = (
        QUIVER_LAST_FRAME
        if QUIVER_LAST_FRAME in available_frames
        else available_frames[-1]
    )

    make_displacement_vector_quiver_plot(
        profile_df,
        quiver_first,
        quiver_last,
    )

    # Reference-frame bleached-region detection plot.
    plot_bleached_detection_profile(
        detection_raw_profile,
        detection_smoothed_profile,
        detection_threshold,
        budget_bleached_indices,
        BLEACHED_REFERENCE_FRAME,
    )

    # Retained all-frame quality-control outputs.
    if SAVE_BLEACHED_DETECTION_PROFILES_ALL_FRAMES:
        save_bleached_detection_profiles_all_frames(
            stack=stack,
            contours=base_contours,
            detected_bleached_indices_by_frame=(
                detected_bleached_indices_by_frame
            ),
            fixed_threshold=detection_threshold,
        )

    if SAVE_MAPPING_POINTS_OVERLAY:
        save_mapping_points_overlay_tiff(
            stack=stack,
            contours=contours,
            bleached_indices=bleached_indices,
            output_path=OUTPUT_FOLDER / "mapping_points_overlay.tif",
            point_radius=MAPPING_POINT_RADIUS,
            label_every_n=MAPPING_LABEL_EVERY_N_POINTS,
        )

    print("\nAnalysis complete.")
    print(f"Outputs saved to:\n{OUTPUT_FOLDER}")

    return global_df, local_df, profile_df


if __name__ == "__main__":
    run_analysis()
