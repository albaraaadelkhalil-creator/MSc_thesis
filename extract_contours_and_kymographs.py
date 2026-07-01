#!/usr/bin/env python3
"""
GUV cortex contour extraction for linear cortices and contour-kymograph generation.

This script segments a time-lapse TIFF stack, tracks one GUV over time, extracts
an outer cortex contour by radial intensity profiling, measures simple contour
shape descriptors, and optionally generates a contour kymograph.

We developed this code as the intensity of the linear cortex was decreasing 
overtime as a result of photobleaching making it impossible for J-filament to extract a contour.

Author: Albaraa Adel Farouk Khalil 
ChatGPT was used as a tool to help develope this code
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from skimage.draw import polygon_perimeter
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, disk, remove_small_objects


# =============================================================================
# USER SETTINGS
# =============================================================================
@dataclass(frozen=True)
class AnalysisConfig:
    """All user-adjustable analysis settings."""

    # Input/output
    input_path: Path = Path(
        r"C:\Master End Project\Images\FRAP\26.02.26.Trial_2_Branched_vs_Linear"
        r"\Trial_2_Linear_vs_Branched_L2\FRAP003\affine.tif"
    )
    output_folder_name: str = "output_segmented"

    # Segmentation
    method: str = "otsu"  # "otsu" or "gmm"
    per_frame_thresholds: bool = True
    n_classes: int = 3
    guv_classes: tuple[int, ...] = (1, 2)  # 0=background, 1=lumen, 2=cortex

    # Optional pre-processing for segmentation/contour detection
    apply_gaussian_filter: bool = False
    gaussian_sigma_px: float = 1.0

    # Binary mask cleaning
    min_guv_area_px: int = 3000
    closing_radius_px: int = 5

    # GUV tracking weights
    distance_weight: float = 1.0
    area_weight: float = 300.0
    overlap_weight: float = 300.0

    # Radial contour extraction
    n_contour_points: int = 720
    r_min_px: int = 40
    r_max_px: int = 130
    search_margin_px: int = 10
    radius_smooth_window: int = 71
    max_radius_jump_px: int = 8

    # Safety mask derived from frame 0
    use_frame0_safety_mask: bool = True
    safety_margin_px: int = 15

    # Kymograph generation
    make_kymographs: bool = True
    kymo_width_px: int = 3
    kymo_use_filtered_stack: bool = False
    kymo_save_display_uint8: bool = True
    flip_kymograph_vertical: bool = False
    transpose_kymograph: bool = True

    # Kymograph row orientation
    rotate_kymograph_rows: bool = True
    kymo_current_start: str = "right"  # "right", "top", "left", "bottom"
    kymo_target_start: str = "top"  # "right", "top", "left", "bottom", "custom"
    kymo_custom_shift_rows: int = 0


# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def make_run_tag(config: AnalysisConfig) -> str:
    """Create a compact output-name tag describing the main analysis settings."""
    threshold_tag = "perframe" if config.per_frame_thresholds else "global"
    filter_tag = (
        f"gauss{config.gaussian_sigma_px:g}"
        if config.apply_gaussian_filter
        else "no_gauss"
    )
    return (
        f"{config.method}_{threshold_tag}_{filter_tag}"
        f"_safety{config.safety_margin_px}px"
    )


def load_tiff_stack(input_path: Path) -> np.ndarray:
    """Load a 2D TIFF image or 3D TIFF stack as [frame, y, x]."""
    stack = np.squeeze(tiff.imread(str(input_path)))

    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    elif stack.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D stack, got shape {stack.shape}.")

    return stack


def normalize_to_uint8(image: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """
    Percentile-normalize an image for display only.

    The returned uint8 image should not be used for quantitative intensity analysis.
    """
    img = image.astype(float)
    finite = np.isfinite(img)

    if finite.sum() == 0:
        return np.zeros_like(img, dtype=np.uint8)

    low, high = np.percentile(img[finite], (p_low, p_high))
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-9)
    return (255 * img).astype(np.uint8)


# =============================================================================
# SEGMENTATION
# =============================================================================
def compute_thresholds(frame_or_stack: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Compute class-separating intensity thresholds using multi-Otsu or GMM."""
    method = config.method.lower()

    if method == "otsu":
        return np.asarray(threshold_multiotsu(frame_or_stack, classes=config.n_classes))

    if method == "gmm":
        pixels = frame_or_stack.reshape(-1, 1)
        gmm = GaussianMixture(n_components=config.n_classes, random_state=0)
        gmm.fit(pixels)
        means = np.sort(gmm.means_.ravel())
        return np.asarray([(means[i] + means[i + 1]) / 2 for i in range(config.n_classes - 1)])

    raise ValueError("config.method must be either 'otsu' or 'gmm'.")


def segment_stack(stack_for_processing: np.ndarray, config: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment each frame into intensity classes.

    Returns
    -------
    raw_class_stack:
        Integer labels: 0=background, 1=intermediate/lumen, 2=bright/cortex.
    display_label_stack:
        The same labels scaled to uint8 values for visual inspection.
    threshold_log:
        Array with columns [frame, t1, t2].
    """
    n_frames = stack_for_processing.shape[0]
    raw_class_stack = np.zeros(stack_for_processing.shape, dtype=np.uint8)
    display_label_stack = np.zeros(stack_for_processing.shape, dtype=np.uint8)
    threshold_log: list[list[float]] = []

    global_thresholds: Optional[np.ndarray] = None
    if not config.per_frame_thresholds:
        global_thresholds = compute_thresholds(stack_for_processing, config)
        print(f"Global thresholds: {global_thresholds}")

    for frame_index in range(n_frames):
        frame = stack_for_processing[frame_index]
        thresholds = global_thresholds if global_thresholds is not None else compute_thresholds(frame, config)

        if len(thresholds) != 2:
            raise ValueError("This script expects n_classes=3, which gives exactly two thresholds.")

        t1, t2 = thresholds
        threshold_log.append([frame_index, float(t1), float(t2)])

        labels = np.zeros_like(frame, dtype=np.uint8)
        labels[frame >= t1] = 1
        labels[frame >= t2] = 2

        raw_class_stack[frame_index] = labels
        display_label_stack[frame_index] = (labels * 127).astype(np.uint8)

    return raw_class_stack, display_label_stack, np.asarray(threshold_log, dtype=float)


# =============================================================================
# MASK CLEANING AND GUV TRACKING
# =============================================================================
def clean_binary_mask(mask: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Close holes/gaps, fill internal holes, and remove small objects."""
    cleaned = closing(mask.astype(bool), disk(config.closing_radius_px))
    cleaned = binary_fill_holes(cleaned)
    cleaned = remove_small_objects(cleaned, min_size=config.min_guv_area_px)
    return cleaned.astype(bool)


def score_candidate(
    region,
    previous_centroid: tuple[float, float],
    previous_area: float,
    previous_mask: np.ndarray,
    label_image: np.ndarray,
    config: AnalysisConfig,
) -> float:
    """Score candidate GUV objects using motion, area change, and mask overlap."""
    cy, cx = region.centroid
    py, px = previous_centroid

    distance = np.hypot(cy - py, cx - px)
    relative_area_change = abs(region.area - previous_area) / max(previous_area, 1)

    candidate_mask = label_image == region.label
    overlap = np.logical_and(candidate_mask, previous_mask).sum()
    union = np.logical_or(candidate_mask, previous_mask).sum()
    intersection_over_union = overlap / union if union > 0 else 0.0

    return (
        config.distance_weight * distance
        + config.area_weight * relative_area_change
        - config.overlap_weight * intersection_over_union
    )


def select_tracked_guv(
    mask: np.ndarray,
    config: AnalysisConfig,
    previous_centroid: Optional[tuple[float, float]] = None,
    previous_area: Optional[float] = None,
    previous_mask: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], Optional[tuple[float, float]], Optional[float]]:
    """Select the largest GUV in frame 0 and then track the closest matching object."""
    label_image = label(mask)
    regions = [r for r in regionprops(label_image) if r.area >= config.min_guv_area_px]

    if not regions:
        return None, None, None

    if previous_centroid is None or previous_area is None or previous_mask is None:
        best = max(regions, key=lambda r: r.area)
    else:
        best = min(
            regions,
            key=lambda r: score_candidate(
                r,
                previous_centroid=previous_centroid,
                previous_area=previous_area,
                previous_mask=previous_mask,
                label_image=label_image,
                config=config,
            ),
        )

    return label_image == best.label, best.centroid, float(best.area)


def make_frame0_safety_mask(guv_initial: np.ndarray, config: AnalysisConfig) -> Optional[np.ndarray]:
    """Create a dilated mask around the largest frame-0 GUV to reject distant objects later."""
    cleaned = clean_binary_mask(guv_initial, config)
    label_image = label(cleaned)
    regions = regionprops(label_image)

    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    frame0_mask = label_image == largest.label
    return dilation(frame0_mask, disk(config.safety_margin_px)).astype(bool)


# =============================================================================
# RADIAL CONTOUR EXTRACTION AND METRICS
# =============================================================================
def radial_cortex_contour(
    frame: np.ndarray,
    center: tuple[float, float],
    config: AnalysisConfig,
    previous_radius: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Extract the outer cortex contour from radial intensity profiles.

    For each angle around the GUV centroid, a radial intensity profile is sampled.
    The bright cortex peak is found first, and the outer cortex position is assigned
    to the strongest negative gradient after that peak.
    """
    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, config.n_contour_points, endpoint=False)

    if previous_radius is None:
        radial_range = np.arange(config.r_min_px, config.r_max_px)
    else:
        radial_range = np.arange(
            max(config.r_min_px, int(previous_radius - config.search_margin_px)),
            min(config.r_max_px, int(previous_radius + config.search_margin_px)),
        )

    if len(radial_range) < 10:
        radial_range = np.arange(config.r_min_px, config.r_max_px)

    radii = np.empty(config.n_contour_points, dtype=float)

    for angle_index, theta in enumerate(angles):
        ys = cy + radial_range * np.sin(theta)
        xs = cx + radial_range * np.cos(theta)

        profile = map_coordinates(frame, [ys, xs], order=1, mode="nearest")
        profile_smooth = savgol_filter(profile, window_length=9, polyorder=2, mode="nearest")
        gradient = np.gradient(profile_smooth)

        peak_index = int(np.argmax(profile_smooth))

        if peak_index < len(gradient) - 3:
            best_index = peak_index + int(np.argmin(gradient[peak_index:]))
        else:
            best_index = peak_index

        radii[angle_index] = radial_range[best_index]

    median_radius = np.nanmedian(radii)
    failed = np.abs(radii - median_radius) > config.max_radius_jump_px
    radii[failed] = np.nan

    valid = np.isfinite(radii)
    if valid.sum() > 10:
        radii[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), radii[valid])
    else:
        radii[:] = median_radius

    smooth_window = make_valid_savgol_window(config.radius_smooth_window, len(radii), minimum=5)
    extended_radii = np.r_[radii, radii, radii]
    radii = savgol_filter(
        extended_radii,
        window_length=smooth_window,
        polyorder=3,
        mode="wrap",
    )[len(radii) : 2 * len(radii)]

    contour_y = cy + radii * np.sin(angles)
    contour_x = cx + radii * np.cos(angles)
    contour = np.column_stack([contour_y, contour_x])

    return contour, float(np.nanmedian(radii))


def make_valid_savgol_window(window: int, signal_length: int, minimum: int = 5) -> int:
    """Return an odd Savitzky-Golay window smaller than the signal length."""
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window >= signal_length:
        window = signal_length - 1 if signal_length % 2 == 0 else signal_length - 2
    return max(window, minimum)


def contour_to_mask(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a contour into an 8-bit binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    rr, cc = polygon_perimeter(
        np.round(contour[:, 0]).astype(int),
        np.round(contour[:, 1]).astype(int),
        shape=shape,
        clip=True,
    )
    mask[rr, cc] = 255
    return mask


def contour_metrics(contour: np.ndarray) -> tuple[float, float, float, float, float]:
    """Calculate area, perimeter, circularity, solidity, and equivalent diameter."""
    y = contour[:, 0]
    x = contour[:, 1]

    perimeter = np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])))
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else np.nan

    points = np.column_stack([x, y])
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hx, hy = hull_points[:, 0], hull_points[:, 1]
        hull_area = 0.5 * abs(np.dot(hx, np.roll(hy, 1)) - np.dot(hy, np.roll(hx, 1)))
        solidity = area / hull_area if hull_area > 0 else np.nan
    else:
        solidity = np.nan

    return float(area), float(perimeter), float(circularity), float(solidity), float(equivalent_diameter)


# =============================================================================
# CONTOUR STORAGE AND KYMOGRAPH GENERATION
# =============================================================================
def save_contours_txt(contours: dict[str, Optional[np.ndarray]], output_path: Path, z_value: int = 0) -> None:
    """Save contour coordinates as frame, point index, x, y, z text columns."""
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("#\n")
        for key in sorted(contours):
            contour = contours[key]
            frame_index = int(key.replace("frame_", "")) + 1  # one-based indexing
            handle.write(f"{frame_index}\n")

            if contour is None:
                continue

            for point_index, (y, x) in enumerate(contour):
                handle.write(f"{frame_index}\t{point_index}\t{x:.12f}\t{y:.12f}\t{z_value}\n")


def contour_normals_from_center(contour: np.ndarray, centroid: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate radial unit normals from the centroid to each contour point."""
    cy, cx = centroid
    ny = contour[:, 0] - cy
    nx = contour[:, 1] - cx
    norm = np.hypot(ny, nx) + 1e-9
    return ny / norm, nx / norm


def sample_contour_intensity(
    frame: np.ndarray,
    contour: np.ndarray,
    centroid: tuple[float, float],
    width_px: int = 1,
) -> np.ndarray:
    """Sample intensity along a contour or an averaged radial band around it."""
    y = contour[:, 0]
    x = contour[:, 1]

    if width_px <= 1:
        return map_coordinates(frame, [y, x], order=1, mode="nearest")

    if width_px % 2 == 0:
        width_px += 1

    half_width = width_px // 2
    offsets = np.arange(-half_width, half_width + 1)
    ny, nx = contour_normals_from_center(contour, centroid)

    profiles = [map_coordinates(frame, [y + offset * ny, x + offset * nx], order=1, mode="nearest") for offset in offsets]
    return np.nanmean(np.asarray(profiles), axis=0)


def make_contour_kymograph(
    image_stack: np.ndarray,
    contours: dict[str, Optional[np.ndarray]],
    metrics: np.ndarray,
    width_px: int = 1,
) -> np.ndarray:
    """Create a 2D kymograph with contour position as rows and time as columns."""
    valid_contours = [c for c in contours.values() if c is not None]
    if not valid_contours:
        raise ValueError("No valid contours found. Cannot create kymograph.")

    n_frames = image_stack.shape[0]
    n_points = valid_contours[0].shape[0]
    kymograph = np.full((n_points, n_frames), np.nan, dtype=np.float32)

    for frame_index in range(n_frames):
        contour = contours.get(f"frame_{frame_index:04d}")
        if contour is None:
            continue

        centroid_y, centroid_x = metrics[frame_index, 1], metrics[frame_index, 2]
        if not np.isfinite(centroid_y) or not np.isfinite(centroid_x):
            continue

        profile = sample_contour_intensity(
            frame=image_stack[frame_index],
            contour=contour,
            centroid=(centroid_y, centroid_x),
            width_px=width_px,
        )
        kymograph[:, frame_index] = profile.astype(np.float32)

    return kymograph


def rotate_kymograph_rows(
    kymograph: np.ndarray,
    current_start: str = "right",
    target_start: str = "top",
    custom_shift_rows: int = 0,
) -> tuple[np.ndarray, int]:
    """Circularly shift kymograph rows to change the displayed contour start point."""
    start_to_fraction = {"right": 0.00, "bottom": 0.25, "left": 0.50, "top": 0.75}
    current_start = current_start.lower()
    target_start = target_start.lower()
    n_rows = kymograph.shape[0]

    if target_start == "custom":
        shift_rows = int(custom_shift_rows)
    else:
        if current_start not in start_to_fraction:
            raise ValueError("current_start must be 'right', 'top', 'left', or 'bottom'.")
        if target_start not in start_to_fraction:
            raise ValueError("target_start must be 'right', 'top', 'left', 'bottom', or 'custom'.")

        current_index = int(round(start_to_fraction[current_start] * n_rows))
        target_index = int(round(start_to_fraction[target_start] * n_rows))
        shift_rows = -(target_index - current_index)

    return np.roll(kymograph, shift=shift_rows, axis=0), shift_rows


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================
def track_contours(
    raw_class_stack: np.ndarray,
    stack_for_processing: np.ndarray,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Optional[np.ndarray]], np.ndarray]:
    """Track the GUV and extract one radial cortex contour per frame."""
    n_frames, height, width = stack_for_processing.shape

    guv_mask_stack = np.zeros((n_frames, height, width), dtype=np.uint8)
    contour_mask_stack = np.zeros((n_frames, height, width), dtype=np.uint8)
    safety_mask_stack = np.zeros((n_frames, height, width), dtype=np.uint8)

    contours: dict[str, Optional[np.ndarray]] = {}
    metrics_log: list[list[float]] = []

    previous_centroid: Optional[tuple[float, float]] = None
    previous_area: Optional[float] = None
    previous_mask: Optional[np.ndarray] = None
    previous_radius: Optional[float] = None
    frame0_safety_mask: Optional[np.ndarray] = None

    for frame_index in range(n_frames):
        guv_initial = np.isin(raw_class_stack[frame_index], config.guv_classes)

        if config.use_frame0_safety_mask:
            if frame_index == 0:
                frame0_safety_mask = make_frame0_safety_mask(guv_initial, config)
                if frame0_safety_mask is None:
                    print("Warning: could not create frame-0 safety mask.")
                else:
                    print("Created frame-0 safety mask.")
            elif frame0_safety_mask is not None:
                guv_initial = guv_initial & frame0_safety_mask

        guv_clean = clean_binary_mask(guv_initial, config)
        guv_mask, centroid, area = select_tracked_guv(
            guv_clean,
            config=config,
            previous_centroid=previous_centroid,
            previous_area=previous_area,
            previous_mask=previous_mask,
        )

        key = f"frame_{frame_index:04d}"
        if guv_mask is None or centroid is None or area is None:
            print(f"Frame {frame_index}: no GUV found.")
            contours[key] = None
            metrics_log.append([frame_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        contour, current_radius = radial_cortex_contour(
            frame=stack_for_processing[frame_index],
            center=centroid,
            config=config,
            previous_radius=previous_radius,
        )

        contour_area, perimeter, circularity, solidity, equivalent_diameter = contour_metrics(contour)

        guv_mask_stack[frame_index] = guv_mask.astype(np.uint8) * 255
        contour_mask_stack[frame_index] = contour_to_mask(contour, shape=(height, width))
        if frame0_safety_mask is not None:
            safety_mask_stack[frame_index] = frame0_safety_mask.astype(np.uint8) * 255

        contours[key] = contour
        metrics_log.append(
            [
                frame_index,
                centroid[0],
                centroid[1],
                contour_area,
                perimeter,
                circularity,
                solidity,
                equivalent_diameter,
            ]
        )

        previous_centroid = centroid
        previous_area = area
        previous_mask = guv_mask
        previous_radius = current_radius

    return guv_mask_stack, contour_mask_stack, safety_mask_stack, contours, np.asarray(metrics_log, dtype=float)


def make_overlay_stack(stack: np.ndarray, contour_mask_stack: np.ndarray) -> np.ndarray:
    """Create an RGB overlay stack with the contour shown in red."""
    n_frames, height, width = stack.shape
    overlay = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    display_stack = normalize_to_uint8(stack)

    for frame_index in range(n_frames):
        gray = display_stack[frame_index]
        overlay[frame_index, :, :, 0] = gray
        overlay[frame_index, :, :, 1] = gray
        overlay[frame_index, :, :, 2] = gray

        contour_pixels = contour_mask_stack[frame_index] > 0
        overlay[frame_index, contour_pixels, 0] = 255
        overlay[frame_index, contour_pixels, 1] = 0
        overlay[frame_index, contour_pixels, 2] = 0

    return overlay


def save_histogram(stack_for_processing: np.ndarray, thresholds: Iterable[float], output_path: Path, title: str) -> None:
    """Save an intensity histogram with the final threshold values overlaid."""
    plt.figure(figsize=(8, 5))
    plt.hist(stack_for_processing.ravel(), bins=256)
    for threshold in thresholds:
        plt.axvline(threshold, linestyle="--")
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(config: AnalysisConfig = AnalysisConfig()) -> None:
    """Run the complete segmentation, contour tracking, and kymograph pipeline."""
    run_tag = make_run_tag(config)
    output_dir = config.input_path.parent / config.output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = config.input_path.stem

    labels_path = output_dir / f"{stem}_labels_{run_tag}.tif"
    thresholds_path = output_dir / f"{stem}_thresholds_{run_tag}.csv"
    histogram_path = output_dir / f"{stem}_histogram_{run_tag}.png"
    guv_mask_path = output_dir / f"{stem}_guv_mask_{run_tag}.tif"
    safety_mask_path = output_dir / f"{stem}_frame0_safety_mask_{run_tag}.tif"
    contour_mask_path = output_dir / f"{stem}_contour_mask_{run_tag}.tif"
    overlay_path = output_dir / f"{stem}_contour_overlay_{run_tag}.tif"
    contours_npz_path = output_dir / f"{stem}_contours_{run_tag}.npz"
    contours_txt_path = output_dir / f"{stem}_contours_{run_tag}.txt"
    metrics_path = output_dir / f"{stem}_contour_metrics_{run_tag}.csv"
    kymograph_path = output_dir / f"{stem}_contour_kymograph_{run_tag}_width{config.kymo_width_px}px.tif"
    kymograph_display_path = output_dir / f"{stem}_contour_kymograph_DISPLAY_{run_tag}_width{config.kymo_width_px}px.tif"

    stack = load_tiff_stack(config.input_path)
    print(f"Loaded stack with shape [frames, y, x]: {stack.shape}")

    if config.apply_gaussian_filter:
        stack_for_processing = gaussian_filter(stack.astype(float), sigma=(0, config.gaussian_sigma_px, config.gaussian_sigma_px))
        print(f"Using Gaussian-filtered stack for segmentation and contour detection, sigma={config.gaussian_sigma_px} px")
    else:
        stack_for_processing = stack.astype(float)
        print("Using raw stack for segmentation and contour detection.")

    raw_class_stack, display_label_stack, threshold_log = segment_stack(stack_for_processing, config)

    guv_mask_stack, contour_mask_stack, safety_mask_stack, contours, metrics = track_contours(
        raw_class_stack=raw_class_stack,
        stack_for_processing=stack_for_processing,
        config=config,
    )

    # Save segmentation and contour outputs.
    tiff.imwrite(labels_path, display_label_stack)
    tiff.imwrite(guv_mask_path, guv_mask_stack)
    tiff.imwrite(contour_mask_path, contour_mask_stack)
    if np.any(safety_mask_stack):
        tiff.imwrite(safety_mask_path, safety_mask_stack)

    np.savez(contours_npz_path, **{k: v for k, v in contours.items() if v is not None})
    save_contours_txt(contours, contours_txt_path)

    np.savetxt(
        thresholds_path,
        threshold_log,
        delimiter=",",
        header="frame,t1,t2",
        comments="",
    )
    np.savetxt(
        metrics_path,
        metrics,
        delimiter=",",
        header="frame,centroid_y,centroid_x,area_px,perimeter_px,circularity,solidity,equivalent_diameter_px",
        comments="",
    )

    # Kymograph outputs.
    if config.make_kymographs:
        kymo_source_stack = stack_for_processing if config.kymo_use_filtered_stack else stack.astype(float)
        source_label = "Gaussian-filtered" if config.kymo_use_filtered_stack else "original"
        print(f"Creating contour kymograph from {source_label} stack.")

        kymograph = make_contour_kymograph(
            image_stack=kymo_source_stack,
            contours=contours,
            metrics=metrics,
            width_px=config.kymo_width_px,
        )

        if config.rotate_kymograph_rows:
            kymograph, applied_shift = rotate_kymograph_rows(
                kymograph,
                current_start=config.kymo_current_start,
                target_start=config.kymo_target_start,
                custom_shift_rows=config.kymo_custom_shift_rows,
            )
            print(f"Applied kymograph row shift: {applied_shift} rows.")

        if config.flip_kymograph_vertical:
            kymograph = np.flipud(kymograph)
        if config.transpose_kymograph:
            kymograph = kymograph.T

        tiff.imwrite(kymograph_path, kymograph.astype(np.float32))
        if config.kymo_save_display_uint8:
            tiff.imwrite(kymograph_display_path, normalize_to_uint8(kymograph))

    # Visual QC outputs.
    overlay_stack = make_overlay_stack(stack, contour_mask_stack)
    tiff.imwrite(overlay_path, overlay_stack)

    save_histogram(
        stack_for_processing=stack_for_processing,
        thresholds=threshold_log[-1, 1:],
        output_path=histogram_path,
        title=f"Intensity histogram with thresholds ({run_tag})",
    )

    print("\nSaved outputs:")
    for path in [
        labels_path,
        guv_mask_path,
        contour_mask_path,
        overlay_path,
        contours_npz_path,
        contours_txt_path,
        metrics_path,
        thresholds_path,
        histogram_path,
    ]:
        print(f"  - {path}")

    if config.make_kymographs:
        print(f"  - {kymograph_path}")
        if config.kymo_save_display_uint8:
            print(f"  - {kymograph_display_path}")

    if np.any(safety_mask_stack):
        print(f"  - {safety_mask_path}")

    print("Done.")


if __name__ == "__main__":
    main()
