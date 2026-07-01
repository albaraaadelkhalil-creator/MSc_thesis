README — ablation_analysis.py
=============================

Purpose
-------

ablation_analysis.py analyzes the mechanical response of one giant unilamellar
vesicle (GUV) during and after photobleaching. It combines a registered TIFF
image stack with a contour file exported from JFilament, identifies the
bleached part of the cortex, maps contour points over time, and calculates
shape and deformation metrics relative to a mean prebleach reference contour.

This script is the first step in the analysis workflow. It should be run once
for each individual GUV experiment. Its CSV outputs are then used by
mechanical_perturbation_compare_samples.py to compare several experiments.

Main calculations
-----------------

The script calculates and saves:

1. GUV area.
2. Area change relative to the mean prebleach area.
3. GUV perimeter.
4. Perimeter change relative to the mean prebleach perimeter.
5. Solidity.
6. Circularity.
7. Mean absolute radial displacement relative to the mean prebleach contour.
8. Mean absolute local segment strain relative to the mean prebleach contour.
9. Bleached-region and non-bleached-region versions of radial displacement and
   local strain.
10. Pointwise contour coordinates and displacement vectors.
11. Quiver plots showing contour displacement vectors relative to the mean
    prebleach contour.
12. Quality-control outputs for bleached-region detection and contour mapping.

Required input files
--------------------

1. A TIFF time series containing one GUV.

   The TIFF should contain one fluorescence channel and have the dimensions:

       frame × y × x

   Before running this script, correct the stack for translational and
   rotational drift, for example using StackReg in Fiji/ImageJ. A multichannel
   TIFF should first be reduced to the channel used for contour and bleach-region
   analysis.

2. A contour file exported from JFilament.

   The code accepts a JFilament-like .snakes or text file containing contour
   points for all frames. Numeric rows are expected to contain:

       frame_id   point_id   x   y   z

   Frame numbering in the contour file is converted internally from one-based
   to zero-based indexing.

Software requirements
---------------------

Use Python 3.10 or later. Required packages are:

    matplotlib
    numpy
    pandas
    pillow
    scipy
    scikit-image
    tifffile

Install them with:

    pip install matplotlib numpy pandas pillow scipy scikit-image tifffile

Recommended workflow
--------------------

1. Register the original TIFF stack in Fiji/ImageJ to correct translation and
   rotation.
2. Extract the GUV contour in every usable frame with JFilament.
3. Export the JFilament contour file.
4. Open ablation_analysis.py.
5. Edit the USER SETTINGS section.
6. Run the script.
7. Inspect all quality-control outputs before accepting the numerical results.
8. Repeat the analysis for every experimental condition or GUV.
9. Use the resulting output folders as input for
   mechanical_perturbation_compare_samples.py.

Essential settings to update
----------------------------

TIFF_PATH
    Full path to the registered TIFF stack.

    Example:

        TIFF_PATH = Path(r"C:\path\to\registered_stack.tif")

CONTOUR_PATH
    Full path to the matching JFilament contour file.

    Example:

        CONTOUR_PATH = Path(r"C:\path\to\registered_stack.snakes")

OUTPUT_FOLDER
    Folder in which all CSV files, plots, and quality-control outputs are saved.
    By default, it is created next to the TIFF file:

        OUTPUT_FOLDER = TIFF_PATH.parent / "guv_deformation_analysis"

PIXEL_SIZE_UM
    Physical pixel size in micrometres per pixel. Update this value using the
    image metadata.

    Example:

        PIXEL_SIZE_UM = 0.2083089

    Set this to None to keep all length measurements in pixels.

PREBLEACH_FRAMES
    Number of frames acquired before the bleaching event. In the current
    workflow, this is usually 5.

Time-axis settings
------------------

TIME_MODE controls how acquisition time is assigned.

1. "constant_post"

   Use when all postbleach frames have the same time interval.

       TIME_MODE = "constant_post"
       PREBLEACH_DT_S = 0.653
       POSTBLEACH_DT_S = 0.653

2. "blocks"

   Use when the acquisition contains several blocks with different frame
   intervals. The current example is:

       POSTBLEACH_BLOCKS = [
           ("Pb1", 15, 0.653),
           ("Pb2", 10, 5.0),
           ("Pb3", 30, 10.0),
       ]

   Each tuple contains:

       block name, number of frames, time interval in seconds

   The total number of frames described by the blocks must cover every
   postbleach frame in the TIFF stack.

3. "explicit"

   Use when every frame has a separately known acquisition time.

       TIME_MODE = "explicit"
       EXPLICIT_TIME_AXIS_S = np.array([...])

   The number of time values must equal the number of TIFF frames.

Contour preprocessing
---------------------

N_CONTOUR_POINTS
    Number of contour points used after resampling. The default is 720.

START_AT_TOP
    When True, contour index 0 is moved to the top of the vesicle. This helps
    keep contour indexing comparable between frames.

FORCE_CLOCKWISE
    Controls contour orientation. Keep this setting identical between all
    experiments.

Bleached-region detection
-------------------------

BLEACHED_REFERENCE_FRAME
    Frame used to establish the initial bleached-region intensity threshold.
    The default equals PREBLEACH_FRAMES, meaning the first postbleach frame.

BLEACHED_REGION_MODE

    "per_frame"
        The absolute threshold calculated in the reference frame is reapplied
        independently to every postbleach frame. The longest continuous region
        below the threshold is treated as the bleached arc.

    "fixed"
        The spatial arc detected in the reference frame is reused for all
        frames.

The current code uses:

    BLEACHED_REGION_MODE = "per_frame"

BLEACHED_PERCENTILE
    Percentile used in the reference frame to define the absolute intensity
    threshold. A value of 10 means that the 10th percentile of the smoothed
    contour intensity profile is used.

INTENSITY_BAND_RADIUS_PX
    Radius of the local pixel neighbourhood averaged around every contour point.

BLEACHED_SMOOTHING_WINDOW
    Circular smoothing window applied to the contour intensity profile.

MIN_BLEACHED_ARC_POINTS
    Minimum number of contour points assigned to a detected bleached region.

BLEACHED_FILL_GAPS_POINTS
    Maximum short bright interruption that can be filled inside an otherwise
    continuous dark bleached region.

BLEACHED_REMOVE_SMALL_OBJECTS_POINTS
    Minimum size of isolated dark contour sections. Smaller detections are
    removed.

EXPAND_BLEACHED_ARC_POINTS
    Number of contour indices added to both sides of the detected bleached arc.

Manual bleached-region definition
---------------------------------

If automatic detection is not reliable, set:

    USE_MANUAL_BLEACHED_ARC = True

and define:

    MANUAL_ARC_START_INDEX
    MANUAL_ARC_END_INDEX

The indices refer to the standardized contour after resampling. Inspect the
mapping_points_overlay.tif file to identify the appropriate index range.

Important analytical logic
--------------------------

1. Each contour is resampled to a fixed number of points and assigned a
   consistent orientation and starting point.
2. The bleached arc is detected from fluorescence intensity along the contour.
3. The contour is split into bleached and non-bleached arcs.
4. The two regions are resampled separately using a fixed point budget.
5. The mean of the prebleach contours is used as the reference geometry.
6. Global shape metrics are calculated from each full contour.
7. Pointwise radial displacement is calculated by projecting contour motion
   onto radial unit vectors derived from the mean prebleach contour.
8. Local segment strain is calculated from changes in neighbouring contour
   segment lengths relative to their mean prebleach lengths.
9. Mean absolute values are calculated separately for the bleached and
   non-bleached regions.

The separate resampling of the bleached and non-bleached regions prevents their
shared boundary from being treated as a normal contour segment during local
strain calculation.

Output files
------------

The main quantitative outputs are:

1. global_metrics_relative_to_prebleach.csv

   Contains one row per frame, including:

       frame
       time_s
       phase
       area_px2
       area_um2
       area_change_vs_prebleach
       perimeter_px
       perimeter_um
       perimeter_change_vs_prebleach
       solidity
       circularity
       global_mean_abs_radial_displacement_px
       global_mean_abs_radial_displacement_um
       global_mean_abs_local_strain

2. regional_metrics_relative_to_prebleach.csv

   Contains one row per frame, including:

       bleached_mean_abs_radial_displacement_px
       bleached_mean_abs_radial_displacement_um
       nonbleached_mean_abs_radial_displacement_px
       nonbleached_mean_abs_radial_displacement_um
       bleached_mean_abs_local_strain
       nonbleached_mean_abs_local_strain

   This is one of the two files required by
   mechanical_perturbation_compare_samples.py.

3. pointwise_metrics_relative_to_prebleach.csv

   Contains pointwise contour information for every frame, including:

       contour coordinates
       bleached-region assignment
       radial displacement
       local segment strain
       x and y displacement-vector components

4. bleached_contour_indices.csv

   Lists the standardized contour indices assigned to the bleached region.

Generated plots
---------------

The script saves plots for:

    area
    area change relative to prebleach
    perimeter
    perimeter change relative to prebleach
    solidity
    circularity
    global mean absolute radial displacement
    regional mean absolute radial displacement
    global mean absolute local strain
    regional mean absolute local strain
    displacement-vector quiver comparison

Plots are saved without being displayed because:

    SHOW_PLOTS = False

The image resolution is controlled by:

    PLOT_DPI = 1200

Quality-control outputs
-----------------------

1. bleached_region_detection_frame_XXX.png

   Shows the reference-frame contour intensity profile, smoothed profile,
   threshold, and detected bleached region.

2. bleached_detection_profiles_all_frames/

   Contains one detection-profile image per frame. Inspect these to confirm that
   the threshold and detected region remain physically sensible over time.

3. mapping_points_overlay.tif

   RGB TIFF stack showing point mapping on the original images:

       cyan   = non-bleached contour points
       yellow = bleached-region contour points
       red    = contour index 0

   This file is essential for checking that contour orientation, starting point,
   and bleached-region assignment are consistent.

4. Displacement-vector quiver plot

   The current contour is shown as a partially transparent black line, the
   bleached region is marked in blue, and displacement arrows are crimson with
   black edges.

Quality-control checklist
-------------------------

Before using the CSV outputs:

1. Confirm that TIFF and contour files describe the same frames.
2. Confirm that the stack is properly registered.
3. Inspect the mapping-points overlay through the entire acquisition.
4. Verify that contour index 0 remains in the same anatomical location.
5. Confirm that the detected bleached region corresponds to the bleach ROI.
6. Inspect all-frame detection profiles.
7. Check that the fixed threshold does not select unrelated dark regions.
8. Confirm that the contour does not reverse direction or jump between frames.
9. Check the quiver plot for implausible correspondence or global drift.
10. Confirm the pixel size and time-axis settings before interpreting physical
    units.

Common problems
---------------

No contour frames overlap the TIFF
    Check that TIFF and contour frame numbering correspond and that the contour
    file contains all expected frames.

POSTBLEACH_BLOCKS do not cover all frames
    Adjust the frame counts in POSTBLEACH_BLOCKS so their total equals the
    number of postbleach TIFF frames.

Bleached region is assigned incorrectly
    Inspect the all-frame detection profiles. Adjust BLEACHED_PERCENTILE,
    BLEACHED_SMOOTHING_WINDOW, gap filling, or isolated-object removal. If
    necessary, use manual arc indices.

Detected region grows or moves unrealistically
    Consider BLEACHED_REGION_MODE = "fixed" if the physical bleach ROI should
    remain fixed, or refine the fixed threshold and detection settings.

Contour mapping is inconsistent
    Inspect mapping_points_overlay.tif. Confirm that the contour starts at the
    top and has a consistent orientation. Poor JFilament contours should be
    corrected before analysis.

Strain values are unexpectedly large
    Check point correspondence, contour segmentation, drift correction, and
    region boundaries. Local strain is highly sensitive to incorrect mapping
    or abrupt contour irregularities.

Output images are very large
    Reduce PLOT_DPI from 1200 to 600 or 300 if required.

Reproducibility recommendations
-------------------------------

For every analyzed GUV, archive:

1. the original and registered TIFF stacks;
2. the JFilament contour file;
3. the exact version of ablation_analysis.py;
4. all USER SETTINGS;
5. the output CSV files;
6. all quality-control plots and overlays;
7. notes describing any manual arc definition or parameter adjustment.

Keep parameter values consistent across experiments whenever possible. Any
experiment-specific changes should be documented before comparing samples.

Relationship to the comparison script
-------------------------------------

Run this script independently for every GUV. Each run creates a
"guv_deformation_analysis" output folder. The comparison script uses these two
files from each output folder:

    global_metrics_relative_to_prebleach.csv
    regional_metrics_relative_to_prebleach.csv

Do not run the comparison script until every individual analysis has passed
quality control.
