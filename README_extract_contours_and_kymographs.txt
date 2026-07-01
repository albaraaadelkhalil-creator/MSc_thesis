GUV Cortex Contour Extraction and Contour-Kymograph Generation

Overview
`extract\_contours\_and\_kymographs.py` is a Python analysis pipeline for extracting and tracking the outer actin-cortex contour of a single giant unilamellar vesicle (GUV) across a time-lapse TIFF stack. This code works with branched and linear cortices. The script was developed for fluorescence time-lapse data in which the cortical signal decreases over time because of photobleaching and or photodamage. Under these conditions, manual or semi-automated contour tracking with tools such as JFilament may become unreliable at later time points. This script therefore combines intensity-based segmentation, object tracking, radial contour extraction, contour-shape quantification, and optional contour-kymograph generation.

The pipeline:
loads a 2D TIFF image or 3D TIFF time series;
optionally applies Gaussian smoothing;
segments each frame into intensity classes using multi-Otsu or a Gaussian mixture model;
identifies and tracks one GUV over time;
extracts the outer cortex contour using radial intensity profiles;
calculates contour-based shape descriptors;
saves masks, coordinates, metrics, overlays, thresholds, and quality-control outputs;
optionally generates a contour kymograph.

The current implementation assumes that the GUV of interest is the largest segmented object in the first frame and that its cortex appears as a bright peripheral feature around a dimmer interior.

Author: Albaraa Adel Farouk Khalil 
Last updated: 27.06.2026
ChatGPT was used as a supporting tool during code development. The author remains responsible for the analysis logic, parameter selection, validation, interpretation, and final use of the generated outputs.

Requirements
The script requires Python 3.10 or later and the following packages:
matplotlib
numpy
scipy
scikit-image
scikit-learn
tifffile

Input data
The input must be either a single 2D TIFF image or a 3D TIFF stack organized as: [frame, y, x]. If a 2D image is supplied, the script automatically adds a frame dimension and treats it as a one-frame stack. The input should contain the fluorescence channel used to visualize the actin cortex. The script does not currently support multichannel TIFF stacks directly. If the original acquisition contains multiple channels, export the relevant actin channel as a separate TIFF stack before running the script.

How to use
Open `extract\_contours\_and\_kymographs.py`.
Locate the `AnalysisConfig` class near the top of the script.
Replace `input\_path` with the path to the TIFF stack.
Adjust the analysis parameters if needed.

The script creates an output folder next to the input TIFF file. Example: input_path: Path = Path(r"C:\path\to\your\time_lapse_stack.tif") By default, the output folder is: output_segmented

Segmentation settings
`method`
Thresholding method used to divide each image into intensity classes. Allowed values: "otsu" or "gmm"
Multi-Otsu "otsu": Multi-Otsu thresholding divides the image into multiple intensity classes by maximizing separation between class intensity distributions. This is generally faster and more deterministic than the Gaussian mixture model.
Gaussian mixture model "gmm": The Gaussian mixture model fits multiple Gaussian distributions to the pixel intensities and defines thresholds between the fitted component means. It can be useful when intensity distributions overlap or are not separated well by Otsu thresholding, but it is slower and may be more sensitive to background structure.


`per\_frame\_thresholds`
Controls whether thresholds are recalculated independently for every frame. When `True`, the script adapts to intensity changes over time, including photobleaching. When `False`, one global pair of thresholds is calculated from the entire stack and applied to all frames. Per-frame thresholding is useful when the overall fluorescence intensity decreases substantially during the acquisition. However, because each frame is segmented independently, it may preserve low-intensity structures that would disappear under a fixed global threshold. The chosen thresholds are saved in a CSV file for inspection.

`n\_classes`: Number of intensity classes.
The current script is written for exactly three classes:
0 = background
1 = intermediate-intensity region, typically lumen
2 = bright region, typically cortex
The code expects two thresholds and raises an error if `n\_classes` does not equal 3.

`guv\_classes`
Intensity classes combined to create the initial GUV mask. The default combines the intermediate and bright classes, thereby including both lumenal and cortical fluorescence in the segmented GUV mask.

Optional preprocessing
`apply\_gaussian\_filter` Applies Gaussian smoothing before segmentation and radial contour extraction.The original stack is still retained for display and, by default, for kymograph generation.
`gaussian\_sigma\_px` Standard deviation of the Gaussian filter in pixels. Larger values suppress more high-frequency noise but can blur narrow cortical structures and shift the apparent contour position.

Binary-mask cleaning
`min\_guv\_area\_px` Minimum connected-component area in pixels. Objects smaller than this threshold are removed. Adjust it according to image magnification, pixel size, and expected GUV radius.
`closing\_radius\_px` Radius of the morphological closing operation. Closing fills small gaps and connects nearby foreground pixels. A larger value may repair fragmented masks but can also merge nearby objects.

GUV tracking: The script tracks one GUV over time. In the first frame, the largest valid connected object is selected. In later frames, each candidate object is scored according to centroid displacement, relative area change, and mask overlap with the previous frame. The score is conceptually: 
score = distance_weight × centroid displacement + area_weight × relative area change − overlap_weight × intersection-over-union
The candidate with the lowest score is selected.

`distance\_weight` Higher values penalize large frame-to-frame movement more strongly.
`area\_weight` Higher values favor objects with areas similar to the previously tracked GUV.
`overlap\_weight` Because overlap is subtracted from the score, higher values favor candidates that overlap strongly with the previous GUV mask. These weights should be adjusted if the GUV moves substantially, changes shape, shrinks, or overlaps with other fluorescent objects.
Frame-0 safety mask: When enabled, the script creates a dilated mask around the largest GUV detected in frame 0. In later frames, candidate GUV pixels are restricted to this region. This reduces the risk that tracking switches to another bright vesicle or fluorescent object elsewhere in the field of view.
`safety\_margin\_px` A larger value permits more movement and deformation. A value that is too small may remove genuine parts of the tracked GUV at later time points. Disable the safety mask if the GUV undergoes substantial translation.

Radial contour extraction
After the GUV has been tracked, the outer cortical contour is extracted independently in each frame. For each frame:
the tracked GUV centroid is used as the radial origin;
radial intensity profiles are sampled at evenly spaced angles;
each profile is smoothed using a Savitzky-Golay filter;
the brightest point is treated as the approximate cortex peak;
the strongest negative intensity gradient after the peak is selected as the outer cortex edge;
implausible radial positions are rejected;
rejected values are interpolated;
the full radius sequence is smoothed around the contour;
the radius-angle representation is converted into Cartesian contour coordinates.

This method is designed for approximately star-convex GUVs, meaning that the outer boundary can be described by one radius for each angle from the centroid.
Parameters:
`n\_contour\_points` At 720 points, the angular spacing is 0.5 degrees. Higher values provide denser sampling but do not necessarily improve accuracy beyond the image resolution.
`r\_min\_px` and ` r\_max\_px` The cortex must lie within this radial interval. Adapt these values to the expected vesicle radius in pixels.
`search\_margin\_px` After the first frame, radial searching is restricted to the previous median radius plus or minus this margin. This stabilizes contour extraction and reduces jumps to unrelated intensity structures.
`max\_radius\_jump\_px` Radial values farther than this distance from the median radius are treated as failed detections and replaced by interpolation. Smaller values create smoother contours; larger values preserve stronger local deformations but may retain noise.
`radius\_smooth\_window` Savitzky-Golay smoothing window applied around the full angular radius sequence. A larger window suppresses local irregularities, whereas a smaller window preserves more local deformation. This setting affects area, perimeter, circularity, and solidity and should be kept constant across compared samples.

Shape descriptors 
The script calculates the following contour metrics for every frame:

Area Polygon: area enclosed by the contour, reported in pixels squared.
Perimeter: Length of the closed contour, reported in pixels.
Circularity: circularity = 4π × area / perimeter², A perfect circle has a circularity of 1. Lower values indicate increasing deviation from a circle.
Solidity: solidity = contour area / convex hull area, A solidity near 1 indicates that the contour is close to convex. Lower values indicate concavities or irregular deformation.
Equivalent diameter: equivalent diameter = 2 × sqrt(area / π), This is the diameter of a circle with the same area as the contour and is reported in pixels.

To convert to physical units:
area in µm² = area_px × pixel_size_µm²
perimeter in µm = perimeter_px × pixel_size_µm
equivalent diameter in µm = equivalent_diameter_px × pixel_size_µm

The current script does not apply this calibration automatically.

Contour kymograph generation
A contour kymograph represents fluorescence intensity along the GUV contour over time. One axis corresponds to time, and the other corresponds to position around the contour.

`make\_kymographs` Set to `False` to skip kymograph generation.
`kymo\_width\_px` Width of the radial sampling band around the contour. When larger than 1, intensities are sampled at the contour and neighboring radial offsets and then averaged. Even values are automatically increased by 1 so that the band remains symmetric.
`kymo\_use\_filtered\_stack` When `False`, intensities are sampled from the original image stack. When `True`, they are sampled from the Gaussian-filtered processing stack.
`kymo\_save\_display\_uint8` When enabled, the script saves both a float32 quantitative kymograph and a percentile-normalized 8-bit display copy. The 8-bit display copy should not be used for quantitative intensity analysis.
`transpose\_kymograph` Before transposition, the kymograph is organized as `\[contour position, frame]`. After transposition, it is `\[frame, contour position]`.
`flip\_kymograph\_vertical` Flips the kymograph vertically for display.
Kymograph row rotation:  The contour is sampled from a defined angular starting position. By default, it starts on the right side of the GUV. The rows can be circularly shifted so that another position appears first.
rotate_kymograph_rows: bool = True
kymo_current_start: str = "right"
kymo_target_start: str = "top" Allowed positions are `right`, `bottom`, `left`, and `top`. A custom shift can be used with: kymo_target_start = "custom"


Output files
The output filenames include a run tag describing the thresholding and preprocessing settings, for example: otsu_perframe_no_gauss_safety15px
Three-class segmentation for visual inspection. Stored values are approximately 0, 127, and 254.
Threshold log.csv
Intensity histogram, Histogram of the processing stack with the final frame's thresholds shown as dashed lines.
Tracked GUV mask, Binary mask of the selected GUV in each frame.
Frame-0 safety mask, Saved only when a valid safety mask is created.
Contour mask, Binary TIFF stack containing the extracted contour.
Contour overlay, RGB quality-control stack showing the extracted contour in red over a percentile-normalized grayscale image.
Contour coordinates in NPZ format, Contains one NumPy array per valid frame. Keys follow `frame\_0000`, `frame\_0001`, and so on. Each array has shape `\[n\_contour\_points, 2]` with columns `\[y, x]`.
Contour coordinates in text format, Stores one-based frame numbers and columns:
Contour metrics.csv, frame,centroid_y,centroid_x,area_px,perimeter_px,circularity,solidity,equivalent_diameter_px. Frames without a valid GUV contain `NaN` values.
Quantitative contour kymograph.tiff, Float32 TIFF containing sampled intensities.
Display contour kymograph, Percentile-normalized 8-bit TIFF intended only for visualization.


Recommended quality-control workflow
Before using the quantitative outputs:
inspect the segmentation stack;
inspect the tracked GUV mask;
inspect the frame-0 safety mask;
inspect the contour overlay over the entire time series;
verify that the contour follows the outer cortex rather than the inner edge;
confirm that tracking does not switch to another vesicle;
inspect threshold values over time;
compare abrupt metric changes with the overlay;
verify the kymograph orientation;
exclude or rerun stacks with persistent contour failures.
The contour overlay should be inspected particularly before bleaching, immediately after bleaching, at late weak-signal time points, and during strong deformation or shrinkage.

Parameter-tuning guidance
Wrong GUV selected: Try a smaller `safety\_margin\_px`, larger `overlap\_weight`, or crop the TIFF around the intended GUV.
GUV lost because it moves: drift correct using StackReg in ImageJ 
Contour lies inside the cortex: Check `r\_min\_px`, `r\_max\_px`, and `search\_margin\_px`, and confirm that the strongest negative gradient corresponds to the intended outer boundary.
Contour jumps to background or nearby structures: Reduce `search\_margin\_px` and `max\_radius\_jump\_px`, or increase `radius\_smooth\_window`.
Contour is too smooth: Reduce `radius\_smooth\_window` or increase `max\_radius\_jump\_px`.
Contour is noisy: Enable Gaussian filtering, increase `radius\_smooth\_window`, or reduce `max\_radius\_jump\_px`.
GUV is not segmented: Try the GMM method, per-frame thresholds, a smaller `min\_guv\_area\_px`, or a larger `closing\_radius\_px`.
Nearby objects merge with the GUV: Reduce `closing\_radius\_px` or crop the field of view.

Assumptions and limitations
The script tracks only one GUV per run.
Your GUV is corrected for drift (recommendation: StackReg ImageJ)
The largest valid object in frame 0 is selected automatically.
The radial method assumes approximately star-convex geometry.
The contour is inferred from fluorescence intensity and its radial gradient.
The outer cortex boundary is operationally defined as the strongest negative gradient after the fluorescence peak.
The previous frame's median radius constrains the next search range.
Smoothing affects all contour-derived shape metrics.
Metrics are reported in pixel units unless calibrated separately.
The script does not estimate contour-position uncertainty.
Missing frames are retained as `NaN` in the metrics CSV.
Display-normalized outputs must not be used for quantitative intensity analysis.