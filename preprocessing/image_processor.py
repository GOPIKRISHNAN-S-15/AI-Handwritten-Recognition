"""
Adaptive image preprocessing pipeline using OpenCV.

Analyzes image characteristics (brightness, contrast, noise, background)
and selects appropriate preprocessing operations automatically.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any


@dataclass
class ImageAnalysis:
    """Results from analyzing an image's characteristics."""
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    background_intensity: float = 0.0
    is_low_contrast: bool = False
    is_noisy: bool = False
    is_dark: bool = False
    is_bright: bool = False
    has_uneven_background: bool = False
    skew_angle: float = 0.0
    is_skewed: bool = False
    original_size: Tuple[int, int] = (0, 0)
    applied_operations: List[str] = field(default_factory=list)


class ImageAnalyzer:
    """Analyzes basic image characteristics to guide adaptive preprocessing."""

    @staticmethod
    def analyze(image: np.ndarray) -> ImageAnalysis:
        """
        Analyze image characteristics.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            ImageAnalysis with detected characteristics.
        """
        analysis = ImageAnalysis()
        analysis.original_size = image.shape[:2]

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Brightness (mean pixel intensity)
        analysis.brightness = float(np.mean(gray))
        analysis.is_dark = analysis.brightness < 80
        analysis.is_bright = analysis.brightness > 200

        # Contrast (standard deviation of pixel intensities)
        analysis.contrast = float(np.std(gray))
        analysis.is_low_contrast = analysis.contrast < 40

        # Noise estimation (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        analysis.noise_level = float(laplacian.var())
        analysis.is_noisy = analysis.noise_level > 1500

        # Background intensity (intensity of border regions)
        h, w = gray.shape
        border_size = max(2, min(h, w) // 10)
        border_pixels = np.concatenate([
            gray[:border_size, :].flatten(),
            gray[-border_size:, :].flatten(),
            gray[:, :border_size].flatten(),
            gray[:, -border_size:].flatten(),
        ])
        analysis.background_intensity = float(np.mean(border_pixels))

        # Check for uneven background
        quadrants = [
            gray[:h // 2, :w // 2],
            gray[:h // 2, w // 2:],
            gray[h // 2:, :w // 2],
            gray[h // 2:, w // 2:],
        ]
        quad_means = [float(np.mean(q)) for q in quadrants]
        analysis.has_uneven_background = (max(quad_means) - min(quad_means)) > 50

        # Skew detection using moments
        analysis.skew_angle = ImageAnalyzer._detect_skew(gray)
        analysis.is_skewed = abs(analysis.skew_angle) > 2.0

        return analysis

    @staticmethod
    def _has_uniform_border(gray: np.ndarray) -> bool:
        """
        True when the image has a pure/flat background border, i.e. it is a
        scanned document or drawing with margin around the content.
        False for tight single-character crops (ink touches all edges).
        Used to decide whether document-level operations (deskew, denoise,
        CLAHE) should be applied.
        """
        h, w = gray.shape
        if h < 5 or w < 5:
            return False
        # Border strips (5% of each side)
        t = max(1, min(h, w) // 20)
        strips = [
            gray[:t, :],
            gray[-t:, :],
            gray[:, :t],
            gray[:, -t:],
        ]
        return all(float(np.std(s)) < 12.0 for s in strips)

    @staticmethod
    def _detect_skew(gray: np.ndarray) -> float:
        """Detect skew angle using image moments."""
        try:
            # Threshold the image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0

            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            if len(largest) < 5:
                return 0.0

            # Fit an ellipse if possible
            rect = cv2.minAreaRect(largest)
            angle = rect[-1]

            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90

            return float(angle)
        except Exception:
            return 0.0


class AdaptivePreprocessor:
    """
    Applies adaptive image preprocessing based on image analysis.
    
    Standardizes handwritten input to match MNIST training distributions:
    - Adaptive polarity detection (white stroke on dark background)
    - Aspect-preserving scale into a 20x20 bounding box
    - Center-of-mass centering on a 28x28 field
    - Pixel normalization [0.0, 1.0]
    """

    def __init__(self, target_size: Tuple[int, int] = (28, 28)):
        self.target_size = target_size
        self.analyzer = ImageAnalyzer()

    def preprocess_with_debug(self, image: np.ndarray) -> Tuple[np.ndarray, ImageAnalysis, Dict[str, np.ndarray]]:
        """
        Preprocess an image while saving all intermediate debug stages.

        Returns:
            Tuple of (model_ready_array, analysis_report, debug_stages_dict)
        """
        debug_stages: Dict[str, np.ndarray] = {}
        analysis = self.analyzer.analyze(image)

        # 1. Original
        debug_stages["original"] = image.copy()

        # 2. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            analysis.applied_operations.append("Grayscale conversion")
        else:
            gray = image.copy()
        debug_stages["grayscale"] = gray.copy()

        # 3. Deskew if needed (only meaningful for documents with a border).
        # Guard: skip all document-level operations on dense multi-character
        # canvases. When the ink ratio is high the image is a composition of
        # several characters (e.g. a drawing-canvas capture of "2026"), and
        # smoothing/deskewing blurs stroke junctions that the segmenter and
        # classifier depend on. A scanned document has sparse ink (typically
        # <10%), while a dense multi-digit canvas exceeds it.
        ink_ratio = 1.0 - (gray > 127).mean() if gray.dtype != np.bool_ else float(gray.mean())
        dense_canvas = ink_ratio > 0.10
        if analysis.is_skewed and ImageAnalyzer._has_uniform_border(gray) and not dense_canvas:
            gray = self._deskew(gray, analysis.skew_angle)
            analysis.applied_operations.append(f"Deskewing ({analysis.skew_angle:.1f}°)")

        # 4. Denoise if noisy. Applied only to scanned/uploaded documents
        # (which carry a uniform background border). Skipping denoising on
        # tight single-character crops preserves the exact stroke geometry
        # the classifier was trained on; smoothing alters stroke edges and
        # measurably hurts per-character accuracy (e.g. digit '0' flipping
        # to '4' after denoising a bordered crop).
        if analysis.is_noisy and ImageAnalyzer._has_uniform_border(gray) and not dense_canvas:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            analysis.applied_operations.append("Noise reduction (fastNlMeansDenoising)")

        # 5. Contrast enhancement if low contrast (documents only)
        if analysis.is_low_contrast and ImageAnalyzer._has_uniform_border(gray) and not dense_canvas:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            analysis.applied_operations.append("Contrast enhancement (CLAHE)")

        # 6. Polarity & Binarization (Foreground = 255 White, Background = 0 Black)
        is_light_on_dark = analysis.background_intensity < 127
        thresh_type = cv2.THRESH_BINARY if is_light_on_dark else cv2.THRESH_BINARY_INV

        # Generate inverted grayscale image (ink is white, background is black)
        gray_inverted = gray if is_light_on_dark else 255 - gray
        debug_stages["gray_inverted"] = gray_inverted.copy()

        is_large_image = image.shape[0] > 100 or image.shape[1] > 100

        # We create a binary mask purely to find the bounding box
        if analysis.has_uneven_background and not is_light_on_dark and is_large_image:
            binary_mask = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresh_type,
                blockSize=11, C=5
            )
            analysis.applied_operations.append("Adaptive thresholding (uneven background) for mask")
        else:
            _, binary_mask = cv2.threshold(
                gray, 0, 255,
                thresh_type + cv2.THRESH_OTSU
            )
            analysis.applied_operations.append(f"Otsu binarization ({'normal' if is_light_on_dark else 'inverted'}) for mask")

        debug_stages["binarized"] = binary_mask.copy()

        # 7. Crop bounding box of content.
        ink_rows = (binary_mask > 0).any(axis=1)
        ink_cols = (binary_mask > 0).any(axis=0)
        has_uniform_border = not (
            ink_rows[0] and ink_rows[-1] and ink_cols[0] and ink_cols[-1]
        )

        coords = cv2.findNonZero(binary_mask)
        if coords is not None and has_uniform_border:
            x, y, cw, ch = cv2.boundingRect(coords)
            # CRUCIAL DIFFERENCE: Crop the grayscale inverted image, not the binary mask!
            cropped = gray_inverted[y:y+ch, x:x+cw]
            analysis.applied_operations.append(f"Content cropped ({cw}×{ch}px)")
        else:
            cropped = gray_inverted
            cw, ch = gray_inverted.shape[1], gray_inverted.shape[0]
            if coords is not None:
                analysis.applied_operations.append("Tight character crop (no border cropping)")

        debug_stages["cropped"] = cropped.copy()

        # 8. Pad to square BEFORE resizing to 20x20 (MNIST standard)
        th, tw = self.target_size
        box_size = 20
        if cw > 0 and ch > 0:
            max_dim = max(cw, ch)
            square = np.zeros((max_dim, max_dim), dtype=cropped.dtype)
            y_off = (max_dim - ch) // 2
            x_off = (max_dim - cw) // 2
            square[y_off:y_off+ch, x_off:x_off+cw] = cropped
            
            scale = box_size / float(max_dim)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
            resized_square = cv2.resize(square, (box_size, box_size), interpolation=interp)
        else:
            resized_square = np.zeros((box_size, box_size), dtype=np.uint8)

        # 9. Center in 28x28 canvas geometrically first
        canvas = np.zeros((th, tw), dtype=np.uint8)
        y_start = (th - box_size) // 2
        x_start = (tw - box_size) // 2
        canvas[y_start:y_start+box_size, x_start:x_start+box_size] = resized_square

        # 9.5 Contrast stretch (Normalization to span 0-255 fully)
        min_val = canvas.min()
        max_val = canvas.max()
        if max_val > min_val and max_val < 255:
            canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
            analysis.applied_operations.append("Contrast stretch to 0-255")

        # 10. Center of Mass Shift (Standard MNIST Normalization)
        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shift_x = float(np.round((tw / 2.0) - cx))
            shift_y = float(np.round((th / 2.0) - cy))
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(
                canvas, M, (tw, th),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            analysis.applied_operations.append("Center-of-mass alignment (14, 14)")

        debug_stages["model_input"] = canvas.copy()

        # Normalized float32 for model
        model_input = canvas.astype(np.float32) / 255.0
        analysis.applied_operations.append("Pixel normalization (0.0-1.0)")

        return model_input, analysis, debug_stages

    def preprocess(
        self,
        image: np.ndarray,
        for_model: bool = True,
    ) -> Tuple[np.ndarray, ImageAnalysis]:
        """Adaptively preprocess an image."""
        model_input, analysis, debug = self.preprocess_with_debug(image)
        if for_model:
            return model_input, analysis
        return debug["model_input"], analysis

    def preprocess_for_display(self, image: np.ndarray) -> Tuple[np.ndarray, ImageAnalysis]:
        """Preprocess for display (uint8 0-255 image)."""
        _, analysis, debug = self.preprocess_with_debug(image)
        return debug["model_input"], analysis

    # ------------------------------------------------------------------
    # Multi-character segmentation (vertical projection splitting)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_vertical_projection(
        strip: np.ndarray,
        min_gap_width: int = 1,
        column_ink_ratio: float = 0.03,
        min_segment_width: int = 3,
    ) -> List[Tuple[int, int]]:
        """
        Split a binary blob into character columns using the vertical
        projection profile.

        A column is considered ink-bearing when at least `column_ink_ratio`
        of its pixels contain ink. Ink columns separated by a gap wider than
        `min_gap_width` are treated as separate characters. Narrow residual
        strips (< min_segment_width px) are discarded as stroke noise.

        Returns a list of (start_col, end_col) pairs (end exclusive).
        """
        h = strip.shape[0]
        col_ink = strip.sum(axis=0) / 255.0
        ink_threshold = max(1.0, h * column_ink_ratio)
        is_ink = col_ink > ink_threshold

        segments: List[Tuple[int, int]] = []
        in_segment = False
        start = 0
        gap = 0

        for i, v in enumerate(is_ink):
            if v and not in_segment:
                start, in_segment, gap = i, True, 0
            elif not v and in_segment:
                gap += 1
                if gap > min_gap_width:
                    segments.append((start, i - gap + 1))
                    in_segment = False
            else:
                gap = 0

        if in_segment:
            segments.append((start, len(is_ink)))

        return [(s, e) for s, e in segments if (e - s) >= min_segment_width]

    @staticmethod
    def _median_height(boxes: List[Tuple[int, int, int, int]]) -> float:
        heights = sorted(b[3] for b in boxes)
        return float(np.median(heights)) if heights else 0.0

    def _split_wide_component(
        self,
        binary: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        height_scale: float,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Attempt to split one wide connected component into multiple
        character boxes via vertical projection.

        Returns None when the component cannot be safely split (keeps the
        whole box, flagged as a potential merged glyph by the caller).
        """
        strip = binary[y:y + h, x:x + w]
        sub = self._split_by_vertical_projection(strip)
        if len(sub) < 2:
            return None

        boxes = []
        for (s1, s2) in sub:
            bw = s2 - s1
            bh = h
            # Reject segments whose aspect is implausible for a digit glyph
            if bw / max(bh, 1) > 2.5:
                return None
            boxes.append((x + s1, y, bw, bh))
        return boxes

    def _merge_overlapping_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        min_overlap: int = 2,
    ) -> List[Tuple[int, int, int, int]]:
        """Merge boxes whose x-ranges overlap (protects slightly overlapping strokes)."""
        if len(boxes) <= 1:
            return boxes
        merged = []
        current = list(boxes[0])
        for (x, y, w, h) in boxes[1:]:
            if x < current[0] + current[2] - min_overlap:
                # x-ranges overlap -> merge into one box
                current[2] = max(current[2], (x + w) - current[0])
                current[1] = min(current[1], y)
                current[3] = max(current[3], (y + h) - current[1])
            else:
                merged.append(tuple(current))
                current = [x, y, w, h]
        merged.append(tuple(current))
        return merged

    def segment_characters(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Segments an image containing multiple characters into individual
        character crops.

        Pipeline:
        1. Grayscale + adaptive thresholding (polarity-aware)
        2. Morphological opening to remove isolated noise dots
        3. Connected-component contour detection
        4. Wide-component split via vertical projection profile
        5. Overlap merge for slightly overlapping strokes
        6. Height-scale consistency check -> tall/short boxes are merged
           with neighbors or kept with a wide-glyph flag
        7. Left-to-right sorting and padded cropping

        Returns a list of (cropped_image, (x, y, w, h)) sorted left to right.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        analysis = self.analyzer.analyze(image)
        is_light_on_dark = analysis.background_intensity < 127
        thresh_type = cv2.THRESH_BINARY if is_light_on_dark else cv2.THRESH_BINARY_INV

        _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

        # Remove isolated noise dots (2x2 morphological open) ONLY for
        # component detection. Erosion shrinks ink by 1 px per side, so
        # bounding boxes computed on the opened binary would cut off the
        # outermost stroke pixels and measurably hurt classification
        # (e.g. digit '0' flipping to '4'). Crops are therefore taken from
        # the ORIGINAL binary's ink extents.
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        # 1. Connected components via contours (on the cleaned binary)
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = image.shape[:2]
        raw_boxes: List[Tuple[int, int, int, int]] = []
        kernel = np.ones((3, 3), np.uint8)
        for c in contours:
            if cv2.contourArea(c) <= 50:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Expand the component bbox and recover the TRUE ink extents
            # from the original (pre-open) binary so no stroke pixels are lost.
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(img_w, x + w + 1), min(img_h, y + h + 1)
            region_ink = (binary[y0:y1, x0:x1] > 0)
            if not region_ink.any():
                continue
            ry, rx = np.where(region_ink)
            raw_boxes.append((x0 + int(rx.min()), y0 + int(ry.min()),
                              int(rx.max() - rx.min()) + 1,
                              int(ry.max() - ry.min()) + 1))

        # 2. Merge boxes with overlapping x-ranges (slightly overlapping strokes)
        raw_boxes.sort(key=lambda b: b[0])
        merged_boxes = self._merge_overlapping_boxes(raw_boxes, min_overlap=2)

        # 3. Split wide components with vertical projection
        median_h = self._median_height(merged_boxes)
        final_boxes: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in merged_boxes:
            aspect = w / max(h, 1)
            # A single digit glyph is rarely wider than 1.5x its height.
            # Components that fail this are candidates for projection splitting.
            if aspect > 1.5:
                sub = self._split_wide_component(binary, x, y, w, h, median_h)
                if sub:
                    final_boxes.extend(sub)
                    continue
            final_boxes.append((x, y, w, h))

        # 4. Sort left to right (stable after merge/split)
        final_boxes.sort(key=lambda b: b[0])

        # 5. Crop each segment tightly around the content.
        # No extra padding is added: a surrounding border of uniform background
        # makes the adaptive analysis (denoising, adaptive thresholding on
        # "uneven background") misbehave, producing inverted binarization on
        # otherwise clean glyphs. Tight crops match what the classifier was
        # trained on (MNIST-style single-character images).
        segments: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for (x, y, w, h) in final_boxes:
            y_start = max(0, y)
            y_end = min(image.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(image.shape[1], x + w)
            crop = image[y_start:y_end, x_start:x_end].copy()
            segments.append((crop, (x_start, y_start, x_end - x_start, y_end - y_start)))

        # If no valid segments found, return the whole image as one segment
        if not segments:
            segments.append((image.copy(), (0, 0, image.shape[1], image.shape[0])))

        return segments

    @staticmethod
    def _deskew(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

