"""
Document segmentation pipeline for extracting individual characters
from handwritten document images.
"""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CharacterRegion:
    """A detected character region within a document."""
    image: np.ndarray          # Cropped character image
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    line_index: int = 0        # Which line this character belongs to
    word_index: int = 0        # Which word within the line
    char_index: int = 0        # Position within the word
    confidence: float = 0.0    # Will be filled after recognition


class DocumentSegmenter:
    """
    Segments a handwritten document image into individual characters.
    
    Pipeline:
        1. Preprocessing (binarization)
        2. Line detection via horizontal projection
        3. Character detection via contours within each line
        4. Sorting and extraction of character regions
    """

    def __init__(
        self,
        min_char_area: int = 20,
        max_char_area_ratio: float = 0.8,
        line_gap_threshold: int = 5,
        char_padding: int = 0,
    ):
        self.min_char_area = min_char_area
        self.max_char_area_ratio = max_char_area_ratio
        self.line_gap_threshold = line_gap_threshold
        self.char_padding = char_padding

    def segment(self, image: np.ndarray) -> Tuple[List[CharacterRegion], np.ndarray]:
        """
        Segment a document image into individual characters.
        """
        print("\n" + "="*50)
        print(">>> EXECUTING NEW DUAL-PROJECTION SEGMENTATION <<<")
        print("="*50 + "\n")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew the document image to ensure horizontal lines
        gray = self._deskew(gray)

        # Binarize
        binary = self._binarize(gray)

        # Detect lines via horizontal projection
        line_regions = self._detect_lines(binary)

        # Detect characters within each line
        characters: List[CharacterRegion] = []
        for line_idx, (y_start, y_end) in enumerate(line_regions):
            line_strip = binary[y_start:y_end, :]
            char_bboxes = self._detect_characters_in_line(line_strip, line_index=line_idx)

            for char_idx, (cx, cy, cw, ch, word_idx) in enumerate(char_bboxes):
                # Absolute coordinates
                abs_y = y_start + cy
                region = CharacterRegion(
                    image=self._extract_char(gray, cx, abs_y, cw, ch),
                    bbox=(cx, abs_y, cw, ch),
                    line_index=line_idx,
                    word_index=word_idx,
                    char_index=char_idx,
                )
                characters.append(region)

        # If line detection didn't work well, fall back to full-image contour detection
        if not characters:
            characters = self._fallback_detection(gray, binary)

        # Create annotated image
        annotated = self._draw_annotations(image, characters)

        return characters, annotated

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Deskew the image using Hough Line Transform on text blocks."""
        # Binarize for deskewing
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 5
        )
        
        # Dilate horizontally to connect text into lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Use Hough Lines to find the dominant angle
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line.flatten()
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (-45 to 45 degrees)
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                # Rotate image
                h, w = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                gray = cv2.warpAffine(
                    gray, M, (w, h), 
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                
        return gray

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Binarize the image using adaptive thresholding."""
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold works better for documents
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=5
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    def _detect_lines(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect text lines using horizontal projection profile.
        
        Returns list of (y_start, y_end) for each detected line.
        """
        h, w = binary.shape
        
        # 1. Remove vertical lines (like notebook margins) that would connect text lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        binary_no_vert = cv2.subtract(binary, vertical_lines)
        
        # Horizontal projection (sum of white pixels per row)
        h_proj = np.sum(binary_no_vert, axis=1) / 255.0
        max_proj = np.max(h_proj)
        
        if max_proj == 0:
            return []
            
        # Dynamic noise threshold (e.g., 2% of maximum projection)
        noise_threshold = max_proj * 0.02
        is_text = h_proj > noise_threshold
        
        # Minimum constraints
        min_line_height = max(5, h // 200)
        min_gap_height = max(3, h // 300)
        
        lines = []
        in_line = False
        start = 0
        
        for i in range(h):
            if is_text[i] and not in_line:
                start = i
                in_line = True
            elif not is_text[i] and in_line:
                # Check if this is a real gap (wider than min_gap_height)
                # Look ahead
                is_real_gap = True
                for j in range(i, min(i + min_gap_height, h)):
                    if is_text[j]:
                        is_real_gap = False
                        break
                        
                if is_real_gap:
                    end = i
                    if (end - start) >= min_line_height:
                        lines.append((start, end))
                    in_line = False
                    
        # Close any open line
        if in_line:
            end = h
            if (end - start) >= min_line_height:
                lines.append((start, end))
                
        # If no lines detected, treat entire image as one line
        if not lines:
            lines = [(0, h)]
            
        return lines

    def _detect_characters_in_line(
        self, line_binary: np.ndarray, line_index: int = 0
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Detect individual characters and group them into words using vertical projection.
        Returns list of (x, y, w, h, word_index) bounding boxes sorted left to right.
        """
        h, w = line_binary.shape
        
        # Vertical projection (sum of white pixels per column)
        v_proj = np.sum(line_binary, axis=0) / 255.0
        max_proj = np.max(v_proj)
        
        if max_proj == 0:
            return []
            
        noise_threshold = max(1.0, max_proj * 0.05)
        is_ink = v_proj > noise_threshold
        
        # 1. Find character bounds (contiguous ink regions horizontally)
        char_bounds = []
        in_char = False
        start = 0
        min_char_width = 2
        
        for i in range(w):
            if is_ink[i] and not in_char:
                start = i
                in_char = True
            elif not is_ink[i] and in_char:
                end = i
                if (end - start) >= min_char_width:
                    char_bounds.append((start, end))
                in_char = False
                
        if in_char:
            end = w
            if (end - start) >= min_char_width:
                char_bounds.append((start, end))
                
        if not char_bounds:
            return []
            
        # 2. Identify words based on gap width between character bounds
        gaps = []
        for i in range(len(char_bounds) - 1):
            gap = char_bounds[i+1][0] - char_bounds[i][1]
            gaps.append(gap)
            
        if gaps:
            median_gap = float(np.median(gaps))
            # Word gap threshold: e.g. 2.0x the median character gap
            word_gap_threshold = max(5.0, median_gap * 2.0)
        else:
            word_gap_threshold = 9999.0
            
        word_index = 0
        bboxes = []
        
        for i, (x_start, x_end) in enumerate(char_bounds):
            if i > 0:
                gap = x_start - char_bounds[i-1][1]
                if gap > word_gap_threshold:
                    word_index += 1
                    
            # Extract the actual vertical boundaries for this character via horizontal projection
            char_strip = line_binary[:, x_start:x_end]
            h_proj = np.sum(char_strip, axis=1)
            y_coords = np.where(h_proj > 0)[0]
            
            if len(y_coords) > 0:
                y_start = y_coords[0]
                y_end = y_coords[-1] + 1
                cw = x_end - x_start
                ch = y_end - y_start
                # Filter out pure noise
                if cw > 1 and ch > 4:
                    bboxes.append((x_start, y_start, cw, ch, word_index))
                    
        return bboxes

    def _fallback_detection(
        self, gray: np.ndarray, binary: np.ndarray
    ) -> List[CharacterRegion]:
        """Fallback: detect characters from entire image using contours."""
        h, w = binary.shape
        max_area = h * w * self.max_char_area_ratio

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        characters: List[CharacterRegion] = []
        bboxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < self.min_char_area or area > max_area:
                continue
            bboxes.append((x, y, cw, ch))

        # Sort by position: top-to-bottom, then left-to-right
        bboxes.sort(key=lambda b: (b[1] // 30, b[0]))  # Group by ~30px rows

        for idx, (x, y, cw, ch) in enumerate(bboxes):
            region = CharacterRegion(
                image=self._extract_char(gray, x, y, cw, ch),
                bbox=(x, y, cw, ch),
                line_index=0,
                char_index=idx,
            )
            characters.append(region)

        return characters

    def _extract_char(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Extract and pad a character region."""
        p = self.char_padding
        img_h, img_w = gray.shape[:2]

        x1 = max(0, x - p)
        y1 = max(0, y - p)
        x2 = min(img_w, x + w + p)
        y2 = min(img_h, y + h + p)

        return gray[y1:y2, x1:x2]

    def _draw_annotations(
        self, image: np.ndarray, characters: List[CharacterRegion]
    ) -> np.ndarray:
        """Draw bounding boxes on the image for visualization."""
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        # Color palette for different lines
        colors = [
            (0, 212, 255),   # Cyan
            (124, 58, 237),  # Purple
            (236, 72, 153),  # Pink
            (16, 185, 129),  # Green
            (245, 158, 11),  # Orange
        ]

        for char in characters:
            x, y, w, h = char.bbox
            color = colors[char.line_index % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        return annotated


def detect_spaces(
    bboxes: List[Tuple[int, int, int, int]],
    space_threshold_multiplier: float = 1.5,
) -> List[int]:
    """
    Detect likely space positions between characters based on
    horizontal gaps between bounding boxes.

    Args:
        bboxes: List of (x, y, w, h) bounding boxes sorted left to right.
        space_threshold_multiplier: How many times the median gap = a space.

    Returns:
        List of indices after which a space should be inserted.
    """
    if len(bboxes) < 2:
        return []

    # Calculate gaps between consecutive characters
    gaps = []
    for i in range(len(bboxes) - 1):
        x1_end = bboxes[i][0] + bboxes[i][2]
        x2_start = bboxes[i + 1][0]
        gaps.append(x2_start - x1_end)

    if not gaps:
        return []

    median_gap = float(np.median(gaps))
    threshold = median_gap * space_threshold_multiplier

    space_indices = [i for i, gap in enumerate(gaps) if gap > threshold]
    return space_indices
