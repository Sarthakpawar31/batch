"""
vision/leaf_detection.py

Lightweight leaf / plant detector using classical computer vision
(HSV colour segmentation + contour analysis).  No heavy DNN required
for this stage – keeps RAM and CPU usage minimal.

Pipeline:
  BGR frame → HSV → green-range mask → morphological cleanup
            → contour analysis → bounding boxes + confidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CAM_CFG

logger = logging.getLogger(__name__)

# ── HSV range for "green" vegetation ─────────────────────────────────────────
# Hue: 35–85 °  (yellow-green → cyan-green)
# Saturation: 40–255 (avoid grey, washed-out)
# Value: 30–255 (avoid very dark areas)
_HSV_LOW  = np.array([35,  40,  30],  dtype=np.uint8)
_HSV_HIGH = np.array([85, 255, 255],  dtype=np.uint8)

# Morphological kernel
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass
class LeafRegion:
    """A detected plant/leaf region in the image."""
    x:          int        # top-left x
    y:          int        # top-left y
    w:          int        # width
    h:          int        # height
    area:       int        # pixel area
    confidence: float      # 0–1, based on green-pixel density in ROI
    roi:        np.ndarray # cropped BGR region (for inference)


def detect_leaves(frame: np.ndarray) -> List[LeafRegion]:
    """
    Detect green plant regions in a BGR frame.

    Returns a list of LeafRegion objects sorted by area (largest first).
    Returns an empty list when no significant vegetation is found.
    """
    if frame is None or frame.size == 0:
        return []

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)

    # Noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    regions: List[LeafRegion] = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < CAM_CFG.LEAF_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y + h, x:x + w]

        # Confidence = fraction of green pixels in the ROI
        roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_mask = cv2.inRange(roi_hsv, _HSV_LOW, _HSV_HIGH)
        confidence = float(np.count_nonzero(roi_mask)) / float(w * h)

        regions.append(LeafRegion(
            x=x, y=y, w=w, h=h,
            area=area,
            confidence=round(confidence, 3),
            roi=roi,
        ))

    # Sort by area descending – analyse the biggest leaf first
    regions.sort(key=lambda r: r.area, reverse=True)
    logger.debug("Leaf detection: %d region(s) found.", len(regions))
    return regions


def best_leaf(frame: np.ndarray) -> Optional[LeafRegion]:
    """Return the largest, highest-confidence leaf, or None."""
    regions = detect_leaves(frame)
    if not regions:
        return None
    # prefer large area with reasonable confidence
    return max(regions, key=lambda r: r.area * r.confidence)


def draw_detections(frame: np.ndarray,
                    regions: List[LeafRegion]) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the frame (debug / logging only).
    Never displayed in production – written to disk for review.
    """
    out = frame.copy()
    for r in regions:
        color = (0, 200, 50)
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)
        cv2.putText(out, f"{r.confidence:.2f}", (r.x, r.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
