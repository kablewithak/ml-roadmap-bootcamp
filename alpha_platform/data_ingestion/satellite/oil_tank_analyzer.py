"""
Oil storage tank shadow analysis for inventory estimation.

Analyzes satellite imagery of oil storage tanks to estimate fill levels
based on shadow patterns and floating roof positions.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class OilTankAnalyzer:
    """
    Analyze oil storage tanks from satellite imagery.

    Uses computer vision to detect tanks, measure shadows, and estimate
    fill levels based on floating roof height.
    """

    def __init__(
        self,
        min_tank_radius_m: float = 10.0,
        max_tank_radius_m: float = 100.0,
        image_resolution_m: float = 0.5,
    ):
        """
        Initialize oil tank analyzer.

        Args:
            min_tank_radius_m: Minimum tank radius in meters
            max_tank_radius_m: Maximum tank radius in meters
            image_resolution_m: Image resolution in meters per pixel
        """
        self.config = get_config()
        self.min_tank_radius_m = min_tank_radius_m
        self.max_tank_radius_m = max_tank_radius_m
        self.image_resolution_m = image_resolution_m

        # Convert to pixels
        self.min_tank_radius_px = int(min_tank_radius_m / image_resolution_m)
        self.max_tank_radius_px = int(max_tank_radius_m / image_resolution_m)

        logger.info(
            f"Initialized oil tank analyzer "
            f"(radius range: {min_tank_radius_m}-{max_tank_radius_m}m)"
        )

    def detect_tanks(
        self, image_path: str, visualize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect oil storage tanks in satellite imagery.

        Args:
            image_path: Path to satellite image
            visualize: Whether to create visualization

        Returns:
            List of detected tanks with metadata
        """
        logger.info(f"Detecting tanks in: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.min_tank_radius_px * 2,
            param1=50,
            param2=30,
            minRadius=self.min_tank_radius_px,
            maxRadius=self.max_tank_radius_px,
        )

        tanks = []
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i, (x, y, r) in enumerate(circles[0, :]):
                # Extract tank region
                tank_region = self._extract_tank_region(image, x, y, r)

                # Analyze tank
                tank_info = self._analyze_tank(tank_region, x, y, r)
                tank_info["tank_id"] = i
                tank_info["center_x"] = int(x)
                tank_info["center_y"] = int(y)
                tank_info["radius_px"] = int(r)
                tank_info["radius_m"] = float(r * self.image_resolution_m)

                tanks.append(tank_info)

        logger.info(f"Detected {len(tanks)} tanks")

        return tanks

    def _extract_tank_region(
        self, image: np.ndarray, x: int, y: int, r: int
    ) -> np.ndarray:
        """Extract region around detected tank."""
        # Extract square region around tank
        margin = int(r * 1.5)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + margin)
        y2 = min(image.shape[0], y + margin)

        return image[y1:y2, x1:x2]

    def _analyze_tank(
        self, tank_region: np.ndarray, x: int, y: int, r: int
    ) -> Dict[str, Any]:
        """
        Analyze individual tank to estimate fill level.

        Uses shadow analysis and floating roof detection.
        """
        gray = cv2.cvtColor(tank_region, cv2.COLOR_BGR2GRAY)

        # Detect floating roof (darker circle in center)
        roof_detected = self._detect_floating_roof(gray)

        # Analyze shadow
        shadow_info = self._analyze_shadow(tank_region)

        # Estimate fill level
        fill_level = self._estimate_fill_level(roof_detected, shadow_info)

        return {
            "has_floating_roof": roof_detected["detected"],
            "floating_roof_offset_px": roof_detected.get("offset", 0),
            "shadow_length_px": shadow_info["length"],
            "shadow_angle_deg": shadow_info["angle"],
            "estimated_fill_level": fill_level,
            "confidence": self._calculate_confidence(roof_detected, shadow_info),
        }

    def _detect_floating_roof(self, gray_tank: np.ndarray) -> Dict[str, Any]:
        """
        Detect floating roof in tank image.

        Floating roofs appear as darker circles that move up/down with fill level.
        """
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray_tank, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Look for circular contours in center
        center = np.array(gray_tank.shape[:2]) // 2
        min_area = (gray_tank.shape[0] * gray_tank.shape[1]) * 0.1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > 0.7:  # Reasonably circular
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Check if near center
                    offset = np.linalg.norm([cx - center[1], cy - center[0]])

                    return {
                        "detected": True,
                        "offset": float(offset),
                        "area": float(area),
                        "circularity": float(circularity),
                    }

        return {"detected": False}

    def _analyze_shadow(self, tank_region: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tank shadow to estimate height and fill level.

        Shadow length is proportional to tank height above ground.
        """
        gray = cv2.cvtColor(tank_region, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find shadow region (darker area extending from tank)
        # Simplified: find darkest regions
        dark_threshold = np.percentile(gray, 20)
        shadow_mask = gray < dark_threshold

        # Calculate shadow properties
        if np.any(shadow_mask):
            # Find shadow centroid
            moments = cv2.moments(shadow_mask.astype(np.uint8))
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # Estimate shadow length
                center = np.array(tank_region.shape[:2]) // 2
                shadow_vector = np.array([cx, cy]) - center[::-1]
                shadow_length = np.linalg.norm(shadow_vector)
                shadow_angle = np.degrees(np.arctan2(shadow_vector[1], shadow_vector[0]))

                return {
                    "length": float(shadow_length),
                    "angle": float(shadow_angle),
                    "detected": True,
                }

        return {"length": 0.0, "angle": 0.0, "detected": False}

    def _estimate_fill_level(
        self, roof_detected: Dict[str, Any], shadow_info: Dict[str, Any]
    ) -> float:
        """
        Estimate tank fill level from roof and shadow analysis.

        Returns:
            Fill level estimate (0.0 = empty, 1.0 = full)
        """
        # If floating roof detected, use offset
        if roof_detected.get("detected", False):
            # Larger offset means roof is lower (less full)
            offset = roof_detected.get("offset", 0)
            # Normalize offset (this is simplified - would need calibration)
            fill_level = max(0.0, min(1.0, 1.0 - (offset / 50.0)))
            return fill_level

        # Otherwise, use shadow length (longer shadow = higher roof = more full)
        if shadow_info.get("detected", False):
            shadow_length = shadow_info.get("length", 0)
            # Normalize shadow length (simplified)
            fill_level = max(0.0, min(1.0, shadow_length / 100.0))
            return fill_level

        # Default to 50% if no information
        return 0.5

    def _calculate_confidence(
        self, roof_detected: Dict[str, Any], shadow_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence in fill level estimate."""
        confidence = 0.0

        if roof_detected.get("detected", False):
            confidence += 0.6
            circularity = roof_detected.get("circularity", 0)
            confidence += 0.2 * circularity

        if shadow_info.get("detected", False):
            confidence += 0.2

        return min(confidence, 1.0)

    def analyze_facility(
        self,
        facility_name: str,
        images_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Analyze oil storage facility over time.

        Args:
            facility_name: Name of facility
            images_dir: Directory with time series of images
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Analysis including inventory trends
        """
        logger.info(f"Analyzing facility: {facility_name}")

        images_path = Path(images_dir)
        image_files = sorted(images_path.glob("*.jpg")) + sorted(
            images_path.glob("*.png")
        )

        if not image_files:
            raise ValueError(f"No images found in {images_dir}")

        # Process each image
        time_series_data = []

        for img_file in image_files:
            try:
                # Extract timestamp from filename
                date_str = img_file.stem.split("_")[0]
                timestamp = datetime.strptime(date_str, "%Y%m%d")

                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue

                # Detect and analyze tanks
                tanks = self.detect_tanks(str(img_file))

                # Calculate aggregate metrics
                if tanks:
                    avg_fill = np.mean([t["estimated_fill_level"] for t in tanks])
                    total_capacity = sum(
                        [np.pi * (t["radius_m"] ** 2) * 20 for t in tanks]
                    )  # Assume 20m height
                    estimated_volume = total_capacity * avg_fill

                    time_series_data.append(
                        {
                            "timestamp": timestamp,
                            "n_tanks": len(tanks),
                            "avg_fill_level": avg_fill,
                            "estimated_volume_m3": estimated_volume,
                            "total_capacity_m3": total_capacity,
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue

        # Create dataframe
        df = pd.DataFrame(time_series_data)

        if not df.empty:
            df = df.sort_values("timestamp")

            # Calculate changes
            df["volume_change_m3"] = df["estimated_volume_m3"].diff()
            df["volume_pct_change"] = df["estimated_volume_m3"].pct_change()

            # Rolling statistics
            df["volume_ma_7d"] = (
                df["estimated_volume_m3"].rolling(window=7, min_periods=1).mean()
            )

        analysis = {
            "facility_name": facility_name,
            "n_observations": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty else None,
                "end": df["timestamp"].max().isoformat() if not df.empty else None,
            },
            "inventory_metrics": {
                "avg_fill_level": float(df["avg_fill_level"].mean())
                if not df.empty
                else 0.0,
                "avg_volume_m3": float(df["estimated_volume_m3"].mean())
                if not df.empty
                else 0.0,
                "total_capacity_m3": float(df["total_capacity_m3"].mean())
                if not df.empty
                else 0.0,
            },
            "data": df,
        }

        logger.info(
            f"Facility analysis complete: {len(df)} observations, "
            f"avg fill {analysis['inventory_metrics']['avg_fill_level']:.1%}"
        )

        return analysis

    def generate_commodity_signal(
        self, facility_analysis: Dict[str, Any], commodity: str = "CL"
    ) -> Dict[str, Any]:
        """
        Generate trading signal for commodity based on inventory analysis.

        Args:
            facility_analysis: Facility analysis results
            commodity: Commodity ticker (e.g., 'CL' for crude oil)

        Returns:
            Trading signal
        """
        df = facility_analysis["data"]

        if df.empty:
            return {
                "commodity": commodity,
                "signal_strength": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
            }

        # Calculate inventory trend
        recent_volume = df["estimated_volume_m3"].tail(7).mean()
        historical_volume = df["estimated_volume_m3"].mean()

        # Inventory buildup = bearish, drawdown = bullish
        volume_change_pct = (
            (recent_volume - historical_volume) / historical_volume
            if historical_volume > 0
            else 0
        )

        # Invert signal (high inventory = bearish)
        signal_strength = -volume_change_pct * 2  # Amplify signal

        # Clip to [-1, 1]
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        signal = {
            "commodity": commodity,
            "signal_strength": float(signal_strength),
            "direction": "long" if signal_strength > 0 else "short",
            "confidence": float(abs(signal_strength)),
            "metrics": {
                "recent_volume_m3": float(recent_volume),
                "historical_volume_m3": float(historical_volume),
                "volume_change_pct": float(volume_change_pct),
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Generated {commodity} signal: "
            f"{signal['direction']} with strength {signal_strength:.3f}"
        )

        return signal
