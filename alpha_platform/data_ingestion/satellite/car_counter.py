"""
Satellite imagery car counting for retail economic activity analysis.

Uses YOLOv8 to count cars in retail parking lots (Walmart, Target, etc.)
to predict quarterly revenue before earnings announcements.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    warnings.warn("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

from alpha_platform.utils.config import get_config, get_data_path
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class SatelliteCarCounter:
    """
    Count cars in satellite imagery of retail parking lots.

    This class processes satellite images to detect and count vehicles,
    providing a leading indicator of retail traffic and potential revenue.
    """

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize the satellite car counter.

        Args:
            model_name: YOLOv8 model variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.config = get_config()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing car counter on device: {self.device}")

        # Load YOLO model
        if YOLO is None:
            raise ImportError("ultralytics not installed")

        self.model = YOLO(model_name)
        self.model.to(self.device)

        # Vehicle class IDs in COCO dataset (used by YOLO)
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        logger.info(f"Loaded {model_name} for car detection")

    def count_cars_in_image(
        self,
        image_path: str,
        visualize: bool = False,
        save_visualization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Count cars in a single satellite image.

        Args:
            image_path: Path to satellite image
            visualize: Whether to create visualization
            save_visualization: Path to save visualization (if visualize=True)

        Returns:
            Dictionary with count results and metadata
        """
        logger.info(f"Processing image: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_height, original_width = image.shape[:2]

        # Run detection
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        # Filter for vehicle classes
        vehicle_detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in self.vehicle_classes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                vehicle_detections.append(
                    {
                        "class": self.vehicle_classes[class_id],
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )

        # Calculate metrics
        total_count = len(vehicle_detections)
        cars_only = len([d for d in vehicle_detections if d["class"] == "car"])
        avg_confidence = (
            np.mean([d["confidence"] for d in vehicle_detections])
            if vehicle_detections
            else 0.0
        )

        # Estimate parking lot occupancy (assuming we can see the full lot)
        parking_area_pixels = original_height * original_width
        avg_car_size = (
            np.mean([d["area"] for d in vehicle_detections])
            if vehicle_detections
            else 100
        )
        estimated_capacity = int(parking_area_pixels / (avg_car_size * 2))
        occupancy_rate = min(total_count / max(estimated_capacity, 1), 1.0)

        result = {
            "total_vehicles": total_count,
            "cars": cars_only,
            "trucks_buses": total_count - cars_only,
            "avg_confidence": float(avg_confidence),
            "occupancy_rate": float(occupancy_rate),
            "estimated_capacity": estimated_capacity,
            "image_width": original_width,
            "image_height": original_height,
            "detections": vehicle_detections,
            "timestamp": datetime.now().isoformat(),
        }

        # Create visualization if requested
        if visualize:
            vis_image = self._visualize_detections(image, vehicle_detections)

            if save_visualization:
                cv2.imwrite(save_visualization, vis_image)
                result["visualization_path"] = save_visualization

        logger.info(
            f"Detected {total_count} vehicles (occupancy: {occupancy_rate:.1%})"
        )

        return result

    def _visualize_detections(
        self, image: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Create visualization of detections on image.

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Color map for vehicle types
        color_map = {
            "car": (0, 255, 0),  # Green
            "truck": (255, 0, 0),  # Blue
            "bus": (0, 0, 255),  # Red
            "motorcycle": (255, 255, 0),  # Cyan
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = color_map.get(det["class"], (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1,
            )
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return vis_image

    def process_time_series(
        self,
        image_paths: List[str],
        timestamps: List[datetime],
        location_metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Process a time series of satellite images for a location.

        Args:
            image_paths: List of paths to satellite images
            timestamps: Corresponding timestamps for each image
            location_metadata: Optional metadata about the location

        Returns:
            DataFrame with time series of car counts and metrics
        """
        logger.info(f"Processing time series of {len(image_paths)} images")

        results = []
        for img_path, timestamp in zip(image_paths, timestamps):
            try:
                count_result = self.count_cars_in_image(img_path)
                count_result["timestamp"] = timestamp

                if location_metadata:
                    count_result.update(location_metadata)

                results.append(count_result)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        df = pd.DataFrame(results)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Calculate time-based features
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["hour"] = df["timestamp"].dt.hour
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Calculate rolling statistics
            df["cars_ma_7d"] = df["cars"].rolling(window=7, min_periods=1).mean()
            df["cars_std_7d"] = df["cars"].rolling(window=7, min_periods=1).std()
            df["occupancy_ma_7d"] = (
                df["occupancy_rate"].rolling(window=7, min_periods=1).mean()
            )

            # Calculate change metrics
            df["cars_pct_change"] = df["cars"].pct_change()
            df["cars_diff"] = df["cars"].diff()

        return df

    def analyze_store_traffic(
        self,
        store_name: str,
        ticker: str,
        images_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Analyze store traffic patterns from satellite imagery.

        Args:
            store_name: Name of the store (e.g., "Walmart #1234")
            ticker: Stock ticker symbol
            images_dir: Directory containing satellite images
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis

        Returns:
            Analysis results including trends and predictions
        """
        logger.info(f"Analyzing traffic for {store_name} ({ticker})")

        # Load images from directory
        images_path = Path(images_dir)
        image_files = sorted(images_path.glob("*.jpg")) + sorted(
            images_path.glob("*.png")
        )

        if not image_files:
            raise ValueError(f"No images found in {images_dir}")

        # Extract timestamps from filenames (assuming format: YYYYMMDD_*.jpg)
        timestamps = []
        valid_images = []
        for img_file in image_files:
            try:
                date_str = img_file.stem.split("_")[0]
                timestamp = datetime.strptime(date_str, "%Y%m%d")

                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue

                timestamps.append(timestamp)
                valid_images.append(str(img_file))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse timestamp from {img_file.name}")
                continue

        # Process time series
        metadata = {"store_name": store_name, "ticker": ticker}
        df = self.process_time_series(valid_images, timestamps, metadata)

        # Calculate analysis metrics
        analysis = {
            "store_name": store_name,
            "ticker": ticker,
            "n_observations": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            },
            "traffic_metrics": {
                "avg_cars_per_day": float(df["cars"].mean()),
                "median_cars_per_day": float(df["cars"].median()),
                "std_cars_per_day": float(df["cars"].std()),
                "avg_occupancy_rate": float(df["occupancy_rate"].mean()),
            },
            "trends": {
                "overall_trend": self._calculate_trend(df["cars"]),
                "recent_trend_30d": self._calculate_trend(df["cars"].tail(30)),
                "weekend_vs_weekday": float(
                    df[df["is_weekend"] == 1]["cars"].mean()
                    / df[df["is_weekend"] == 0]["cars"].mean()
                )
                if len(df[df["is_weekend"] == 0]) > 0
                else 1.0,
            },
            "data": df,
        }

        logger.info(
            f"Analysis complete: {len(df)} observations, "
            f"avg {analysis['traffic_metrics']['avg_cars_per_day']:.1f} cars/day"
        )

        return analysis

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction from a series."""
        if len(series) < 2:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.1:
            return "flat"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def generate_alpha_signal(
        self,
        analysis: Dict[str, Any],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate alpha signal from traffic analysis.

        Args:
            analysis: Store traffic analysis results
            benchmark_data: Optional benchmark data for comparison

        Returns:
            Alpha signal with strength and direction
        """
        df = analysis["data"]

        # Calculate z-score of recent traffic vs historical
        recent_avg = df["cars"].tail(7).mean()
        historical_mean = df["cars"].mean()
        historical_std = df["cars"].std()

        z_score = (
            (recent_avg - historical_mean) / historical_std
            if historical_std > 0
            else 0
        )

        # Calculate momentum
        ma_short = df["cars"].tail(7).mean()
        ma_long = df["cars"].tail(30).mean() if len(df) >= 30 else historical_mean
        momentum = (ma_short - ma_long) / ma_long if ma_long > 0 else 0

        # Combine signals
        signal_strength = (z_score + momentum) / 2

        # Normalize to [-1, 1]
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        signal = {
            "ticker": analysis["ticker"],
            "signal_strength": float(signal_strength),
            "direction": "long" if signal_strength > 0 else "short",
            "confidence": float(abs(signal_strength)),
            "components": {
                "z_score": float(z_score),
                "momentum": float(momentum),
            },
            "metrics": {
                "recent_avg_cars": float(recent_avg),
                "historical_avg_cars": float(historical_mean),
                "pct_change": float((recent_avg - historical_mean) / historical_mean)
                if historical_mean > 0
                else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Generated signal for {analysis['ticker']}: "
            f"{signal['direction']} with strength {signal_strength:.3f}"
        )

        return signal
