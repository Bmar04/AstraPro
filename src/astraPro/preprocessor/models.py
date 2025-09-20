"""
Simple data models for target fusion.
"""

from dataclasses import dataclass
from typing import List
import time


@dataclass
class FusedTarget:
    """A detected target."""
    x: float                    # World X coordinate (meters)
    y: float                    # World Y coordinate (meters)
    confidence: float           # 0.0 to 1.0
    sensor_ids: List[int]       # Which sensors detected this
    timestamp: float            # When detected


@dataclass
class FusionResult:
    """Results from fusion."""
    targets: List[FusedTarget]  # All targets found
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class FusionConfig:
    """Fusion settings."""
    max_distance: float = 0.5           # Max distance to group measurements
    min_confidence: float = 0.3         # Min confidence to output
    allow_single_sensor: bool = True    # Allow single sensor targets


def calculate_confidence(measurements) -> float:
    """Simple confidence calculation."""
    if not measurements:
        return 0.0
    
    # Count unique sensors
    unique_sensors = len(set(m.sensor_id for m in measurements))
    
    # Base confidence
    if unique_sensors >= 2:
        confidence = 0.8  # Multi-sensor
    else:
        confidence = 0.5  # Single sensor
    
    # Distance boost (closer = better)
    avg_distance = sum(m.distance for m in measurements) / len(measurements)
    if avg_distance < 2.0:
        confidence += 0.1
    
    return min(0.9, confidence)


def create_target(measurements, config: FusionConfig) -> FusedTarget:
    """Create target from measurements."""
    # Weighted average position
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    
    for m in measurements:
        weight = 1.0 / (m.distance + 0.1)  # Closer = higher weight
        weighted_x += m.x_world * weight
        weighted_y += m.y_world * weight
        total_weight += weight
    
    x = weighted_x / total_weight
    y = weighted_y / total_weight
    confidence = calculate_confidence(measurements)
    sensor_ids = [m.sensor_id for m in measurements]
    timestamp = max(m.timestamp for m in measurements)
    
    return FusedTarget(x, y, confidence, sensor_ids, timestamp)