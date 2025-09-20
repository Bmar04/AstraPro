"""
Coordinate transformations for sonar fusion system.

Transforms sensor measurements from polar coordinates to world Cartesian coordinates.
All transformations are automatically calculated from sensor positions in config.
"""

import numpy as np
import math
from typing import Tuple, Dict, Any


def calculate_home_direction(sensor_pos: Dict[str, float], field_center: Dict[str, float]) -> float:
    """
    Calculate the home direction angle for a sensor pointing toward field center.
    
    Args:
        sensor_pos: Sensor position {'x': float, 'y': float, 'z': float}
        field_center: Field center position {'x': float, 'y': float}
    
    Returns:
        Home direction angle in degrees (0-360 degrees)
    """
    dx = field_center['x'] - sensor_pos['x']
    dy = field_center['y'] - sensor_pos['y']
    
    # Calculate angle from sensor to field center
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-360 degree range
    if angle_deg < 0:
        angle_deg += 360.0
        
    return angle_deg


def polar_to_cartesian(range_m: float, pan_angle_deg: float) -> Tuple[float, float]:
    """
    Convert polar coordinates to Cartesian in sensor local frame.
    
    Args:
        range_m: Distance to target in meters
        pan_angle_deg: Pan angle in degrees (0 degrees = home direction)
    
    Returns:
        (x_local, y_local) in sensor's coordinate system
    """
    pan_rad = math.radians(pan_angle_deg)
    x_local = range_m * math.cos(pan_rad)
    y_local = range_m * math.sin(pan_rad)
    
    return x_local, y_local


def sensor_local_to_world(x_local: float, y_local: float, 
                         sensor_config: Dict[str, Any], 
                         field_config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Transform from sensor local coordinates to world coordinates.
    
    Args:
        x_local: X coordinate in sensor local frame
        y_local: Y coordinate in sensor local frame  
        sensor_config: Sensor configuration from config.yaml
        field_config: Field configuration from config.yaml
    
    Returns:
        (x_world, y_world) in world coordinate system
    """
    # Get sensor position
    sensor_pos = sensor_config['position']
    px = sensor_pos['x']
    py = sensor_pos['y']
    
    # Calculate home direction automatically
    home_direction_deg = calculate_home_direction(sensor_pos, field_config['center'])
    home_direction_rad = math.radians(home_direction_deg)
    
    # Rotation matrix from local to world coordinates
    cos_alpha = math.cos(home_direction_rad)
    sin_alpha = math.sin(home_direction_rad)
    
    # Apply rotation and translation
    x_world = cos_alpha * x_local - sin_alpha * y_local + px
    y_world = sin_alpha * x_local + cos_alpha * y_local + py
    
    return x_world, y_world


def measurement_to_world(range_m: float, angle_deg: float,
                        sensor_config: Dict[str, Any],
                        field_config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Complete transformation: polar measurement to world coordinates.
    
    Args:
        range_m: Distance measurement in meters
        angle_deg: Angle from sensor to object in world coordinates (degrees)
        sensor_config: Sensor configuration from config.yaml
        field_config: Field configuration from config.yaml
    
    Returns:
        (x_world, y_world) in world coordinate system
    """
    # Direct conversion from polar to world coordinates
    # The angle is already in world coordinates from the sensor
    angle_rad = math.radians(angle_deg)
    
    # Get sensor position
    sensor_pos = sensor_config['position']
    sensor_x = sensor_pos['x']
    sensor_y = sensor_pos['y']
    
    # Calculate world position directly
    x_world = sensor_x + range_m * math.cos(angle_rad)
    y_world = sensor_y + range_m * math.sin(angle_rad)
    
    return x_world, y_world


def get_sensor_home_direction(sensor_config: Dict[str, Any], 
                             field_config: Dict[str, Any]) -> float:
    """
    Get the calculated home direction for a sensor.
    Useful for debugging and validation.
    
    Args:
        sensor_config: Sensor configuration from config.yaml
        field_config: Field configuration from config.yaml
    
    Returns:
        Home direction angle in degrees
    """
    return calculate_home_direction(sensor_config['position'], field_config['center'])


def validate_sensor_coverage(sensor_config: Dict[str, Any],
                           field_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze sensor coverage area for validation.
    
    Args:
        sensor_config: Sensor configuration from config.yaml
        field_config: Field configuration from config.yaml
    
    Returns:
        Dictionary with coverage analysis
    """
    home_dir = get_sensor_home_direction(sensor_config, field_config)
    pan_range = sensor_config['servo']['pan_range']
    
    # Calculate actual coverage angles in world coordinates
    left_angle = home_dir + pan_range[0]  # pan_range[0] is typically negative
    right_angle = home_dir + pan_range[1]  # pan_range[1] is typically positive
    
    return {
        'home_direction': home_dir,
        'left_coverage': left_angle % 360,
        'right_coverage': right_angle % 360,
        'coverage_span': pan_range[1] - pan_range[0]
    }


# Inverse transformations for validation and testing

def world_to_sensor_local(x_world: float, y_world: float,
                         sensor_config: Dict[str, Any],
                         field_config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Inverse transformation: world coordinates to sensor local coordinates.
    Used for testing and validation.
    """
    # Get sensor position
    sensor_pos = sensor_config['position']
    px = sensor_pos['x'] 
    py = sensor_pos['y']
    
    # Translate to sensor origin
    x_translated = x_world - px
    y_translated = y_world - py
    
    # Get home direction and create inverse rotation matrix
    home_direction_deg = calculate_home_direction(sensor_pos, field_config['center'])
    home_direction_rad = math.radians(home_direction_deg)
    
    cos_alpha = math.cos(-home_direction_rad)  # Negative for inverse rotation
    sin_alpha = math.sin(-home_direction_rad)
    
    # Apply inverse rotation
    x_local = cos_alpha * x_translated - sin_alpha * y_translated
    y_local = sin_alpha * x_translated + cos_alpha * y_translated
    
    return x_local, y_local


def cartesian_to_polar(x_local: float, y_local: float) -> Tuple[float, float]:
    """
    Convert local Cartesian coordinates back to polar.
    Used for testing and validation.
    """
    range_m = math.sqrt(x_local**2 + y_local**2)
    pan_angle_rad = math.atan2(y_local, x_local)
    pan_angle_deg = math.degrees(pan_angle_rad)
    
    return range_m, pan_angle_deg


def world_to_measurement(x_world: float, y_world: float,
                        sensor_config: Dict[str, Any],
                        field_config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Complete inverse transformation: world coordinates to polar measurement.
    Used for simulation and testing.
    """
    # Step 1: World to local coordinates
    x_local, y_local = world_to_sensor_local(x_world, y_world, sensor_config, field_config)
    
    # Step 2: Local Cartesian to polar
    range_m, pan_angle_deg = cartesian_to_polar(x_local, y_local)
    
    return range_m, pan_angle_deg