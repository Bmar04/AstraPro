"""
Test cases for coordinate transformations.
Validates automatic home direction calculation and coordinate transforms.
"""

import pytest
import math
import yaml
from preprocessor.transforms import (
    calculate_home_direction,
    polar_to_cartesian,
    sensor_local_to_world,
    measurement_to_world,
    world_to_measurement,
    get_sensor_home_direction,
    validate_sensor_coverage
)

# Test configuration data
TEST_FIELD_CONFIG = {
    'center': {'x': 0.0, 'y': 0.0},
    'width': 6.0,
    'height': 6.0
}

TEST_SENSORS = {
    'sensor_1': {
        'id': 1,
        'position': {'x': -3.0, 'y': -3.0, 'z': 0.5},
        'servo': {'pan_range': [-90, 90], 'pan_speed': 180.0},
        'max_range': 8.0
    },
    'sensor_2': {
        'id': 2,
        'position': {'x': 3.0, 'y': -3.0, 'z': 0.5},
        'servo': {'pan_range': [-90, 90], 'pan_speed': 180.0},
        'max_range': 8.0
    },
    'sensor_3': {
        'id': 3,
        'position': {'x': 3.0, 'y': 3.0, 'z': 0.5},
        'servo': {'pan_range': [-90, 90], 'pan_speed': 180.0},
        'max_range': 8.0
    },
    'sensor_4': {
        'id': 4,
        'position': {'x': -3.0, 'y': 3.0, 'z': 0.5},
        'servo': {'pan_range': [-90, 90], 'pan_speed': 180.0},
        'max_range': 8.0
    }
}


class TestHomeDirectionCalculation:
    """Test automatic home direction calculation."""
    
    def test_calculate_home_direction_corner_sensors(self):
        """Test home direction calculation for corner-mounted sensors."""
        field_center = {'x': 0.0, 'y': 0.0}
        
        # Test each corner sensor
        test_cases = [
            ({'x': -3.0, 'y': -3.0}, 45.0),   # Southwest -> Northeast (45°)
            ({'x': 3.0, 'y': -3.0}, 135.0),   # Southeast -> Northwest (135°)
            ({'x': 3.0, 'y': 3.0}, 225.0),    # Northeast -> Southwest (225°)
            ({'x': -3.0, 'y': 3.0}, 315.0)    # Northwest -> Southeast (315°)
        ]
        
        for sensor_pos, expected_angle in test_cases:
            actual_angle = calculate_home_direction(sensor_pos, field_center)
            assert abs(actual_angle - expected_angle) < 0.001, f"Expected {expected_angle}°, got {actual_angle}°"
    
    def test_calculate_home_direction_edge_cases(self):
        """Test home direction for edge positions."""
        field_center = {'x': 0.0, 'y': 0.0}
        
        test_cases = [
            ({'x': -5.0, 'y': 0.0}, 0.0),     # Due west -> Due east (0°)
            ({'x': 0.0, 'y': -5.0}, 90.0),    # Due south -> Due north (90°)
            ({'x': 5.0, 'y': 0.0}, 180.0),    # Due east -> Due west (180°)
            ({'x': 0.0, 'y': 5.0}, 270.0)     # Due north -> Due south (270°)
        ]
        
        for sensor_pos, expected_angle in test_cases:
            actual_angle = calculate_home_direction(sensor_pos, field_center)
            assert abs(actual_angle - expected_angle) < 0.001, f"Expected {expected_angle}°, got {actual_angle}°"


class TestCoordinateTransforms:
    """Test coordinate transformation functions."""
    
    def test_polar_to_cartesian_basic(self):
        """Test basic polar to Cartesian conversion."""
        test_cases = [
            (1.0, 0.0, 1.0, 0.0),      # 0° -> (1, 0)
            (1.0, 90.0, 0.0, 1.0),     # 90° -> (0, 1)
            (1.0, 180.0, -1.0, 0.0),   # 180° -> (-1, 0)
            (1.0, 270.0, 0.0, -1.0),   # 270° -> (0, -1)
            (math.sqrt(2), 45.0, 1.0, 1.0)  # 45° -> (1, 1)
        ]
        
        for range_m, angle_deg, expected_x, expected_y in test_cases:
            x, y = polar_to_cartesian(range_m, angle_deg)
            assert abs(x - expected_x) < 0.001, f"X: Expected {expected_x}, got {x}"
            assert abs(y - expected_y) < 0.001, f"Y: Expected {expected_y}, got {y}"
    
    def test_sensor_local_to_world_sensor_1(self):
        """Test local to world transform for sensor 1 (SW corner)."""
        sensor_config = TEST_SENSORS['sensor_1']
        
        # Target at (1, 0) in local coords should be at field center when pan=0°
        x_world, y_world = sensor_local_to_world(1.0, 0.0, sensor_config, TEST_FIELD_CONFIG)
        
        # Sensor 1 at (-3, -3) with 45° home direction
        # Local (1, 0) rotated 45° = (0.707, 0.707), then translated to (-3, -3)
        # Should give approximately (-2.29, -2.29)
        expected_x = -3.0 + 1.0 * math.cos(math.radians(45))
        expected_y = -3.0 + 1.0 * math.sin(math.radians(45))
        
        assert abs(x_world - expected_x) < 0.001
        assert abs(y_world - expected_y) < 0.001


class TestEndToEndTransformation:
    """Test complete measurement to world coordinate transformation."""
    
    def test_measurement_to_world_field_center(self):
        """Test that all sensors pointing at field center give same result."""
        # Target at field center should be visible to all sensors at pan=0°
        target_x, target_y = 0.0, 0.0
        
        results = []
        for sensor_name, sensor_config in TEST_SENSORS.items():
            # Calculate expected range to field center
            sensor_pos = sensor_config['position']
            expected_range = math.sqrt(sensor_pos['x']**2 + sensor_pos['y']**2)
            
            # Measure at pan=0° (pointing at field center)
            x_world, y_world = measurement_to_world(
                expected_range, 0.0, sensor_config, TEST_FIELD_CONFIG
            )
            results.append((x_world, y_world))
        
        # All sensors should report approximately the same world coordinates
        for i, (x, y) in enumerate(results[1:], 1):
            assert abs(x - results[0][0]) < 0.01, f"Sensor {i+1} X mismatch: {x} vs {results[0][0]}"
            assert abs(y - results[0][1]) < 0.01, f"Sensor {i+1} Y mismatch: {y} vs {results[0][1]}"
    
    def test_round_trip_transformation(self):
        """Test that forward and inverse transforms are consistent."""
        test_points = [
            (0.0, 0.0),    # Field center
            (1.0, 1.0),    # Northeast quadrant
            (-1.0, 1.0),   # Northwest quadrant
            (2.0, -2.0),   # Southeast quadrant
        ]
        
        for sensor_config in TEST_SENSORS.values():
            for target_x, target_y in test_points:
                # Forward: world -> measurement
                range_m, pan_deg = world_to_measurement(target_x, target_y, sensor_config, TEST_FIELD_CONFIG)
                
                # Skip if target is outside sensor range
                if range_m > sensor_config['max_range'] or abs(pan_deg) > 90:
                    continue
                
                # Inverse: measurement -> world
                recovered_x, recovered_y = measurement_to_world(range_m, pan_deg, sensor_config, TEST_FIELD_CONFIG)
                
                # Should recover original coordinates
                assert abs(recovered_x - target_x) < 0.001, f"X recovery failed: {recovered_x} vs {target_x}"
                assert abs(recovered_y - target_y) < 0.001, f"Y recovery failed: {recovered_y} vs {target_y}"


class TestSensorCoverage:
    """Test sensor coverage analysis."""
    
    def test_sensor_coverage_calculation(self):
        """Test sensor coverage area calculation."""
        for sensor_name, sensor_config in TEST_SENSORS.items():
            coverage = validate_sensor_coverage(sensor_config, TEST_FIELD_CONFIG)
            
            # Check that coverage span matches configured pan range
            expected_span = sensor_config['servo']['pan_range'][1] - sensor_config['servo']['pan_range'][0]
            assert abs(coverage['coverage_span'] - expected_span) < 0.001
            
            # Check that coverage angles are in valid range
            assert 0 <= coverage['left_coverage'] <= 360
            assert 0 <= coverage['right_coverage'] <= 360
            assert 0 <= coverage['home_direction'] <= 360


class TestConfigIntegration:
    """Test integration with actual config file."""
    
    def test_with_actual_config(self):
        """Test transforms using actual config.yaml file."""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            field_config = config['field']
            sensors = config['sensors']
            
            # Test that each sensor can calculate its home direction
            for sensor in sensors:
                home_dir = get_sensor_home_direction(sensor, field_config)
                assert 0 <= home_dir <= 360, f"Invalid home direction: {home_dir}"
                
                # Test a basic transformation
                x_world, y_world = measurement_to_world(3.0, 0.0, sensor, field_config)
                assert isinstance(x_world, float)
                assert isinstance(y_world, float)
                
        except FileNotFoundError:
            pytest.skip("config.yaml not found - skipping integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])