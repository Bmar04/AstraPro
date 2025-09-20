#!/usr/bin/env python3
"""
Test coordinate system conversion to verify fix
"""

import sys
import os
import yaml
import math

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.preprocessor.transforms import measurement_to_world

def test_coordinate_conversion():
    print("=== COORDINATE CONVERSION TEST ===")
    
    # Load config
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sensors = {sensor['id']: sensor for sensor in config['sensors']}
    field_config = config['field']
    
    # Test case: object at center of field (0, 0)
    test_objects = [
        (0.0, 0.0, "Center of field"),
        (0.8, 0.0, "CircularMovement position"),
        (-1.0, -1.0, "CornerToCornerWalk start"),
        (0.5, -0.5, "RandomWalk area")
    ]
    
    print(f"Field center: ({field_config['center']['x']}, {field_config['center']['y']})")
    print(f"Field size: {field_config['width']}x{field_config['height']}m")
    print()
    
    for sensor_id in [1, 2, 3, 4]:
        sensor_config = sensors[sensor_id]
        print(f"Sensor {sensor_id} at ({sensor_config['position']['x']}, {sensor_config['position']['y']}):")
        
        for obj_x, obj_y, description in test_objects:
            # Calculate what sensor would measure for this object
            dx = obj_x - sensor_config['position']['x']
            dy = obj_y - sensor_config['position']['y']
            
            distance = math.sqrt(dx*dx + dy*dy)
            angle_to_obj = math.degrees(math.atan2(dy, dx))
            
            # Check if this would be detectable (within range and reasonable angle)
            if distance <= sensor_config['max_range']:
                # Convert back using our preprocessor
                converted_x, converted_y = measurement_to_world(
                    distance, angle_to_obj, sensor_config, field_config
                )
                
                error_x = abs(converted_x - obj_x)
                error_y = abs(converted_y - obj_y)
                
                print(f"  {description}: ({obj_x:.2f}, {obj_y:.2f})")
                print(f"    Distance: {distance:.2f}m, Angle: {angle_to_obj:.1f}°")
                print(f"    Converted: ({converted_x:.2f}, {converted_y:.2f})")
                print(f"    Error: ({error_x:.3f}, {error_y:.3f})")
                
                if error_x < 0.01 and error_y < 0.01:
                    print(f"    ✓ GOOD")
                else:
                    print(f"    ✗ ERROR")
                print()
        print()

if __name__ == "__main__":
    test_coordinate_conversion()