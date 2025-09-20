#!/usr/bin/env python3
"""
Debug rotating sensor behavior - why sensor 2 isn't detecting in +x -y region
"""

import sys
import os
import time
import queue
import math

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.io_module.simulator import create_default_simulation

def debug_sensor_rotation():
    print("=== SENSOR ROTATION DEBUG ===")
    
    # Create simulator
    sim = create_default_simulation("circle")
    raw_queue = queue.Queue()
    
    # Run for a short time to see sensor behavior
    sim.start(raw_queue)
    time.sleep(5)
    sim.stop()
    
    # Collect all measurements
    measurements = []
    while not raw_queue.empty():
        m = raw_queue.get_nowait()
        measurements.append(m)
    
    print(f"Total measurements collected: {len(measurements)}")
    
    # Group by sensor
    sensor_measurements = {}
    for m in measurements:
        if m.sensor_id not in sensor_measurements:
            sensor_measurements[m.sensor_id] = []
        sensor_measurements[m.sensor_id].append(m)
    
    # Analyze each sensor's behavior
    for sensor_id in [1, 2, 3, 4]:
        if sensor_id not in sensor_measurements:
            print(f"\nSensor {sensor_id}: NO MEASUREMENTS")
            continue
            
        meas = sensor_measurements[sensor_id]
        print(f"\nSensor {sensor_id}: {len(meas)} measurements")
        
        # Show range of angles detected
        angles = [m.angle for m in meas]
        distances = [m.distance for m in meas]
        
        print(f"  Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
        print(f"  Distance range: {min(distances):.2f}m to {max(distances):.2f}m")
        
        # Show some sample measurements
        print(f"  Sample measurements:")
        for i, m in enumerate(meas[:5]):
            print(f"    {i+1}: angle={m.angle:.1f}°, dist={m.distance:.2f}m, time={m.timestamp:.1f}")
    
    # Now analyze sensor rotation during circular movement
    print(f"\n=== SENSOR ROTATION ANALYSIS ===")
    
    # Get sensor configs to understand rotation
    sensors = sim.sensors
    for sensor in sensors:
        print(f"\nSensor {sensor.id} at ({sensor.x:.1f}, {sensor.y:.1f}):")
        print(f"  Current angle: {sensor.current_angle:.1f}°")
        print(f"  Servo speed: {sensor.servo_speed:.1f}°/cycle")
        print(f"  Field of view: {sensor.field_of_view:.1f}°")
        
        # Calculate what angles this sensor should see for +x -y region
        # For a point at (0.5, -0.5), what angle from this sensor?
        test_x, test_y = 0.5, -0.5
        dx = test_x - sensor.x
        dy = test_y - sensor.y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        distance_to_target = math.sqrt(dx*dx + dy*dy)
        
        print(f"  For target at (0.5, -0.5):")
        print(f"    Angle from sensor: {angle_to_target:.1f}°")
        print(f"    Distance: {distance_to_target:.2f}m")
        print(f"    Within range? {distance_to_target <= sensor.max_range}")
        
        # Check if sensor sweeps through this angle
        angle_diff = abs(angle_to_target - sensor.current_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        print(f"    Current angle diff: {angle_diff:.1f}°")
        print(f"    Within FOV? {angle_diff <= sensor.field_of_view/2}")
        
        # Check sensor's sweep range
        print(f"    Sensor sweeps: {sensor.current_angle - 90:.1f}° to {sensor.current_angle + 90:.1f}°")
        will_detect = (angle_to_target >= sensor.current_angle - 90 and 
                      angle_to_target <= sensor.current_angle + 90)
        print(f"    Will eventually detect? {will_detect}")

if __name__ == "__main__":
    debug_sensor_rotation()