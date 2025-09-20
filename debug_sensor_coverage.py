#!/usr/bin/env python3
"""
Debug sensor coverage analysis - why +x -y region has poor coverage
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def analyze_sensor_coverage():
    print("=== SENSOR COVERAGE ANALYSIS ===")
    
    # Sensor positions (corners of 3x3m field)
    sensors = {
        1: (-1.5, -1.5),  # Southwest
        2: (1.5, -1.5),   # Southeast  
        3: (1.5, 1.5),    # Northeast
        4: (-1.5, 1.5)    # Northwest
    }
    
    # Circular path points (radius 0.8m around center)
    circle_points = []
    for angle in range(0, 360, 10):
        rad = math.radians(angle)
        x = 0.8 * math.cos(rad)
        y = 0.8 * math.sin(rad)
        circle_points.append((x, y, angle))
    
    print(f"Sensor positions:")
    for sid, (sx, sy) in sensors.items():
        print(f"  Sensor {sid}: ({sx:.1f}, {sy:.1f})")
    
    # Analyze coverage for each region
    regions = {
        "+x -y": [],  # Problem region
        "+x +y": [],
        "-x +y": [],
        "-x -y": []
    }
    
    for x, y, angle in circle_points:
        # Determine region
        if x >= 0 and y >= 0:
            region = "+x +y"
        elif x < 0 and y >= 0:
            region = "-x +y"
        elif x >= 0 and y < 0:
            region = "+x -y"  # Problem region
        else:
            region = "-x -y"
        
        # Check which sensors can see this point
        visible_sensors = []
        for sid, (sx, sy) in sensors.items():
            distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            if distance <= 2.5:  # Within sensor range
                # Check if angle is within sensor's rotating range
                # For simplicity, assume sensors can rotate to see this point
                visible_sensors.append(sid)
        
        regions[region].append({
            'position': (x, y),
            'angle': angle,
            'visible_sensors': visible_sensors,
            'sensor_count': len(visible_sensors),
            'closest_sensor_distance': min([math.sqrt((x - sx)**2 + (y - sy)**2) for sx, sy in sensors.values()])
        })
    
    # Print analysis
    print(f"\n=== REGION COVERAGE ANALYSIS ===")
    for region_name, points in regions.items():
        if not points:
            continue
        
        avg_sensors = sum(p['sensor_count'] for p in points) / len(points)
        min_sensors = min(p['sensor_count'] for p in points)
        max_sensors = max(p['sensor_count'] for p in points)
        avg_distance = sum(p['closest_sensor_distance'] for p in points) / len(points)
        
        print(f"\n{region_name} region ({len(points)} points):")
        print(f"  Sensor coverage: avg={avg_sensors:.1f}, min={min_sensors}, max={max_sensors}")
        print(f"  Avg distance to closest sensor: {avg_distance:.2f}m")
        
        # Find problem points
        poor_coverage = [p for p in points if p['sensor_count'] <= 1]
        if poor_coverage:
            print(f"  Poor coverage points ({len(poor_coverage)}):")
            for p in poor_coverage[:3]:  # Show first 3
                print(f"    ({p['position'][0]:.2f}, {p['position'][1]:.2f}) angle={p['angle']}° sensors={p['visible_sensors']}")
        
        if region_name == "+x -y":
            print(f"  *** ANALYZING PROBLEM REGION ***")
            
            # Check specific positions that should be visible
            problem_points = [p for p in points if 0.3 <= p['position'][0] <= 0.8 and -0.8 <= p['position'][1] <= -0.3]
            print(f"  Critical area (0.3-0.8, -0.8--0.3): {len(problem_points)} points")
            
            for p in problem_points:
                x, y = p['position']
                print(f"    ({x:.2f}, {y:.2f}): sensors={p['visible_sensors']}")
                
                # Check distance to each sensor
                for sid, (sx, sy) in sensors.items():
                    dist = math.sqrt((x - sx)**2 + (y - sy)**2)
                    angle_to_point = math.degrees(math.atan2(y - sy, x - sx))
                    print(f"      Sensor {sid} at ({sx:.1f}, {sy:.1f}): dist={dist:.2f}m, angle={angle_to_point:.1f}°")
    
    # Visualize coverage
    plt.figure(figsize=(10, 8))
    
    # Plot sensors
    for sid, (sx, sy) in sensors.items():
        plt.plot(sx, sy, 's', markersize=12, color='black', markerfacecolor='yellow')
        plt.annotate(f'S{sid}', (sx, sy), xytext=(5, 5), textcoords='offset points')
        
        # Plot sensor range
        circle = plt.Circle((sx, sy), 2.5, fill=False, linestyle='--', alpha=0.3, color='gray')
        plt.gca().add_patch(circle)
    
    # Plot circular path with color coding by sensor coverage
    for region_name, points in regions.items():
        colors = {'+x -y': 'red', '+x +y': 'blue', '-x +y': 'green', '-x -y': 'orange'}
        for p in points:
            x, y = p['position']
            # Size based on sensor count
            size = 20 + p['sensor_count'] * 15
            alpha = 0.3 + p['sensor_count'] * 0.2
            plt.scatter(x, y, s=size, c=colors[region_name], alpha=alpha)
    
    # Draw field boundary
    field_rect = plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, edgecolor='black', linewidth=2)
    plt.gca().add_patch(field_rect)
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(True, alpha=0.3)
    plt.title('Sensor Coverage Analysis\n(Larger/darker dots = better coverage)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    
    # Add legend
    for region_name, color in [('-x +y', 'green'), ('+x +y', 'blue'), ('+x -y', 'red'), ('-x -y', 'orange')]:
        plt.scatter([], [], c=color, label=region_name, s=50)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    analyze_sensor_coverage()