#!/usr/bin/env python3
"""
Debug circular movement tracking - focus on +x -y region issues
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
from astraPro.preprocessor.processor import start_preprocessor
from astraPro.preprocessor.triangulation import TriangulationEngine
from astraPro.aitracker.tracker import TrackerSystem

def debug_circular_tracking():
    print("=== CIRCULAR MOVEMENT DEBUG - +X -Y REGION FOCUS ===")
    
    # Create simulator with only circular movement
    sim = create_default_simulation("circle")  # Only circular movement
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    sim.start(raw_queue)
    
    # Start preprocessor
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=False)
    
    # Create tracking components
    fusion_engine = TriangulationEngine(max_distance=0.6, min_confidence=0.1)
    tracking_system = TrackerSystem(gate_threshold=2.5)
    
    print("Running circular tracking debug for 20 seconds...")
    start_time = time.time()
    
    # Track the circular object through different regions
    region_data = {
        "-x +y": [],  # Northwest
        "+x +y": [],  # Northeast  
        "+x -y": [],  # Southeast (problem region)
        "-x -y": []   # Southwest
    }
    
    while time.time() - start_time < 20:
        current_time = time.time() - start_time
        
        # Get ground truth
        ground_truth = sim.get_ground_truth_objects()
        if not ground_truth:
            time.sleep(0.1)
            continue
            
        obj = ground_truth[0]  # Only one circular object
        
        # Determine which region object is in
        region = ""
        if obj.x >= 0 and obj.y >= 0:
            region = "+x +y"
        elif obj.x < 0 and obj.y >= 0:
            region = "-x +y"
        elif obj.x >= 0 and obj.y < 0:
            region = "+x -y"  # Problem region
        else:
            region = "-x -y"
        
        # Process measurements
        measurements_this_cycle = []
        while not processed_queue.empty():
            try:
                processed_measurement = processed_queue.get_nowait()
                fusion_engine.add_measurement(processed_measurement)
                measurements_this_cycle.append(processed_measurement)
            except queue.Empty:
                break
        
        # Run fusion and tracking
        tracks_this_cycle = []
        if fusion_engine.should_fuse():
            result = fusion_engine.fuse_targets()
            
            if result.targets:
                tracking_result = tracking_system.update(result.targets, result.timestamp)
                tracks_this_cycle = tracking_result.get_active_tracks()
        
        # Store data for this region
        region_data[region].append({
            'time': current_time,
            'ground_truth': (obj.x, obj.y),
            'measurements': len(measurements_this_cycle),
            'targets': len(result.targets) if 'result' in locals() and result.targets else 0,
            'tracks': len(tracks_this_cycle),
            'measurement_positions': [(m.x_world, m.y_world) for m in measurements_this_cycle],
            'track_positions': [t.get_position() for t in tracks_this_cycle] if tracks_this_cycle else []
        })
        
        # Print detailed info for problem region
        if region == "+x -y" and (measurements_this_cycle or tracks_this_cycle):
            print(f"\n+X -Y REGION at t={current_time:.1f}s:")
            print(f"  Ground Truth: ({obj.x:.2f}, {obj.y:.2f})")
            print(f"  Raw measurements: {len(measurements_this_cycle)}")
            for i, m in enumerate(measurements_this_cycle):
                print(f"    Measurement {i+1}: Sensor {m.sensor_id} -> ({m.x_world:.2f}, {m.y_world:.2f})")
            
            if 'result' in locals() and result.targets:
                print(f"  Fused targets: {len(result.targets)}")
                for i, target in enumerate(result.targets):
                    sensors_str = ", ".join(map(str, target.sensor_ids))
                    print(f"    Target {i+1}: ({target.x:.2f}, {target.y:.2f}) conf={target.confidence:.2f} sensors=[{sensors_str}]")
            
            print(f"  Active tracks: {len(tracks_this_cycle)}")
            for track in tracks_this_cycle:
                pos = track.get_position()
                print(f"    Track {track.id}: ({pos[0]:.2f}, {pos[1]:.2f}) status={track.status}")
        
        time.sleep(0.1)
    
    sim.stop()
    
    # Analyze results by region
    print(f"\n=== REGION ANALYSIS ===")
    for region_name, data in region_data.items():
        if not data:
            continue
            
        total_measurements = sum(d['measurements'] for d in data)
        total_targets = sum(d['targets'] for d in data)
        total_tracks = sum(d['tracks'] for d in data)
        avg_measurements = total_measurements / len(data) if data else 0
        avg_targets = total_targets / len(data) if data else 0
        avg_tracks = total_tracks / len(data) if data else 0
        
        print(f"\n{region_name} region ({len(data)} samples):")
        print(f"  Avg measurements per cycle: {avg_measurements:.1f}")
        print(f"  Avg targets per cycle: {avg_targets:.1f}")
        print(f"  Avg tracks per cycle: {avg_tracks:.1f}")
        
        # Check sensor coverage in this region
        sensor_usage = {}
        for d in data:
            for m_pos in d['measurement_positions']:
                # Determine which sensor likely created this measurement
                for sensor_id in range(1, 5):
                    sensor_pos = {1: (-1.5, -1.5), 2: (1.5, -1.5), 3: (1.5, 1.5), 4: (-1.5, 1.5)}[sensor_id]
                    dist = math.sqrt((m_pos[0] - sensor_pos[0])**2 + (m_pos[1] - sensor_pos[1])**2)
                    if dist <= 2.6:  # Within sensor range + margin
                        sensor_usage[sensor_id] = sensor_usage.get(sensor_id, 0) + 1
                        break
        
        print(f"  Sensor usage: {sensor_usage}")
        
        if region_name == "+x -y":
            print(f"  *** PROBLEM REGION DETECTED ***")
            if avg_measurements < 0.5:
                print(f"  Issue: Very few sensor measurements in this region")
            if avg_targets < avg_measurements * 0.5:
                print(f"  Issue: Poor fusion - measurements not creating targets")
            if avg_tracks < avg_targets * 0.5:
                print(f"  Issue: Poor tracking - targets not creating tracks")

if __name__ == "__main__":
    debug_circular_tracking()