#!/usr/bin/env python3
"""
Complete system investigation - trace object movements and tracking
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

def investigate_system():
    print("=== COMPLETE SYSTEM INVESTIGATION ===")
    
    # Create simulator
    sim = create_default_simulation()
    
    print("\n1. CHECKING OBJECT GROUND TRUTH...")
    ground_truth = sim.get_ground_truth_objects()
    print(f"   Number of objects: {len(ground_truth)}")
    
    for i, obj in enumerate(ground_truth):
        pattern_type = type(sim.movement_patterns[i]).__name__
        print(f"   Object {i+1}: pos=({obj.x:.2f}, {obj.y:.2f}) vel=({obj.vx:.2f}, {obj.vy:.2f}) pattern={pattern_type}")
    
    # Test movement over time
    print(f"\n2. TESTING OBJECT MOVEMENT OVER 5 SECONDS...")
    for step in range(10):  # 10 steps of 0.5 seconds each
        sim.update_simulation(0.5)
        ground_truth = sim.get_ground_truth_objects()
        
        print(f"   Step {step+1} (t={step*0.5:.1f}s):")
        for i, obj in enumerate(ground_truth):
            pattern_type = type(sim.movement_patterns[i]).__name__
            print(f"     {pattern_type}: pos=({obj.x:.2f}, {obj.y:.2f})")
        print()
    
    # Test sensor detection
    print(f"\n3. TESTING SENSOR DETECTION...")
    raw_queue = queue.Queue()
    sim.start(raw_queue)
    
    time.sleep(3)  # Let it generate measurements
    
    raw_measurements = []
    while not raw_queue.empty():
        m = raw_queue.get_nowait()
        raw_measurements.append(m)
    
    sim.stop()
    
    print(f"   Generated {len(raw_measurements)} raw measurements:")
    # Group by sensor
    sensor_measurements = {}
    for m in raw_measurements:
        if m.sensor_id not in sensor_measurements:
            sensor_measurements[m.sensor_id] = []
        sensor_measurements[m.sensor_id].append(m)
    
    for sensor_id, measurements in sensor_measurements.items():
        print(f"     Sensor {sensor_id}: {len(measurements)} measurements")
        for m in measurements[:3]:  # Show first 3
            print(f"       angle={m.angle:.1f}°, dist={m.distance:.2f}m")
    
    # Test full pipeline with live tracking
    print(f"\n4. TESTING LIVE TRACKING PIPELINE...")
    
    # Restart simulator
    sim = create_default_simulation()
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    sim.start(raw_queue)
    
    # Start preprocessor
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=False)
    
    # Create tracking components
    fusion_engine = TriangulationEngine(max_distance=0.6, min_confidence=0.1)
    tracking_system = TrackerSystem(gate_threshold=2.5)
    
    print("   Running live tracking for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        # Get ground truth for comparison
        ground_truth = sim.get_ground_truth_objects()
        
        # Process measurements
        while not processed_queue.empty():
            try:
                processed_measurement = processed_queue.get_nowait()
                fusion_engine.add_measurement(processed_measurement)
            except queue.Empty:
                break
        
        # Run fusion and tracking
        if fusion_engine.should_fuse():
            result = fusion_engine.fuse_targets()
            
            if result.targets:
                tracking_result = tracking_system.update(result.targets, result.timestamp)
                active_tracks = tracking_result.get_active_tracks()
                
                if active_tracks:
                    print(f"\n   t={time.time()-start_time:.1f}s:")
                    print(f"     GROUND TRUTH:")
                    for i, obj in enumerate(ground_truth):
                        pattern_type = type(sim.movement_patterns[i]).__name__
                        print(f"       {pattern_type}: pos=({obj.x:.2f}, {obj.y:.2f})")
                    
                    print(f"     TRACKS ({len(active_tracks)} active):")
                    for track in active_tracks:
                        pos = track.get_position()
                        print(f"       Track {track.id}: pos=({pos[0]:.2f}, {pos[1]:.2f}) status={track.status}")
                    print()
        
        time.sleep(0.1)
    
    sim.stop()
    
    print("\n=== INVESTIGATION COMPLETE ===")
    print("Look for:")
    print("1. Are objects moving in expected patterns? (circle, diagonal line, random)")
    print("2. Are sensors detecting objects correctly?")  
    print("3. Are tracks following ground truth movements?")
    print("4. Are there multiple false tracks instead of pattern-following tracks?")

if __name__ == "__main__":
    investigate_system()