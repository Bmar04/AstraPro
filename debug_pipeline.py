#!/usr/bin/env python3
"""
Deep debug investigation - trace every step of the pipeline
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

def debug_pipeline():
    print("=== DEEP PIPELINE DEBUG ===")
    
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    
    # Start components
    print("\n1. TESTING SIMULATOR...")
    sim = create_default_simulation()
    sim.start(raw_queue)
    
    # Let simulator run and collect measurements
    time.sleep(3)
    raw_measurements = []
    while not raw_queue.empty():
        m = raw_queue.get_nowait()
        raw_measurements.append(m)
    
    print(f"   Collected {len(raw_measurements)} raw measurements:")
    for i, m in enumerate(raw_measurements[:5]):  # Show first 5
        print(f"     {i+1}: Sensor {m.sensor_id}, angle={m.angle:.1f}°, dist={m.distance:.2f}m")
    
    if len(raw_measurements) == 0:
        print("   ERROR: No raw measurements from simulator!")
        sim.stop()
        return
    
    # Test preprocessor
    print(f"\n2. TESTING PREPROCESSOR...")
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=True)
    
    # Feed raw measurements to preprocessor
    for m in raw_measurements:
        raw_queue.put(m)
    
    time.sleep(1)  # Let preprocessor work
    
    processed_measurements = []
    while not processed_queue.empty():
        m = processed_queue.get_nowait()
        processed_measurements.append(m)
    
    print(f"   Got {len(processed_measurements)} processed measurements:")
    for i, m in enumerate(processed_measurements[:5]):  # Show first 5
        print(f"     {i+1}: Sensor {m.sensor_id}, world=({m.x_world:.2f}, {m.y_world:.2f})")
    
    if len(processed_measurements) == 0:
        print("   ERROR: No processed measurements from preprocessor!")
        sim.stop()
        return
    
    # Test triangulation/fusion
    print(f"\n3. TESTING TRIANGULATION...")
    fusion_engine = TriangulationEngine(max_distance=0.6, min_confidence=0.1)
    
    print(f"   Fusion config: max_distance={fusion_engine.max_distance}, min_confidence={fusion_engine.min_confidence}")
    
    # Add all processed measurements
    for m in processed_measurements:
        fusion_engine.add_measurement(m)
    
    print(f"   Added {len(processed_measurements)} measurements to fusion engine")
    print(f"   Current measurements in fusion: {len(fusion_engine.measurements)}")
    
    # Force fusion
    result = fusion_engine.fuse_targets()
    
    print(f"   Fusion result: {len(result.targets)} targets")
    for i, target in enumerate(result.targets):
        sensors_str = ", ".join(map(str, target.sensor_ids))
        print(f"     Target {i+1}: ({target.x:.2f}, {target.y:.2f}) conf={target.confidence:.2f} sensors=[{sensors_str}]")
    
    if len(result.targets) == 0:
        print("   ERROR: No targets from fusion!")
        
        # Debug fusion grouping
        print("   DEBUG: Checking fusion grouping...")
        groups = fusion_engine._group_measurements()
        print(f"   Groups found: {len(groups)}")
        for i, group in enumerate(groups):
            print(f"     Group {i+1}: {len(group)} measurements")
            for m in group:
                print(f"       Sensor {m.sensor_id}: ({m.x_world:.2f}, {m.y_world:.2f})")
        
        sim.stop()
        return
    
    # Test tracking
    print(f"\n4. TESTING TRACKING...")
    tracking_system = TrackerSystem(gate_threshold=2.5)
    
    print(f"   Tracker initialized")
    
    # Multiple tracking updates to build tracks
    for round in range(5):
        print(f"\n   Round {round+1}:")
        tracking_result = tracking_system.update(result.targets, result.timestamp + round * 0.15)
        
        active_tracks = tracking_result.get_active_tracks()
        confirmed_tracks = tracking_result.get_confirmed_tracks()
        
        print(f"     Tracks: {len(tracking_result.tracks)} total, {len(active_tracks)} active, {len(confirmed_tracks)} confirmed")
        
        for track in active_tracks:
            pos = track.get_position()
            print(f"       Track {track.id}: pos=({pos[0]:.2f}, {pos[1]:.2f}) status={track.status} hits={track.hits} conf={track.confidence:.2f}")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Raw measurements: {len(raw_measurements)}")
    print(f"Processed measurements: {len(processed_measurements)}")
    print(f"Fused targets: {len(result.targets)}")
    print(f"Final tracks: {len(tracking_result.tracks)}")
    
    if len(tracking_result.tracks) > 0:
        print("SUCCESS: Tracks are being created!")
    else:
        print("PROBLEM: No tracks created despite having targets")
    
    sim.stop()
    print("\nDebug complete")

if __name__ == "__main__":
    debug_pipeline()