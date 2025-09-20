#!/usr/bin/env python3
"""
Test the measurement pipeline step by step
"""

import sys
import os
import time
import queue

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.io_module.simulator import create_default_simulation
from astraPro.preprocessor.processor import start_preprocessor
from astraPro.preprocessor.triangulation import TriangulationEngine

def test_pipeline():
    print("=== Testing Pipeline Step by Step ===")
    
    # Step 1: Test simulator
    print("\n1. Testing simulator...")
    sim = create_default_simulation()
    raw_queue = queue.Queue()
    sim.start(raw_queue)
    
    # Collect some measurements
    time.sleep(2)
    measurements = []
    while not raw_queue.empty():
        m = raw_queue.get_nowait()
        measurements.append(m)
        print(f"   Raw measurement: sensor={m.sensor_id}, angle={m.angle:.1f}, dist={m.distance:.2f}")
    
    print(f"   Got {len(measurements)} raw measurements")
    sim.stop()
    
    if len(measurements) == 0:
        print("   ERROR: No measurements from simulator!")
        return
    
    # Step 2: Test preprocessor
    print("\n2. Testing preprocessor...")
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    
    # Add measurements to raw queue
    for m in measurements:
        raw_queue.put(m)
    
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=True)
    
    # Wait for processing
    time.sleep(1)
    
    processed_measurements = []
    while not processed_queue.empty():
        m = processed_queue.get_nowait()
        processed_measurements.append(m)
        print(f"   Processed measurement: sensor={m.sensor_id}, x={m.x_world:.2f}, y={m.y_world:.2f}")
    
    print(f"   Got {len(processed_measurements)} processed measurements")
    
    if len(processed_measurements) == 0:
        print("   ERROR: No processed measurements!")
        return
    
    # Step 3: Test triangulation
    print("\n3. Testing triangulation...")
    fusion_engine = TriangulationEngine(max_distance=0.6, min_confidence=0.1)
    
    # Add processed measurements
    for m in processed_measurements:
        fusion_engine.add_measurement(m)
    
    # Force fusion
    result = fusion_engine.fuse_targets()
    
    print(f"   Got {len(result.targets)} fused targets")
    for i, target in enumerate(result.targets):
        print(f"   Target {i+1}: ({target.x:.2f}, {target.y:.2f}) confidence={target.confidence:.2f}")
    
    print("\n=== Pipeline Test Complete ===")

if __name__ == "__main__":
    test_pipeline()