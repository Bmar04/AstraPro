#!/usr/bin/env python3
"""
Test visualizer for 10 seconds then stop
"""

import sys
import os
import time
import queue
import threading

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.io_module.simulator import create_default_simulation
from astraPro.preprocessor.processor import start_preprocessor
from astraPro.preprocessor.triangulation import TriangulationEngine
from astraPro.aitracker.tracker import TrackerSystem

def test_viz():
    print("=== Testing Visualizer for 10 seconds ===")
    
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    
    # Start components
    sim = create_default_simulation()
    sim.start(raw_queue)
    
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=False)
    
    fusion_engine = TriangulationEngine(max_distance=0.6, min_confidence=0.1)
    tracking_system = TrackerSystem(gate_threshold=2.5)
    
    # Initialize visualization
    try:
        from astraPro.visualizer.live_tracker import LiveTracker
        live_tracker = LiveTracker()
        import matplotlib.pyplot as plt
        plt.ion()
        live_tracker.start()
        print("Live tracker started")
    except Exception as e:
        print(f"Error starting visualizer: {e}")
        return
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10:  # Run for 10 seconds
            # Process measurements
            while not processed_queue.empty():
                try:
                    processed_measurement = processed_queue.get_nowait()
                    fusion_engine.add_measurement(processed_measurement)
                except queue.Empty:
                    break
            
            # Run fusion
            if fusion_engine.should_fuse():
                result = fusion_engine.fuse_targets()
                
                if result.targets:
                    tracking_result = tracking_system.update(result.targets, result.timestamp)
                    
                    # Update visualization
                    live_tracker.add_tracking_data(tracking_result, result.targets)
                    
                    # Print debug info
                    confirmed_tracks = tracking_result.get_confirmed_tracks()
                    active_tracks = tracking_result.get_active_tracks()
                    
                    if active_tracks:
                        print(f"TRACKS: Active={len(active_tracks)}, Confirmed={len(confirmed_tracks)}")
            
            # Process matplotlib events
            plt.pause(0.01)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print("Stopping...")
        sim.stop()
        live_tracker.stop()
        plt.ioff()
    
    print("Test complete")

if __name__ == "__main__":
    test_viz()