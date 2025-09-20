"""
AstraPro Multi-Target Tracking System - Main Entry Point
"""

import sys
import os
import time
import queue
import threading

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.io_module.serial_reader import SensorReader
from astraPro.io_module.simulator import simulate_measurements, create_default_simulation
from astraPro.preprocessor.processor import start_preprocessor
from astraPro.preprocessor.triangulation import TriangulationEngine


def main():
    import sys
    
    debug_print = "--debug" in sys.argv
    simulate = "--sim" in sys.argv
    use_aitracker = "--aitracker" in sys.argv
    use_tracker = "--tracker" in sys.argv
    log_data = "--log" in sys.argv
    visualize = "--viz" in sys.argv
    
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    
    # Start preprocessor
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=debug_print)
    
    # Create triangulation engine for fusion (more permissive for rotating sensors)
    fusion_engine = TriangulationEngine(max_distance=1.0, min_confidence=0.05)
    
    # Initialize tracking system if requested
    tracking_system = None
    if use_aitracker:
        print("Using AI Tracker...")
        from astraPro.aitracker.tracker import TrackerSystem
        tracking_system = TrackerSystem(gate_threshold=3.5)
    elif use_tracker:
        print("Using Manual Tracker...")
        print("Manual tracker not implemented yet")
        tracking_system = None
    
    # Initialize visualization and logging
    data_logger = None
    live_tracker = None
    simulator = None
    
    if log_data:
        from astraPro.visualizer.data_logger import TrackDataLogger
        data_logger = TrackDataLogger()
    
    if visualize:
        try:
            from astraPro.visualizer.working_viz import WorkingVisualizer
            live_tracker = WorkingVisualizer()
            print("Working visualizer started")
        except Exception as e:
            print(f"Warning: visualization failed: {e}")
            live_tracker = None
    
    if simulate:
        print("Starting with simulation...")
        simulator = create_default_simulation()
        simulator.start(raw_queue)
    else:
        reader = SensorReader("COM3")
        print("Starting with serial data...")
    
    try:
        while True:
            # Process measurements from processed queue
            while not processed_queue.empty():
                try:
                    processed_measurement = processed_queue.get_nowait()
                    fusion_engine.add_measurement(processed_measurement)
                except queue.Empty:
                    break
            
            # Run fusion when it's time
            if fusion_engine.should_fuse():
                result = fusion_engine.fuse_targets()
                
                # Process with tracking system if available
                if tracking_system is not None:
                    tracking_result = tracking_system.update(result.targets, result.timestamp)
                    
                    # Log data if enabled
                    if data_logger:
                        data_logger.log_tracking_result(tracking_result, result.targets)
                        # Log ground truth if simulation is running
                        if simulator:
                            ground_truth_objects = simulator.get_ground_truth_objects()
                            data_logger.log_ground_truth(ground_truth_objects, result.timestamp)
                    
                    # Update live visualization if enabled
                    if live_tracker:
                        live_tracker.update(tracking_result, result.targets)
                    
                    # Print tracking results (only in debug mode)
                    if debug_print:
                        confirmed_tracks = tracking_result.get_confirmed_tracks()
                        active_tracks = tracking_result.get_active_tracks()
                        
                        if active_tracks:
                            print(f"\n=== TRACKS (t={tracking_result.timestamp:.2f}) ===")
                            print(f"Active: {len(active_tracks)}, Confirmed: {len(confirmed_tracks)}")
                            
                            for track in confirmed_tracks:
                                pos = track.get_position()
                                vel = track.get_velocity()
                                print(f"Track {track.id}: pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                                      f"vel=({vel[0]:.2f}, {vel[1]:.2f}) "
                                      f"conf={track.confidence:.2f} hits={track.hits}")
                        else:
                            print("No active tracks")
                else:
                    # Print fused targets (only in debug mode) when no tracking
                    if debug_print:
                        if result.targets:
                            print(f"\n=== FUSED TARGETS (t={result.timestamp:.2f}) ===")
                            for i, target in enumerate(result.targets):
                                sensors_str = ", ".join(map(str, target.sensor_ids))
                                print(f"Target {i+1}: ({target.x:.2f}, {target.y:.2f}) "
                                      f"confidence={target.confidence:.2f} sensors=[{sensors_str}]")
                        else:
                            print("No targets detected")
            
            # Read serial if not simulating
            if not simulate:
                measurement = reader.get_measurement()
                if measurement:
                    raw_queue.put(measurement)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup visualization and logging
        if live_tracker:
            live_tracker.close()
        if data_logger:
            data_logger.close()
        if simulator:
            simulator.stop()


if __name__ == "__main__":
    main()