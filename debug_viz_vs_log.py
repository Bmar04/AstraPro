#!/usr/bin/env python3
"""
Debug the difference between what the live visualizer shows vs what gets logged
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
from astraPro.aitracker.tracker import TrackerSystem
from astraPro.visualizer.data_logger import TrackDataLogger

def debug_viz_vs_log():
    print("=== DEBUGGING VISUALIZER VS LOGGING ===")
    
    # Start components
    raw_queue = queue.Queue()
    processed_queue = queue.Queue()
    
    sim = create_default_simulation("circle")  # Only circular movement
    sim.start(raw_queue)
    
    config_path = os.path.join(project_root, "config.yaml")
    preprocessor_thread = start_preprocessor(raw_queue, processed_queue, config_path, debug_print=False)
    
    fusion_engine = TriangulationEngine(max_distance=1.0, min_confidence=0.05)
    tracking_system = TrackerSystem(gate_threshold=3.5)
    
    # Initialize logger
    data_logger = TrackDataLogger(session_name="debug_viz_test")
    
    print("Running for 20 seconds to compare visualizer vs logging...")
    start_time = time.time()
    
    track_ids_seen_live = set()
    track_ids_logged = set()
    frame_count = 0
    
    while time.time() - start_time < 20:
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
                
                # Log the data (what analysis tool will see)
                data_logger.log_tracking_result(tracking_result, result.targets)
                
                # Simulate what visualizer sees
                all_tracks = tracking_result.tracks  # All tracks from tracker
                active_tracks = tracking_result.get_active_tracks()  # Active tracks only
                
                frame_count += 1
                
                # Track what we see
                for track in all_tracks:
                    track_ids_logged.add(track.id)
                
                for track in active_tracks:
                    track_ids_seen_live.add(track.id)
                
                # Print comparison every 20 frames
                if frame_count % 20 == 0:
                    print(f"\nFrame {frame_count} (t={time.time()-start_time:.1f}s):")
                    print(f"  ALL tracks from tracker: {len(all_tracks)} tracks")
                    for track in all_tracks:
                        print(f"    Track {track.id}: status={track.status.value}, hits={track.hits}, conf={track.confidence:.2f}")
                    
                    print(f"  ACTIVE tracks (what visualizer might show): {len(active_tracks)} tracks")
                    for track in active_tracks:
                        print(f"    Track {track.id}: status={track.status.value}, hits={track.hits}, conf={track.confidence:.2f}")
                    
                    print(f"  Track IDs seen live so far: {sorted(track_ids_seen_live)}")
                    print(f"  Track IDs logged so far: {sorted(track_ids_logged)}")
        
        time.sleep(0.1)
    
    sim.stop()
    data_logger.close()
    
    print(f"\n=== FINAL COMPARISON ===")
    print(f"Track IDs seen in live tracking: {sorted(track_ids_seen_live)}")
    print(f"Track IDs that got logged: {sorted(track_ids_logged)}")
    print(f"Difference: {sorted(track_ids_logged - track_ids_seen_live)}")
    
    # Check what's in the CSV
    print(f"\n=== CHECKING CSV FILE ===")
    import pandas as pd
    df = pd.read_csv(data_logger.tracks_file)
    csv_track_ids = set(df['track_id'].unique())
    print(f"Track IDs in CSV: {sorted(csv_track_ids)}")
    print(f"CSV track count by status:")
    print(df['status'].value_counts())
    
    # Check if there's a difference in filtering
    if len(track_ids_seen_live) == 1 and len(csv_track_ids) > 1:
        print(f"\n*** FOUND THE ISSUE ***")
        print(f"Live tracking only shows active tracks, but CSV logs ALL tracks (including lost ones)")
        print("This explains why visualizer shows 1 track but analysis finds many")

if __name__ == "__main__":
    debug_viz_vs_log()