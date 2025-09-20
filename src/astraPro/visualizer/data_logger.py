"""
Data logging utilities for tracking data export.
"""

import csv
import time
import os
from typing import List, Optional
from datetime import datetime


class TrackDataLogger:
    """
    CSV data logger for tracking results.
    """
    
    def __init__(self, output_dir: str = "data", session_name: Optional[str] = None):
        """
        Initialize data logger.
        
        Args:
            output_dir: Directory to save CSV files
            session_name: Name for this logging session (auto-generated if None)
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate session name
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"tracking_session_{timestamp}"
        
        self.session_name = session_name
        
        # File paths
        self.tracks_file = os.path.join(output_dir, f"{session_name}_tracks.csv")
        self.detections_file = os.path.join(output_dir, f"{session_name}_detections.csv")
        self.stats_file = os.path.join(output_dir, f"{session_name}_stats.csv")
        self.ground_truth_file = os.path.join(output_dir, f"{session_name}_ground_truth.csv")
        
        # Initialize CSV files
        self._init_csv_files()
        
        print(f"Data logging initialized:")
        print(f"  Session: {session_name}")
        print(f"  Tracks: {self.tracks_file}")
        print(f"  Detections: {self.detections_file}")
        print(f"  Stats: {self.stats_file}")
        print(f"  Ground Truth: {self.ground_truth_file}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        
        # Tracks CSV
        with open(self.tracks_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'track_id', 'status', 'pos_x', 'pos_y', 
                'vel_x', 'vel_y', 'confidence', 'hits', 'misses', 
                'age', 'pos_uncertainty', 'vel_uncertainty'
            ])
        
        # Detections CSV
        with open(self.detections_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'detection_id', 'pos_x', 'pos_y', 
                'confidence', 'sensor_ids', 'num_sensors'
            ])
        
        # Stats CSV
        with open(self.stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total_tracks', 'confirmed_tracks', 
                'new_tracks', 'lost_tracks', 'avg_confidence',
                'num_detections'
            ])
        
        # Ground Truth CSV
        with open(self.ground_truth_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'object_id', 'pos_x', 'pos_y', 
                'vel_x', 'vel_y', 'active'
            ])
    
    def log_tracking_result(self, tracking_result, fused_targets):
        """
        Log complete tracking result to CSV files.
        
        Args:
            tracking_result: TrackingResult from tracker
            fused_targets: List of FusedTarget detections
        """
        timestamp = tracking_result.timestamp
        
        # Log tracks
        self._log_tracks(tracking_result.tracks, timestamp)
        
        # Log detections
        self._log_detections(fused_targets, timestamp)
        
        # Log statistics
        self._log_statistics(tracking_result, fused_targets, timestamp)
    
    def _log_tracks(self, tracks, timestamp):
        """Log track data."""
        with open(self.tracks_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for track in tracks:
                pos = track.get_position()
                vel = track.get_velocity()
                
                writer.writerow([
                    timestamp,
                    track.id,
                    track.status.value,
                    pos[0],
                    pos[1],
                    vel[0],
                    vel[1],
                    track.confidence,
                    track.hits,
                    track.misses,
                    track.get_age(),
                    track.kalman.get_position_uncertainty(),
                    track.kalman.get_velocity_uncertainty()
                ])
    
    def _log_detections(self, detections, timestamp):
        """Log detection data."""
        with open(self.detections_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i, detection in enumerate(detections):
                sensor_ids_str = ','.join(map(str, detection.sensor_ids))
                
                writer.writerow([
                    timestamp,
                    i,
                    detection.x,
                    detection.y,
                    detection.confidence,
                    sensor_ids_str,
                    len(detection.sensor_ids)
                ])
    
    def _log_statistics(self, tracking_result, detections, timestamp):
        """Log session statistics."""
        tracks = tracking_result.tracks
        
        # Calculate statistics
        confirmed_tracks = len([t for t in tracks if t.status.value == 'confirmed'])
        new_tracks = len([t for t in tracks if t.status.value == 'new'])
        lost_tracks = len([t for t in tracks if t.status.value == 'lost'])
        
        avg_confidence = sum(t.confidence for t in tracks) / max(len(tracks), 1)
        
        with open(self.stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                len(tracks),
                confirmed_tracks,
                new_tracks,
                lost_tracks,
                avg_confidence,
                len(detections)
            ])
    
    def log_raw_measurement(self, measurement, timestamp=None):
        """Log raw sensor measurement (for debugging)."""
        if timestamp is None:
            timestamp = time.time()
        
        # Create raw measurements file if needed
        raw_file = os.path.join(self.output_dir, f"{self.session_name}_raw.csv")
        
        # Check if file exists and create header if needed
        file_exists = os.path.exists(raw_file)
        
        with open(raw_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    'timestamp', 'sensor_id', 'distance', 'pan', 'tilt',
                    'x_world', 'y_world', 'confidence'
                ])
            
            writer.writerow([
                timestamp,
                measurement.sensor_id,
                measurement.distance,
                measurement.pan,
                measurement.tilt,
                measurement.x_world,
                measurement.y_world,
                getattr(measurement, 'confidence', 1.0)
            ])
    
    def log_ground_truth(self, objects, timestamp=None):
        """
        Log ground truth object positions.
        
        Args:
            objects: List of ObjectState from simulator
            timestamp: Timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = time.time()
            
        with open(self.ground_truth_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i, obj in enumerate(objects):
                writer.writerow([
                    timestamp,
                    i,  # object_id
                    obj.x,
                    obj.y,
                    obj.vx,
                    obj.vy,
                    True  # active (assume all logged objects are active)
                ])

    def close(self):
        """Close logger and print summary."""
        print(f"\nData logging session '{self.session_name}' completed.")
        print(f"Files saved in: {self.output_dir}/")
    
    def get_file_paths(self):
        """Get all CSV file paths."""
        return {
            'tracks': self.tracks_file,
            'detections': self.detections_file,
            'stats': self.stats_file,
            'ground_truth': self.ground_truth_file,
            'session_name': self.session_name
        }