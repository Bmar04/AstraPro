"""
Main tracking interface for AI tracker system.
"""

from typing import List, Optional
from .track_manager import TrackManager
from .models import TrackingResult


class TrackerSystem:
    """
    Main interface for the AI tracking system.
    """
    
    def __init__(self, 
                 dt: float = 0.15,
                 association_algorithm: str = "nearest_neighbor",
                 gate_threshold: float = 3.0,
                 process_noise: float = 2.0,
                 measurement_noise: float = 0.3):
        """
        Initialize tracker system.
        
        Args:
            dt: Time step between updates (seconds)
            association_algorithm: "nearest_neighbor" or "gnn"
            gate_threshold: Maximum Mahalanobis distance for association
            process_noise: Process noise for Kalman filters (m/s²)
            measurement_noise: Measurement noise for Kalman filters (meters)
        """
        self.track_manager = TrackManager(
            dt=dt,
            association_algorithm=association_algorithm,
            gate_threshold=gate_threshold
        )
        
        # Store parameters for new tracks
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def update(self, fused_targets: List, timestamp: Optional[float] = None) -> TrackingResult:
        """
        Update tracking system with new fused targets.
        
        Args:
            fused_targets: List of FusedTarget objects from triangulation
            timestamp: Current timestamp (uses current time if None)
            
        Returns:
            TrackingResult with updated tracks
        """
        return self.track_manager.update(fused_targets, timestamp)
    
    def get_confirmed_tracks(self):
        """Get only confirmed tracks."""
        return self.track_manager.get_confirmed_tracks()
    
    def get_active_tracks(self):
        """Get all active tracks."""
        return self.track_manager.get_active_tracks()
    
    def get_track_count(self) -> int:
        """Get total number of tracks."""
        return self.track_manager.get_track_count()
    
    def reset(self):
        """Reset tracker, clearing all tracks."""
        self.track_manager.reset()
    
    def get_statistics(self):
        """Get tracking statistics."""
        return self.track_manager.get_statistics()