"""
Track manager for handling multiple tracks lifecycle.
"""

import time
from typing import List, Dict, Optional
from .models import Track, TrackStatus, TrackingResult
from .association import create_associator, AssociationResult


class TrackManager:
    """
    Manages multiple tracks with automatic creation, confirmation, and deletion.
    """
    
    def __init__(self, 
                 dt: float = 0.15,
                 association_algorithm: str = "nearest_neighbor",
                 gate_threshold: float = 3.0,
                 max_tracks: int = 20,
                 confirm_hits: int = 3,
                 delete_misses: int = 10):
        """
        Initialize track manager.
        
        Args:
            dt: Time step for Kalman filters
            association_algorithm: "nearest_neighbor" or "gnn"
            gate_threshold: Maximum Mahalanobis distance for association
            max_tracks: Maximum number of tracks to maintain
            confirm_hits: Number of hits needed to confirm a track
            delete_misses: Number of misses before deleting a track
        """
        self.dt = dt
        self.max_tracks = max_tracks
        self.confirm_hits = confirm_hits
        self.delete_misses = delete_misses
        
        # Create associator
        self.associator = create_associator(association_algorithm, gate_threshold)
        
        # Track storage
        self.tracks: Dict[int, Track] = {}
        self.last_update_time = time.time()
    
    def update(self, detections: List, timestamp: Optional[float] = None) -> TrackingResult:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of FusedTarget detections
            timestamp: Current timestamp (uses current time if None)
            
        Returns:
            TrackingResult with updated tracks
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time step since last update
        dt = timestamp - self.last_update_time
        self.last_update_time = timestamp
        
        # Predict all existing tracks forward in time
        self._predict_tracks(dt)
        
        # Associate detections with tracks
        association_result = self._associate_detections(detections)
        
        # Update tracks with associated detections
        self._update_associated_tracks(association_result, detections)
        
        # Handle missed detections
        self._handle_missed_tracks(association_result)
        
        # Create new tracks from unassociated detections
        self._create_new_tracks(association_result, detections)
        
        # Clean up old/invalid tracks
        self._cleanup_tracks()
        
        # Return current tracking result
        return TrackingResult(list(self.tracks.values()), timestamp)
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks (confirmed or new)."""
        return [track for track in self.tracks.values() 
                if track.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [track for track in self.tracks.values() 
                if track.status == TrackStatus.CONFIRMED]
    
    def get_track_count(self) -> int:
        """Get total number of tracks."""
        return len(self.tracks)
    
    def reset(self):
        """Reset track manager, clearing all tracks."""
        self.tracks.clear()
        Track._next_id = 1  # Reset track ID counter
        self.last_update_time = time.time()
    
    def _predict_tracks(self, dt: float):
        """Predict all tracks forward in time."""
        if dt > 0:
            for track in self.tracks.values():
                track.predict(dt)
    
    def _associate_detections(self, detections: List) -> AssociationResult:
        """Associate detections with existing tracks."""
        active_tracks = self.get_active_tracks()
        return self.associator.associate(active_tracks, detections)
    
    def _update_associated_tracks(self, association_result: AssociationResult, detections: List):
        """Update tracks that were associated with detections."""
        for track_id, detection_idx in association_result.associations.items():
            if track_id in self.tracks:
                track = self.tracks[track_id]
                detection = detections[detection_idx]
                track.update(detection)
                
                # Update track status based on hits
                if (track.status == TrackStatus.NEW and 
                    track.hits >= self.confirm_hits):
                    track.status = TrackStatus.CONFIRMED
    
    def _handle_missed_tracks(self, association_result: AssociationResult):
        """Handle tracks that didn't receive detections."""
        for track_id in association_result.unassociated_tracks:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.miss()
    
    def _create_new_tracks(self, association_result: AssociationResult, detections: List):
        """Create new tracks from unassociated detections."""
        # Check if we have room for new tracks
        if len(self.tracks) >= self.max_tracks:
            return
        
        for detection_idx in association_result.unassociated_detections:
            # Check track limit again
            if len(self.tracks) >= self.max_tracks:
                break
            
            detection = detections[detection_idx]
            
            # Create new track
            new_track = Track(detection, dt=self.dt)
            self.tracks[new_track.id] = new_track
    
    def _cleanup_tracks(self):
        """Remove tracks that should be deleted."""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Delete tracks that have been lost for too long
            if (track.status == TrackStatus.LOST and 
                track.misses >= self.delete_misses):
                tracks_to_delete.append(track_id)
            
            # Delete new tracks that haven't been confirmed and are old
            elif (track.status == TrackStatus.NEW and 
                  track.get_age() > 2.0 and track.misses >= 3):
                tracks_to_delete.append(track_id)
        
        # Remove deleted tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_tracks_near_position(self, x: float, y: float, max_distance: float = 1.0) -> List[Track]:
        """Get tracks near a given position."""
        nearby_tracks = []
        for track in self.tracks.values():
            if track.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]:
                pos = track.get_position()
                distance = ((pos[0] - x)**2 + (pos[1] - y)**2)**0.5
                if distance <= max_distance:
                    nearby_tracks.append(track)
        return nearby_tracks
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        status_counts = {status: 0 for status in TrackStatus}
        for track in self.tracks.values():
            status_counts[track.status] += 1
        
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': status_counts[TrackStatus.CONFIRMED],
            'new_tracks': status_counts[TrackStatus.NEW],
            'lost_tracks': status_counts[TrackStatus.LOST],
            'average_confidence': sum(t.confidence for t in self.tracks.values()) / max(len(self.tracks), 1)
        }