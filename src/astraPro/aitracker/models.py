"""
Track data models for the tracking system.
"""

import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from .kalman import KalmanFilter


class TrackStatus(Enum):
    """Track lifecycle status."""
    NEW = "new"                # Just created, needs confirmation
    CONFIRMED = "confirmed"    # Confirmed track, actively being updated
    LOST = "lost"             # Temporarily lost, predicting


@dataclass
class TrackState:
    """Snapshot of track state at a specific time."""
    id: int
    state: np.ndarray          # [px, py, vx, vy]
    covariance: np.ndarray     # 4x4 uncertainty matrix
    timestamp: float
    status: TrackStatus        # NEW, CONFIRMED, LOST
    hits: int                  # Number of successful associations
    misses: int               # Number of missed detections
    confidence: float         # Track quality score (0.0 to 1.0)
    
    def get_position(self) -> Tuple[float, float]:
        """Get position from state."""
        return float(self.state[0]), float(self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity from state."""
        return float(self.state[2]), float(self.state[3])


class Track:
    """
    A persistent track representing a single target over time.
    """
    
    _next_id = 1
    
    def __init__(self, initial_detection, dt: float = 0.15):
        """
        Initialize track from first detection.
        
        Args:
            initial_detection: FusedTarget from triangulation
            dt: Time step for Kalman filter
        """
        self.id = Track._next_id
        Track._next_id += 1
        
        self.created_time = time.time()
        self.last_update = self.created_time
        
        # Initialize Kalman filter
        self.kalman = KalmanFilter(dt=dt)
        measurement = np.array([initial_detection.x, initial_detection.y])
        self.kalman.initialize_from_measurement(measurement)
        
        # Track statistics
        self.hits = 1
        self.misses = 0
        self.status = TrackStatus.NEW
        self.confidence = initial_detection.confidence
        
        # Store state history
        self.state_history: List[TrackState] = []
        self._add_state_to_history()
    
    def predict(self, dt: float) -> TrackState:
        """
        Predict track state at given time step.
        
        Args:
            dt: Time step for prediction
            
        Returns:
            Predicted track state
        """
        self.kalman.predict(dt)
        return self._create_current_state(time.time())
    
    def update(self, detection) -> TrackState:
        """
        Update track with new detection.
        
        Args:
            detection: FusedTarget from triangulation
            
        Returns:
            Updated track state
        """
        # Update with measurement
        measurement = np.array([detection.x, detection.y])
        self.kalman.update(measurement)
        
        # Update track statistics
        self.hits += 1
        self.misses = 0  # Reset miss counter
        self.last_update = time.time()
        self.confidence = min(1.0, self.confidence + 0.1)  # Boost confidence
        
        # Update status based on hits (rotating sensors: require 2 hits for confirmation)
        if self.status == TrackStatus.NEW and self.hits >= 2:
            self.status = TrackStatus.CONFIRMED
        elif self.status == TrackStatus.LOST:
            self.status = TrackStatus.CONFIRMED
        
        self._add_state_to_history()
        return self.get_current_state()
    
    def miss(self) -> TrackStatus:
        """
        Handle missed detection.
        
        Returns:
            Updated track status
        """
        self.misses += 1
        self.confidence = max(0.0, self.confidence - 0.2)  # Reduce confidence
        
        # Update status based on misses (rotating sensors: more tolerant)
        if self.misses >= 6 and self.status == TrackStatus.CONFIRMED:
            self.status = TrackStatus.LOST
        elif self.misses >= 5 and self.status == TrackStatus.NEW:
            self.status = TrackStatus.LOST
        
        self._add_state_to_history()
        return self.status
    
    def get_current_state(self) -> TrackState:
        """Get current track state."""
        return self._create_current_state(self.last_update)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return self.kalman.get_position()
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return self.kalman.get_velocity()
    
    def get_age(self) -> float:
        """Get track age in seconds."""
        return time.time() - self.created_time
    
    def mahalanobis_distance(self, detection) -> float:
        """
        Calculate Mahalanobis distance to detection.
        
        Args:
            detection: FusedTarget from triangulation
            
        Returns:
            Mahalanobis distance
        """
        measurement = np.array([detection.x, detection.y])
        return self.kalman.mahalanobis_distance(measurement)
    
    def should_delete(self) -> bool:
        """Check if track should be deleted."""
        return (self.status == TrackStatus.LOST and 
                (self.misses >= 10 or self.get_age() > 5.0))
    
    def _create_current_state(self, timestamp: float) -> TrackState:
        """Create TrackState from current filter state."""
        return TrackState(
            id=self.id,
            state=self.kalman.state.copy(),
            covariance=self.kalman.P.copy(),
            timestamp=timestamp,
            status=self.status,
            hits=self.hits,
            misses=self.misses,
            confidence=self.confidence
        )
    
    def _add_state_to_history(self):
        """Add current state to history."""
        current_state = self.get_current_state()
        self.state_history.append(current_state)
        
        # Limit history size
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]


@dataclass
class TrackingResult:
    """Results from tracking system."""
    tracks: List[Track]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.status == TrackStatus.CONFIRMED]
    
    def get_active_tracks(self) -> List[Track]:
        """Get active tracks (confirmed or new, not lost)."""
        return [t for t in self.tracks if t.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]]