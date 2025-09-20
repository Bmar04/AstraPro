"""
Data association algorithms for tracking system.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .models import Track


class AssociationResult:
    """Result of data association between detections and tracks."""
    
    def __init__(self):
        self.associations: Dict[int, int] = {}  # track_id -> detection_index
        self.unassociated_tracks: List[int] = []  # track_ids with no detection
        self.unassociated_detections: List[int] = []  # detection indices with no track
    
    def add_association(self, track_id: int, detection_idx: int):
        """Add association between track and detection."""
        self.associations[track_id] = detection_idx
    
    def add_unassociated_track(self, track_id: int):
        """Add track that received no detection."""
        self.unassociated_tracks.append(track_id)
    
    def add_unassociated_detection(self, detection_idx: int):
        """Add detection that was not associated with any track."""
        self.unassociated_detections.append(detection_idx)
    
    def get_detection_for_track(self, track_id: int) -> Optional[int]:
        """Get detection index associated with track."""
        return self.associations.get(track_id)
    
    def is_detection_associated(self, detection_idx: int) -> bool:
        """Check if detection is associated with any track."""
        return detection_idx in self.associations.values()


class NearestNeighborAssociator:
    """
    Nearest neighbor data association using Mahalanobis distance.
    """
    
    def __init__(self, gate_threshold: float = 3.0):
        """
        Initialize associator.
        
        Args:
            gate_threshold: Maximum Mahalanobis distance for valid association
        """
        self.gate_threshold = gate_threshold
    
    def associate(self, tracks: List[Track], detections: List) -> AssociationResult:
        """
        Associate detections with tracks using nearest neighbor.
        
        Args:
            tracks: List of existing tracks
            detections: List of FusedTarget detections
            
        Returns:
            AssociationResult with associations and unassociated items
        """
        result = AssociationResult()
        
        if not tracks or not detections:
            # Handle empty cases
            for i, track in enumerate(tracks):
                result.add_unassociated_track(track.id)
            for i in range(len(detections)):
                result.add_unassociated_detection(i)
            return result
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(tracks, detections)
        
        # Perform nearest neighbor assignment
        used_detections = set()
        used_tracks = set()
        
        # Sort all possible associations by distance
        associations = []
        for track_idx, track in enumerate(tracks):
            for det_idx in range(len(detections)):
                distance = distance_matrix[track_idx, det_idx]
                if distance <= self.gate_threshold:
                    associations.append((distance, track_idx, det_idx))
        
        # Sort by distance (best first)
        associations.sort(key=lambda x: x[0])
        
        # Assign in order of best distance
        for distance, track_idx, det_idx in associations:
            track_id = tracks[track_idx].id
            
            if track_id not in used_tracks and det_idx not in used_detections:
                result.add_association(track_id, det_idx)
                used_tracks.add(track_id)
                used_detections.add(det_idx)
        
        # Add unassociated tracks
        for track in tracks:
            if track.id not in used_tracks:
                result.add_unassociated_track(track.id)
        
        # Add unassociated detections
        for det_idx in range(len(detections)):
            if det_idx not in used_detections:
                result.add_unassociated_detection(det_idx)
        
        return result
    
    def _calculate_distance_matrix(self, tracks: List[Track], detections: List) -> np.ndarray:
        """Calculate Mahalanobis distance matrix between tracks and detections."""
        num_tracks = len(tracks)
        num_detections = len(detections)
        distance_matrix = np.full((num_tracks, num_detections), np.inf)
        
        for track_idx, track in enumerate(tracks):
            for det_idx, detection in enumerate(detections):
                distance = track.mahalanobis_distance(detection)
                distance_matrix[track_idx, det_idx] = distance
        
        return distance_matrix


class GlobalNearestNeighborAssociator:
    """
    Global Nearest Neighbor (GNN) associator with gating.
    Uses Hungarian algorithm for optimal assignment.
    """
    
    def __init__(self, gate_threshold: float = 3.0):
        """
        Initialize GNN associator.
        
        Args:
            gate_threshold: Maximum Mahalanobis distance for valid association
        """
        self.gate_threshold = gate_threshold
    
    def associate(self, tracks: List[Track], detections: List) -> AssociationResult:
        """
        Associate detections with tracks using Hungarian algorithm.
        
        Args:
            tracks: List of existing tracks
            detections: List of FusedTarget detections
            
        Returns:
            AssociationResult with optimal associations
        """
        result = AssociationResult()
        
        if not tracks or not detections:
            # Handle empty cases
            for track in tracks:
                result.add_unassociated_track(track.id)
            for i in range(len(detections)):
                result.add_unassociated_detection(i)
            return result
        
        # Calculate distance matrix with gating
        distance_matrix = self._calculate_gated_distance_matrix(tracks, detections)
        
        # Apply Hungarian algorithm
        assignments = self._hungarian_assignment(distance_matrix)
        
        # Process assignments
        for track_idx, det_idx in assignments:
            if (track_idx < len(tracks) and det_idx < len(detections) and
                distance_matrix[track_idx, det_idx] < np.inf):
                track_id = tracks[track_idx].id
                result.add_association(track_id, det_idx)
        
        # Find unassociated tracks
        assigned_tracks = {tracks[track_idx].id for track_idx, _ in assignments 
                          if track_idx < len(tracks)}
        for track in tracks:
            if track.id not in assigned_tracks:
                result.add_unassociated_track(track.id)
        
        # Find unassociated detections
        assigned_detections = {det_idx for _, det_idx in assignments 
                             if det_idx < len(detections)}
        for det_idx in range(len(detections)):
            if det_idx not in assigned_detections:
                result.add_unassociated_detection(det_idx)
        
        return result
    
    def _calculate_gated_distance_matrix(self, tracks: List[Track], detections: List) -> np.ndarray:
        """Calculate distance matrix with gating (invalid associations set to inf)."""
        num_tracks = len(tracks)
        num_detections = len(detections)
        distance_matrix = np.full((num_tracks, num_detections), np.inf)
        
        for track_idx, track in enumerate(tracks):
            for det_idx, detection in enumerate(detections):
                distance = track.mahalanobis_distance(detection)
                if distance <= self.gate_threshold:
                    distance_matrix[track_idx, det_idx] = distance
        
        return distance_matrix
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Simple Hungarian algorithm implementation.
        For production, consider using scipy.optimize.linear_sum_assignment.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            return list(zip(row_indices, col_indices))
        except ImportError:
            # Fallback to greedy assignment if scipy not available
            return self._greedy_assignment(cost_matrix)
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback greedy assignment algorithm."""
        assignments = []
        cost_matrix = cost_matrix.copy()
        
        while True:
            # Find minimum cost
            min_pos = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            min_cost = cost_matrix[min_pos]
            
            if min_cost == np.inf:
                break
            
            # Add assignment
            assignments.append(min_pos)
            
            # Mark row and column as used
            cost_matrix[min_pos[0], :] = np.inf
            cost_matrix[:, min_pos[1]] = np.inf
        
        return assignments


def create_associator(algorithm: str = "nearest_neighbor", gate_threshold: float = 3.0):
    """
    Factory function to create associator.
    
    Args:
        algorithm: "nearest_neighbor" or "gnn" (Global Nearest Neighbor)
        gate_threshold: Maximum Mahalanobis distance for valid association
        
    Returns:
        Associator instance
    """
    if algorithm == "nearest_neighbor":
        return NearestNeighborAssociator(gate_threshold)
    elif algorithm == "gnn":
        return GlobalNearestNeighborAssociator(gate_threshold)
    else:
        raise ValueError(f"Unknown association algorithm: {algorithm}")