"""
Basic unit tests for AI tracker implementation.
"""

import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aitracker.kalman import KalmanFilter
from aitracker.models import Track, TrackStatus
from preprocessor.models import FusedTarget


def test_kalman_initialization():
    """Test Kalman filter initialization."""
    kf = KalmanFilter(dt=0.1, process_noise=1.0, measurement_noise=0.1)
    
    # Check initial state is zero
    assert np.allclose(kf.state, np.zeros(4))
    
    # Check matrices have correct shapes
    assert kf.F.shape == (4, 4)
    assert kf.H.shape == (2, 4)
    assert kf.Q.shape == (4, 4)
    assert kf.R.shape == (2, 2)
    assert kf.P.shape == (4, 4)
    
    print("✓ Kalman filter initialization test passed")


def test_kalman_prediction():
    """Test prediction with constant velocity model."""
    kf = KalmanFilter(dt=0.1)
    
    # Initialize with position and velocity
    measurement = np.array([0.0, 0.0])
    velocity = (1.0, 0.5)
    kf.initialize_from_measurement(measurement, velocity)
    
    # Predict one time step
    state, cov = kf.predict(dt=0.1)
    
    # Check position moved according to velocity
    expected_x = 0.0 + 1.0 * 0.1  # x + vx * dt
    expected_y = 0.0 + 0.5 * 0.1  # y + vy * dt
    
    assert abs(state[0] - expected_x) < 1e-10
    assert abs(state[1] - expected_y) < 1e-10
    assert abs(state[2] - 1.0) < 1e-10  # vx unchanged
    assert abs(state[3] - 0.5) < 1e-10  # vy unchanged
    
    print("✓ Kalman filter prediction test passed")


def test_kalman_update():
    """Test update with measurement."""
    kf = KalmanFilter()
    
    # Initialize
    initial_measurement = np.array([0.0, 0.0])
    kf.initialize_from_measurement(initial_measurement)
    
    # Get initial uncertainty
    initial_uncertainty = kf.get_position_uncertainty()
    
    # Update with another measurement
    new_measurement = np.array([0.1, 0.1])
    state, cov = kf.update(new_measurement)
    
    # Position should move toward measurement
    assert 0.0 < state[0] <= 0.1
    assert 0.0 < state[1] <= 0.1
    
    # Uncertainty should decrease after measurement
    new_uncertainty = kf.get_position_uncertainty()
    assert new_uncertainty < initial_uncertainty
    
    print("✓ Kalman filter update test passed")


def test_track_creation():
    """Test track creation from detection."""
    # Create a fake detection
    detection = FusedTarget(
        x=1.0, y=2.0, confidence=0.8, 
        sensor_ids=[1, 2], timestamp=0.0
    )
    
    track = Track(detection, dt=0.1)
    
    # Check track initialization
    assert track.id >= 1
    assert track.hits == 1
    assert track.misses == 0
    assert track.status == TrackStatus.NEW
    assert track.confidence == 0.8
    
    # Check position
    pos = track.get_position()
    assert abs(pos[0] - 1.0) < 1e-6
    assert abs(pos[1] - 2.0) < 1e-6
    
    print("✓ Track creation test passed")


def test_track_update():
    """Test track update with new detection."""
    # Create initial detection
    detection1 = FusedTarget(
        x=1.0, y=1.0, confidence=0.8,
        sensor_ids=[1, 2], timestamp=0.0
    )
    
    track = Track(detection1, dt=0.1)
    initial_confidence = track.confidence
    
    # Update with new detection
    detection2 = FusedTarget(
        x=1.1, y=1.1, confidence=0.9,
        sensor_ids=[1, 2], timestamp=0.1
    )
    
    state = track.update(detection2)
    
    # Check updates
    assert track.hits == 2
    assert track.misses == 0
    assert track.confidence > initial_confidence
    
    # Position should be influenced by new measurement
    pos = track.get_position()
    assert 1.0 <= pos[0] <= 1.1
    assert 1.0 <= pos[1] <= 1.1
    
    print("✓ Track update test passed")


def test_track_status_transitions():
    """Test track status transitions."""
    detection = FusedTarget(
        x=1.0, y=1.0, confidence=0.8,
        sensor_ids=[1, 2], timestamp=0.0
    )
    
    track = Track(detection, dt=0.1)
    assert track.status == TrackStatus.NEW
    
    # Add more detections to confirm track
    for i in range(3):
        detection = FusedTarget(
            x=1.0 + i*0.1, y=1.0 + i*0.1, confidence=0.8,
            sensor_ids=[1, 2], timestamp=i*0.1
        )
        track.update(detection)
    
    assert track.status == TrackStatus.CONFIRMED
    
    # Test misses
    for i in range(3):
        track.miss()
    
    assert track.status == TrackStatus.LOST
    
    print("✓ Track status transitions test passed")


def test_mahalanobis_distance():
    """Test Mahalanobis distance calculation."""
    detection = FusedTarget(
        x=0.0, y=0.0, confidence=0.8,
        sensor_ids=[1, 2], timestamp=0.0
    )
    
    track = Track(detection, dt=0.1)
    
    # Distance to same position should be small
    same_detection = FusedTarget(
        x=0.0, y=0.0, confidence=0.8,
        sensor_ids=[1, 2], timestamp=0.1
    )
    distance_same = track.mahalanobis_distance(same_detection)
    
    # Distance to far position should be larger
    far_detection = FusedTarget(
        x=5.0, y=5.0, confidence=0.8,
        sensor_ids=[1, 2], timestamp=0.1
    )
    distance_far = track.mahalanobis_distance(far_detection)
    
    assert distance_far > distance_same
    
    print("✓ Mahalanobis distance test passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running AI Tracker Unit Tests...")
    print("=" * 40)
    
    test_kalman_initialization()
    test_kalman_prediction()
    test_kalman_update()
    test_track_creation()
    test_track_update()
    test_track_status_transitions()
    test_mahalanobis_distance()
    
    print("=" * 40)
    print("All AI Tracker tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()