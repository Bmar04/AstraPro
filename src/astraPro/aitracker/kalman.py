"""
Kalman filter implementation for 2D target tracking.
"""

import numpy as np
import time
from typing import Tuple, Optional


class KalmanFilter:
    """
    2D Kalman filter for tracking targets with constant velocity model.
    
    State vector: [x, y, vx, vy]
    - x, y: position (meters)
    - vx, vy: velocity (m/s)
    """
    
    def __init__(self, dt: float = 0.15, process_noise: float = 2.0, measurement_noise: float = 0.3):
        """
        Initialize Kalman filter.
        
        Args:
            dt: Time step between updates (seconds)
            process_noise: Process noise standard deviation (m/s²)
            measurement_noise: Measurement noise standard deviation (meters)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State vector [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State covariance matrix
        self.P = np.eye(4) * 1000.0  # High initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance matrix
        q = process_noise ** 2
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        # Measurement noise covariance matrix
        r = measurement_noise ** 2
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        self.last_update = time.time()
    
    def predict(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state.
        
        Args:
            dt: Time step (uses default if None)
            
        Returns:
            Predicted state and covariance
        """
        if dt is not None:
            # Update transition matrix for different time step
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Update process noise for different time step
            q = self.process_noise ** 2
            Q = np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]) * q
        else:
            F = self.F
            Q = self.Q
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
        return self.state.copy(), self.P.copy()
    
    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the filter with a measurement.
        
        Args:
            measurement: [x, y] position measurement
            
        Returns:
            Updated state and covariance
        """
        # Innovation (residual)
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.last_update = time.time()
        
        return self.state.copy(), self.P.copy()
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return float(self.state[0]), float(self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return float(self.state[2]), float(self.state[3])
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (standard deviation)."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
    
    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty (standard deviation)."""
        return float(np.sqrt(self.P[2, 2] + self.P[3, 3]))
    
    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance to measurement.
        
        Args:
            measurement: [x, y] position measurement
            
        Returns:
            Mahalanobis distance
        """
        # Predicted measurement
        y_pred = self.H @ self.state
        
        # Innovation
        y = measurement - y_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance
        distance = np.sqrt(y.T @ np.linalg.inv(S) @ y)
        
        return float(distance)
    
    def initialize_from_measurement(self, measurement: np.ndarray, velocity: Optional[Tuple[float, float]] = None):
        """
        Initialize filter state from first measurement.
        
        Args:
            measurement: [x, y] position measurement
            velocity: Optional initial velocity estimate
        """
        self.state[0] = measurement[0]  # x
        self.state[1] = measurement[1]  # y
        
        if velocity is not None:
            self.state[2] = velocity[0]  # vx
            self.state[3] = velocity[1]  # vy
        else:
            self.state[2] = 0.0  # vx
            self.state[3] = 0.0  # vy
        
        # Reset covariance with reasonable initial uncertainty
        self.P = np.diag([0.5**2, 0.5**2, 2.0**2, 2.0**2])  # pos: 0.5m, vel: 2.0 m/s
        
        self.last_update = time.time()