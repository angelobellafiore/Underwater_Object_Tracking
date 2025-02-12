import numpy as np

class KalmanFilter:
    def __init__(self):
        """Initialize the Kalman filter for tracking bounding boxes"""
        self.dt = 1.0

        # State transition matrix (position + velocity)
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0],  # x
            [0, 1, 0, 0, 0, self.dt],  # y
            [0, 0, 1, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0],  # h
            [0, 0, 0, 0, 1, 0],  # vx
            [0, 0, 0, 0, 0, 1]  # vy
        ])

        # Observation matrix (we observe only position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0]  # h
        ])

        # Process noise covariance
        self.Q = np.eye(6) * 0.01

        # Measurement noise covariance
        self.R = np.eye(4) * 0.1

    def initiate(self, measurement):
        """Initialize the Kalman filter with first measurement (bounding box)."""
        x, y, w, h = measurement
        state = np.array([x, y, w, h, 0, 0])
        covariance = np.eye(6) * 1
        return state, covariance

    def predict(self, state, covariance):
        """Predict the next state and covariance."""
        state = np.dot(self.F, state)
        covariance = np.dot(self.F, np.dot(covariance, self.F.T)) + self.Q
        return state, covariance

    def update(self, state, covariance, measurement):
        """Update the state using new measurements."""
        y = measurement - np.dot(self.H, state)
        S = np.dot(self.H, np.dot(covariance, self.H.T)) + self.R
        K = np.dot(covariance, np.dot(self.H.T, np.linalg.inv(S)))

        state = state + np.dot(K, y)
        covariance = np.dot(np.eye(6) - np.dot(K, self.H), covariance)
        return state, covariance
