from deepsort.kalman_filter import KalmanFilter

class Track:
    def __init__(self, track_id, bbox, feature):
        self.id = track_id
        self.kf = KalmanFilter()
        self.state, self.covariance = self.kf.initiate(bbox)
        self.features = [feature]
        self.time_since_update = 0
        self.hit_streak = 0

    def predict(self):
        self.state, self.covariance = self.kf.predict(self.state, self.covariance)

    def update(self, bbox, feature):
        print(f"Before update: {self.state[:4]}")
        self.state, self.covariance = self.kf.update(self.state, self.covariance, bbox)
        print(f"After update: {self.state[:4]}")
        self.features.append(feature)
        self.time_since_update = 0
        self.hit_streak += 1