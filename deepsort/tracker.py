from deepsort.track import Track
from deepsort.cost_matrix import compute_cost_matrix
from scipy.optimize import linear_sum_assignment

class DeepSORT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, features):
        """Update tracker with new detections and features."""
        if len(self.tracks) == 0:
            for i, (det, feat) in enumerate(zip(detections, features)):
                self.tracks.append(Track(self.next_id, det, feat))
                self.next_id += 1
            return

        # Step 1: Predict next positions
        for track in self.tracks:
            track.predict()

        # Step 2: Compute cost matrix (IoU + cosine similarity)
        cost_matrix = compute_cost_matrix(detections, features, self.tracks)

        # Step 3: Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Step 4: Update matched tracks
        for d, t in zip(row_ind, col_ind):
            self.tracks[t].update(detections[d], features[d])

        # Step 5: Increment time_since_update for unmatched tracks
        matched_tracks = set(col_ind)
        for track in self.tracks:
            if track.id not in matched_tracks:
                track.time_since_update += 1

        # Step 5.5: Remove lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < 10]

        # Step 6: Add new tracks for unmatched detections
        unmatched_detections = set(range(len(detections))) - set(row_ind)
        for idx in unmatched_detections:
            self.tracks.append(Track(self.next_id, detections[idx], features[idx]))
            self.next_id += 1
