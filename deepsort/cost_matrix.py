import numpy as np
from deepsort.intersect_over_union import get_iou
from deepsort.cosine_similarity import cosine_similarity

def compute_cost_matrix(detections, features, tracklets):
    """Compute the cost matrix using cosine distance (1 - similarity)."""
    cost_matrix = np.zeros((len(detections), len(tracklets)))
    for i, det_bbox in enumerate(detections):
        det_feat = features[i]

        for j, track in enumerate(tracklets):
            track_bbox = track.state[:4]
            iou_score = get_iou(det_bbox, track_bbox)
            cosine_dist = 1 - cosine_similarity(det_feat, track.features[-1])

            alpha = 0.5
            beta = 0.5

            cost_matrix[i, j] = alpha * cosine_dist + beta * (1 - iou_score)

    return cost_matrix
