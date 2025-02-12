import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from deepsort.tracker import DeepSORT
from deepsort.feature_extractor import (simple_features_extractor,
                                        resnet50_features_extractor,
                                        sift_features_extractor)


def track_objects():
    model_path = 'model/best.pt'
    detector = YOLO(model_path)

    # Initialize DeepSORT tracker
    tracker = DeepSORT()

    track_history = defaultdict(lambda: [])  # Track history for visualization

    # Open video input
    cap = cv2.VideoCapture("/dataset/ArmyDiver1.mp4")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the output video file and codec
    output_file = "/results/TrackedArmyDiver1.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Object Detection
        results = detector.predict(frame)
        detections, features = [], []

        for result in results:
            # Extract the class label for each detection
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  #class indices
            class_names = result.names  #class names list

            for box, class_id in zip(result.boxes.xywh.cpu().numpy(), class_ids):
                detections.append(box)
                # Change 'simple_features_extractor' function with:
                # - 'resnet50_features_extractor', if you use ResNet-50 as extractor
                # - 'sift_features_extractor', if you use SIFT as extractor
                # Next, go to feature_extractor.py and uncomment the required lines.
                features.append(simple_features_extractor(frame, box))

                # Get the class name from the class ID
                class_name = class_names[class_id]

                # Draw the class name on the frame (along with bounding box)
                x_center, y_center, w, h = box
                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)

                # Add text with class name
                cv2.putText(frame, class_name, (int(x_min + (w*0.2)), int(y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

        # Step 2: Update DeepSORT Tracker
        tracker.update(detections, features)

        # Step 3: Draw Tracking Results
        for track in tracker.tracks:

            color = colors[track.id % len(colors)]

            x_center, y_center, w, h = track.state[:4]
            tracked = track_history[track.id]
            tracked.append((float(x_center), float(y_center)))
            if len(tracked) > 40:
                tracked.pop(0)

            if len(tracked) > 1:
                points = np.array(tracked, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=3)
            """For the visualization we have to convert the bounding box
            from YOLOv8 format [x_center, y_center, width, height] to tlbr format [x_min, y_min, x_max, y_max]"""
            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color,2)
            cv2.putText(frame, f"ID {track.id}", (int(x_min), int(y_min)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)

        out.write(frame)
        
        cv2.imshow("Detecting video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
