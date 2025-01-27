import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class ObjectTracker:
    def __init__(self):
        self.trackers = []
        self.objects = []

    def update_tracker(self, frame, detections):
        # Create new trackers for detections
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            self.trackers.append(tracker)
            self.objects.append(cls)

    def update(self, frame):
        # Update existing trackers
        for i, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                x1, y1, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, str(self.objects[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify a video file path
    tracker = ObjectTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        # Initialize or update trackers
        if len(tracker.trackers) == 0:
            tracker.update_tracker(frame, detections)
        else:
            tracker.update(frame)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
