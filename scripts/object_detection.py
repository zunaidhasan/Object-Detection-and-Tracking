import cv2
import torch
import numpy as np

# Load the YOLOv5 model (choose version and load a pretrained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(img)

    # Parse results
    detections = results.xyxy[0].numpy()  # Bounding boxes
    return detections

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify a video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        for *box, conf, cls in detections:
            # Draw bounding boxes and labels
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(box[0]), int(box[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
