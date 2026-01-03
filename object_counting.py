import cv2
import numpy as np
from ultralytics import YOLO


capture = cv2.VideoCapture('data/sample_video.mp4')
model = YOLO('yolov8n.pt')

unique_ids = set() # track ids

while True:
    ret, frame = capture.read()
    if not ret:
        break
    results = model.track(frame, classes=[0], persist=True) # 0 -- people class

    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_ids.add(oid) # new item in the id, add to our set
        cv2.putText(annotated_frame, f"Count: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' pressed, break
            break
capture.release()
cv2.destroyAllWindows()

