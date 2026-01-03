import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


capture = cv2.VideoCapture('data/sample_video_2.mp4')
model = YOLO('yolov8n.pt')

id_map = {}
next_id = 1
trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    results = model.track(frame, classes=[0], persist=True) # 0 -- people class
    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.numpy() # boxes coord
        ids = results[0].boxes.id.numpy() # boxes ids

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # center of boxes
            # display the trail for past 20 frames only
            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map: # person has been appearing for more than 5 times and id is new
                id_map[oid] = next_id # then add person to current map
                next_id += 1 # create a new id

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw bbox
                cv2.putText(annotated_frame, f"ID: {sid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1) # draw center point
                # draw trail
                trail_points = list(trail[oid])
                for i in range(1, len(trail_points)):
                    cv2.line(annotated_frame, trail_points[i - 1], trail_points[i], (0, 0, 255), 2)
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' pressed, break
        break
capture.release()
cv2.destroyAllWindows()
            
