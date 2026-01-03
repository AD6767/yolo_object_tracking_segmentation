import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')
capture = cv2.VideoCapture('data/sample_video.mp4')

while True:
    ret, frame = capture.read()
    if not ret:
        break

    results = model.track(source=frame, classes=[0], persist=True) # track people

    for r in results:
        annotated_frame = frame.copy()
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy() # contour
            boxes = r.boxes.xyxy.numpy() # boxes
            ids = r.boxes.id.numpy() # ids

            for i, mask in enumerate(masks): # inside masks there are several points
                person_id = ids[i]
                x1, y1, x2, y2 = boxes[i].astype(int) # bbox
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0])) # resize mask to image frame
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"ID: {person_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' pressed, break
        break
capture.release()
cv2.destroyAllWindows()
