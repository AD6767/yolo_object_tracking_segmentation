import cv2
from ultralytics import YOLO


capture = cv2.VideoCapture('data/sample_video.mp4')
model = YOLO('yolov8n.pt')

while True:
    ret, frame = capture.read()
    if not ret:
        break
    # results = model(frame) # detect all objects
    results = model(frame, classes=[0]) # detect only people
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' pressed, break
        break
capture.release()
cv2.destroyAllWindows()
