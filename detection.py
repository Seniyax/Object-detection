import cv2
import supervision as sv

from ultralytics import YOLO

video = cv2.VideoCapture("Cars Moving On Road Stock Footage - Free Download.mp4")

model = YOLO("yolov8s.pt")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()





