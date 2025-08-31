import cv2
from ultralytics import YOLO

from detection import results, annotated_frame

video = cv2.VideoCapture("Cars Moving On Road Stock Footage - Free Download.mp4")
model = YOLO("yolov8s.pt")

# Initializing tracking
tracker = model.track
car_count = 0
tracked_ids = set()
gate_y = 240
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_number = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("End of video or error reading frame")
        break
    frame_number += 1

    try:
         results = tracker(frame,classes =[2],conf=0.5,iou=0.5,persist=True)
    except Exception as e:
        print(f"Error during tracking at frame {frame_number}: {e}")
        continue

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is None:
                print(f"Warning: No track ID for detection at frame {frame_number}")
                continue

            try:

               track_id = int(box.id)
            except Exception as e:
                print(f"Error converting track ID at frame {frame_number}")
                continue
            xywh = box.xywh[0]
            center_y = xywh[1].item()
            if abs(center_y - gate_y) < 10 and track_id not in tracked_ids:
                car_count += 1
                tracked_ids.add(track_id)
                print(f"Car counted at frame {frame_number}, ID: {track_id}, Total: {car_count}")

            cv2.line(frame,(0,gate_y),(frame_width,gate_y),(0,0,255),2)

            cv2.putText(frame, f"Car Count:{car_count}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            annotated_frame = r.plot()
            cv2.imshow('Car Count', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video.release()
cv2.destroyAllWindows()
print(f"Total Cars Counted: {car_count}")