import os
import random

import cv2
from ultralytics import YOLO

# from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

model = YOLO("yolov8n.pt")


tracker = DeepSort(max_age=30, embedder='mobilenet')
# tracker = Tracker()
# tracker = DeepSortTracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:
    ret, frame = cap.read()
    results = model.predict(frame, classes=[0])

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append(([x1, y1, x2-x1, y2-y1], score, 0))

            

        # tracker.update(detections)
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[1 % len(colors)]), 3)
            cv2.putText(frame, str(track_id),(int(x1), int(y1-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)

            print("track_id ---> ", track_id)
        # for track in tracker.tracks:
        #     bbox = track.bbox
        #     x1, y1, x2, y2 = bbox
        #     track_id = track.track_id
    
    cv2.imshow("YOLOv8 and MediaPipe Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

cap.release()
cv2.destroyAllWindows()