import cv2
from ultralytics import YOLO
from tracker import Tracker


# Load the YOLOv8 model
model = YOLO('weights/yolov8n.pt')
tracker = Tracker()
# Open the video file
# video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0)
# detection_threshold = 0.90

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, classes=[0])
        
        bboxes, classes, scores = tracker.bbx_utils(results)

        tracker.update(frame, bboxes, scores, classes)

        cv2.imshow("YOLOv8 Tracking", frame)
        # break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()