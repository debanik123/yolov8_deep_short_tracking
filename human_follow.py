import cv2
from ultralytics import YOLO
from tracker import Tracker


# Load the YOLOv8 model
model = YOLO('weights/yolov8n.pt')
tracker = Tracker()
# Open the video file
# video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0)
detection_threshold = 0.7
class_names = ['person']

def bbx_utils(results):
    bboxes = []
    classes = []
    scores = []
    for res in results[0].boxes.data.tolist():
        # print(res)
        x1 = int(res[0])
        y1 = int(res[1])
        x2 = int(res[2])
        y2 = int(res[3])
        score = res[4]
        class_id = int(res[5])

        label = class_names[class_id]
        # cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)

        w = (x2-x1)
        h = (y2-y1)

        box = [x1,y1,w,h]
        print("box ---> ",box)
        bboxes.append(box)
        classes.append(label)
        scores.append(score)
    return (bboxes, classes, scores)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, classes=[0])
        
        bboxes, classes, scores = bbx_utils(results)

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