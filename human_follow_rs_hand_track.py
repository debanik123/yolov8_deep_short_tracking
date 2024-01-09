import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
import mediapipe as mp
from pcl_utils import Pcl_utils

class RealSenseYoloHandTracker:
    def __init__(self, yolo_weights_path='weights/yolov8n.pt'):
        self.model = YOLO(yolo_weights_path)
        self.tracker = Tracker()
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # Initialize MediaPipe Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        
        self.pcl_uts = Pcl_utils()
    
    

    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")

    def start(self):
        # Start streaming
        self.pipeline.start(self.config)

    def count_fingers(self, hand_landmarks):
        fingertips = [8, 12, 16, 20]
        count = sum(1 for fingertip in fingertips if hand_landmarks.landmark[fingertip].y < hand_landmarks.landmark[fingertip - 2].y)
        return count

    def hand_tracking(self, frame):
        # MediaPipe Hand Tracking
        h, w, _ = frame.shape
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if results.multi_handedness[0].classification[0].label == 'Left':
                    finger_count = self.count_fingers(hand_landmarks)
                    return finger_count
                else:
                    return 0
        else:
            return 0



    def run(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            if depth_frame:
                self.pcl_uts.obstracle_layer(depth_frame, frame)

            # YOLO Hand Tracking
            yolo_results = self.model.predict(frame, classes=[0])
            yolo_bboxes, yolo_classes, yolo_scores = self.tracker.bbx_utils(yolo_results)
            tracks = self.tracker.update(frame, yolo_bboxes, yolo_scores, yolo_classes)

            tracks_size = len(tracks)
            # print("Number of tracks:", tracks_size)

            for track in tracks:
                try:
                    bbox = track.to_tlbr()
                    track_id = track.track_id
                    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    hand_tracking_frame = frame[y_min:y_max, x_min:x_max]
                    finger_count = self.hand_tracking(hand_tracking_frame)
                    print("Tracker ID: {}, Finger_count : {}".format(track_id, finger_count))

                except Exception as e:
                        print(f"Error: {e}")

                # cv2.putText(frame, f"Finger Count: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

            cv2.imshow("YOLOv8 and MediaPipe Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RealSenseYoloHandTracker()

    try:
        tracker.start()
        tracker.run()
    except RuntimeError as e:
        print(e)
    finally:
        tracker.stop()
