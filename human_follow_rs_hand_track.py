import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
from pcl_utils import Pcl_utils
import mediapipe as mp

class RealSenseYoloHandTracker:
    def __init__(self, yolo_weights_path='weights/yolov8n-pose.pt'):
        self.model = YOLO(yolo_weights_path)
        self.tracker = Tracker()
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.hand_distance_th = 1.0
        self.track_id_ = None

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

    def follow_target(self, target_track_ID):
        # Add your logic to follow the target based on the track ID
        print(f"Following target with Track ID: {target_track_ID}")
    
    def count_fingers(self, hand_landmarks):
        fingertips = [8, 12, 16, 20]
        count = sum(1 for fingertip in fingertips if hand_landmarks.landmark[fingertip].y < hand_landmarks.landmark[fingertip - 2].y)
        return count
        
    def hand_tracking(self, frame, depth):
        # MediaPipe Hand Tracking
        h, w, _ = frame.shape
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.draw_hand_rectangle(frame, hand_landmarks, depth)
                finger_count = self.count_fingers(hand_landmarks)
                return finger_count
        else:
            return 0.0
    
    def draw_hand_rectangle(self, frame, landmarks, depth):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


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

            yolo_bboxes, yolo_scores, yolo_classes =self.tracker.bbx_utils(yolo_results)
            self.tracks_ = self.tracker.update(frame, yolo_bboxes, yolo_scores, yolo_classes)

            for track in self.tracks_:
                self.tracker.draw_bbx(track, frame)
                bbox = track.to_tlbr()
                x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                hand_tracking_frame = frame[y_min:y_max, x_min:x_max]
                finger_count = self.hand_tracking(hand_tracking_frame, depth_frame)
            


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
