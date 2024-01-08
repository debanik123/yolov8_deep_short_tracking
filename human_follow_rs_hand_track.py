import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
import mediapipe as mp

class RealSenseYoloHandTracker:
    def __init__(self, yolo_weights_path='weights/yolov8n.pt'):
        self.model = YOLO(yolo_weights_path)
        self.tracker = Tracker()
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Initialize MediaPipe Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.ox_human_bbx_2d_rr = 300
        self.img_ox = 50
        self.img_oy = 10
        self.num_point_obs = 25
    
    def obstracle_layer(self, frame):
        Y, X, _ = frame.shape

        Ix = X // 2
        Iy = Y //2

        human_bbx_2d_AA = (Ix - self.ox_human_bbx_2d_rr, self.img_ox)
        human_bbx_2d_BB = (Ix + self.ox_human_bbx_2d_rr, self.img_oy)
        human_bbx_2d_CC = (Ix + self.ox_human_bbx_2d_rr, Y-self.img_oy)

        cv2.rectangle(frame, human_bbx_2d_AA, human_bbx_2d_CC, (0, 0, 255), 2)

        diff_bbx = np.abs(human_bbx_2d_AA[0] - human_bbx_2d_BB[0]) // self.num_point_obs
        if(diff_bbx > 0):
            for i in np.arange(human_bbx_2d_AA[0], human_bbx_2d_CC[0] + diff_bbx, diff_bbx):
                for j in np.arange(human_bbx_2d_AA[1], human_bbx_2d_CC[1] + diff_bbx, diff_bbx):
                    x_int = int(round(i))
                    y_int = int(round(j))
                    cv2.circle(frame, (x_int, y_int), radius=1, color=(0, 255, 0), thickness=-1)




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

            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())

            # YOLO Hand Tracking
            yolo_results = self.model.predict(frame, classes=[0])
            yolo_bboxes, yolo_classes, yolo_scores = self.tracker.bbx_utils(yolo_results)
            self.tracker.update(frame, yolo_bboxes, yolo_scores, yolo_classes)

            # MediaPipe Hand Tracking
            finger_count = self.hand_tracking(frame)
            cv2.putText(frame, f"Finger Count: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.obstracle_layer(frame)

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
