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

    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")

    def start(self):
        # Start streaming
        self.pipeline.start(self.config)

    def hand_tracking(self, frame):
        # MediaPipe Hand Tracking
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks for fingers
                finger_landmarks = [hand_landmarks.landmark[i] for i in range(1, 21) if i % 4 == 0]

                # Detect fingers based on Y-coordinate of finger landmarks
                num_fingers_up = sum(1 for landmark in finger_landmarks if landmark.y < finger_landmarks[0].y)

                # Draw hand landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # Connect all hand landmarks with lines
                num_landmarks = len(hand_landmarks.landmark)
                connections = [(i, i + 1) for i in range(num_landmarks - 1)]
                for connection in connections:
                    start_point = (int(hand_landmarks.landmark[connection[0]].x * w), int(hand_landmarks.landmark[connection[0]].y * h))
                    end_point = (int(hand_landmarks.landmark[connection[1]].x * w), int(hand_landmarks.landmark[connection[1]].y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                # Display the number of fingers
                cv2.putText(frame, f'Fingers: {num_fingers_up}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# ... (rest of the class methods remain unchanged)



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
            self.hand_tracking(frame)

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
