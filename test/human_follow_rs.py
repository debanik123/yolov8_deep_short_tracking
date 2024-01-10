import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker

class RealSenseYoloTracker:
    def __init__(self, yolo_weights_path='weights/yolov8n.pt'):
        self.model = YOLO(yolo_weights_path)
        self.tracker = Tracker()

        # Check for the presence of a RealSense camera
        self.check_camera_connection()

        # Configure Intel RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")

    def start(self):
        # Start streaming
        self.pipeline.start(self.config)

    def run(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = self.model.predict(frame, classes=[0])

            bboxes, classes, scores = self.tracker.bbx_utils(results)

            self.tracker.update(frame, bboxes, scores, classes)

            cv2.imshow("YOLOv8 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def stop(self):
        # Stop streaming
        self.pipeline.stop()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RealSenseYoloTracker()

    try:
        tracker.start()
        tracker.run()
    except RuntimeError as e:
        print(e)
    finally:
        tracker.stop()
