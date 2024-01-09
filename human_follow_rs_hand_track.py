import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
from pcl_utils import Pcl_utils

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

            self.tracker.keypoints_utils(frame, yolo_results)

            if self.tracker.target_track_ID is not None:
                for track in self.tracker.tracks_:
                    if track.track_id == self.tracker.target_track_ID:
                        self.tracker.draw_bbx(track, frame)
                


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
