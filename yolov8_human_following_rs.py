import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

import rclpy
from rclpy.node import Node

class YOLOv8TrackingNode(Node):
    def __init__(self):
        super().__init__('yolov8_tracking_node')
        self.model = YOLO('yolov8n-pose.pt')
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        
    
    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")

    def process_frames(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return

            frame = np.asanyarray(color_frame.get_data())

            results = self.model.track(frame, persist=True, classes=[0], conf=0.60, iou=0.7, max_det=10)
            keypoints_tensor = results[0].keypoints.data.tolist()
            boxes_tensor = results[0].boxes.data.tolist()

            for res in boxes_tensor:
                x1 = int(res[0])
                y1 = int(res[1])
                x2 = int(res[2])
                y2 = int(res[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            rclpy.spin_once(self, timeout_sec=0.0000001)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    yolov8_tracking_node = YOLOv8TrackingNode()

    try:
        rclpy.spin(yolov8_tracking_node)
    except KeyboardInterrupt:
        pass

    yolov8_tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
