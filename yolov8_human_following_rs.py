import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

import rclpy
from rclpy.node import Node

from rs_math import PixelToVelocityGenerator_rs

class YOLOv8TrackingNode(Node):
    def __init__(self):
        super().__init__('yolov8_tracking_node')
        self.model = YOLO('yolov8n-pose.pt')
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
            depth_frame = frames.get_depth_frame()
            
            if not color_frame:
                return

            if not depth_frame:
                return

            frame = np.asanyarray(color_frame.get_data())
            pvg_rs = PixelToVelocityGenerator_rs(depth_frame)

            im_midpoint = (int(frame.shape[1] // 2.0), int(frame.shape[0] // 2.0))
            cv2.circle(frame, im_midpoint, radius=5, color=(0, 255, 255), thickness=-1)

            results = self.model.track(frame, persist=True, classes=[0], conf=0.60, iou=0.7, max_det=10)
            for result in results:
                keypoints_tensor = result.keypoints.data.tolist()
                boxes_tensor = result.boxes.data.tolist()
                
                for res in boxes_tensor:
                    x1 = int(res[0])
                    y1 = int(res[1])
                    x2 = int(res[2])
                    y2 = int(res[3])
                    hm_midpoint = (int((x1+x2) // 2.0), int((y1+y2) // 2.0))
                    cv2.circle(frame, hm_midpoint, radius=5, color=(255, 0, 255), thickness=-1)
                    cv2.line(frame, im_midpoint, hm_midpoint, color=(255, 255, 0), thickness=2)

                    linear_velocity, angular_velocity = pvg_rs.generate_velocity_from_pixels(im_midpoint, hm_midpoint)
                    print("Linear Velocity:", linear_velocity, "Angular Velocity:", angular_velocity)

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
    try:
        yolov8_tracking_node = YOLOv8TrackingNode()
        yolov8_tracking_node.process_frames()
    except KeyboardInterrupt:
        pass
    
    finally:
        yolov8_tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
