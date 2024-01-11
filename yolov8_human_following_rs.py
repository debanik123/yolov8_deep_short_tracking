import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


from rs_math import PixelToVelocityGenerator_rs, Keypoints, PixeltoPcl

class YOLOv8TrackingNode(Node):
    def __init__(self):
        super().__init__('yolov8_tracking_node')
        self.model = YOLO('yolov8n-pose.pt')
        self.check_camera_connection()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(self.config)

        self.unique_id = None
        self.track_id_ = None
        self.isFollowing = False

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
    
    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")
    
    def stop_robot(self):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel_msg)
    
    def cmd_vel(self, l_v, a_v):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = l_v
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = a_v
        self.cmd_vel_pub.publish(cmd_vel_msg)
    
    def cluster_create(self, frame, x,y, depth_frame, window_size=15, color_=(255, 0, 255)):
        hm_distances = []
        pixels_min = []
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                current_x = x + i
                current_y = y + j
                pf = PixeltoPcl(depth_frame)
                hm_dis = pf.convert_pixel_to_distance(current_x, current_y)
                if hm_dis:
                    cv2.circle(frame, (current_x, current_y), radius=2, color=color_, thickness=-1)
                    hm_distances.append(hm_dis)
                    pixels_min.append((current_x, current_y))

        
        min_index, min_distance = min(enumerate(hm_distances), key=lambda x: x[1])
        cv2.circle(frame, pixels_min[min_index], radius=5, color=(0,0,255), thickness=-1)
        min_x, min_y = pixels_min[min_index]

        return (min_x, min_y), min_distance



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
            # cv2.circle(frame, im_midpoint, radius=5, color=(0, 255, 255), thickness=-1)
            try:
                results = self.model.track(frame, persist=True, classes=[0], conf=0.60, iou=0.7, max_det=5)
                
                keypoints_tensor = results[0].keypoints.data.tolist()
                boxes_tensor = results[0].boxes.data.tolist()
                ids_tensor = results[0].boxes.id.tolist()

                # print(ids_tensor)

                if self.unique_id not in ids_tensor:
                    self.isFollowing = False
                    # self.stop_robot()

                for bbx, id in zip(boxes_tensor, ids_tensor):
                    if self.unique_id == id:
                        self.isFollowing = True
                        # print(bbx, id)
                        x1 = int(bbx[0])
                        y1 = int(bbx[1])
                        x2 = int(bbx[2])
                        y2 = int(bbx[3])

                        x_mid = int((x1+x2) // 2.0)
                        y_mid = int((y1+y2) // 2.0)
                        hm_midpoint = (x_mid, y_mid)
                        cv2.putText(frame, "Follow : "+str(self.unique_id),hm_midpoint,0, 1, (0,255,255),1, lineType=cv2.LINE_AA)
                        # cv2.circle(frame, hm_midpoint, radius=5, color=(255, 0, 255), thickness=-1)
                        cv2.line(frame, im_midpoint, hm_midpoint, color=(255, 255, 0), thickness=2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        min_hm_pix, min_hm_distance = self.cluster_create(frame, hm_midpoint[0], hm_midpoint[1], depth_frame, color_=(255, 0, 255))
                        min_img_pix, min_Im_distance = self.cluster_create(frame, im_midpoint[0], im_midpoint[1], depth_frame, color_=(0, 255, 255))

                        cv2.putText(frame, str(min_hm_distance) ,(min_hm_pix[0], min_hm_pix[1]-100),0, 0.5, (255,255,255),1, lineType=cv2.LINE_AA)
                        cv2.putText(frame, str(min_Im_distance) ,(min_img_pix[0], min_img_pix[1]+100),0, 0.5, (255,255,255),1, lineType=cv2.LINE_AA)

                        linear_velocity, angular_velocity = pvg_rs.generate_velocity_from_pixels(min_img_pix, min_hm_pix)
                        print("Linear Velocity:", linear_velocity, "Angular Velocity:", angular_velocity)

                        self.cmd_vel(linear_velocity, angular_velocity)
                        
                        linear_x_str = "{:.3f}".format(linear_velocity)
                        angular_z_str = "{:.3f}".format(angular_velocity)
                        cv2.putText(frame, "linear_x: "+ linear_x_str +" angular_z: " + angular_z_str ,(x_mid, y_mid+50),0, 1.0, (255,255,255),1, lineType=cv2.LINE_AA)


                    # elif self.unique_id is None:
                        # self.stop_robot()
                        
                    

            except:
                pass

            # print(results)
            for idx, kps in enumerate(keypoints_tensor):

                try:
                    right_kps = Keypoints(frame, kps[12],kps[6], kps[8])
                    left_kps = Keypoints(frame, kps[11],kps[5], kps[7])

                    right_angle = right_kps.distance()
                    left_angle = left_kps.distance()

                    print("Right angle between hip, shoulder, and elbow:", right_angle, idx)
                    print("Left angle between hip, shoulder, and elbow:", left_angle, idx)

                    if right_angle is not None and right_angle > 70 and right_angle < 95 and not self.isFollowing:
                        self.unique_id = ids_tensor[idx]

                    if left_angle is not None and left_angle > 70 and left_angle < 95 and self.unique_id == ids_tensor[idx]:
                        self.unique_id = None
                        self.isFollowing = False
                except:
                    pass

                
            
            # annotated_frame = results[0].plot()
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)
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
