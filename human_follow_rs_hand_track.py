import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
from pcl_utils import Pcl_utils
import mediapipe as mp

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
import math


class RealSenseFollowme(Node):
    def __init__(self, yolo_weights_path='weights/yolov8n.pt'):
        super().__init__('real_sense_follow_me')
        self.model = YOLO(yolo_weights_path)
        self.tracker = Tracker()
        self.check_camera_connection()
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        self.hand_distance_th = 0.9
        self.track_id_ = None
        self.isFollowing = False
        self.unique_id = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.pcl_uts = Pcl_utils()
    
    

    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found. Connect a RealSense camera and try again.")

    def start_pipeline(self):
        self.pipe.start(self.config)

    def stop_pipeline(self):
        self.pipe.stop()

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
                x_mid, y_mid = self.draw_hand_rectangle(frame, hand_landmarks, depth)
                finger_count = self.count_fingers(hand_landmarks)
                cv2.putText(frame, str(finger_count),(int(x_mid), int(y_mid-11)),0, 1.0, (255,255,255),1, lineType=cv2.LINE_AA)
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

        
        x_mid = (x_min + x_max) //2
        y_mid = (y_min + y_max) //2

        cv2.circle(frame, (x_mid, y_mid), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return x_mid, y_mid
    
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
    
    def calculate_angle(self, ref_point, target_midpoint):
        x1, y1 = ref_point
        x2, y2 = target_midpoint
        
        delta_x = x2 - x1
        delta_y = y2 - y1

        angle_rad = math.atan2(delta_y, delta_x)
        # angle_deg = math.degrees(angle_rad)
        return angle_rad
        
    def run(self):
        while True:
            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            

            if depth_frame:
                self.pcl_uts.obstracle_layer(depth_frame, frame)
            
            track_ids = []
            if self.unique_id not in track_ids:
                self.isFollowing = False
                # self.stop_robot()

            try:
                yolo_results = self.model.predict(frame, classes=[0])

                yolo_bboxes, yolo_scores, yolo_classes =self.tracker.bbx_utils(yolo_results, depth_frame)

                self.tracks_ = self.tracker.update(frame, yolo_bboxes, yolo_scores, yolo_classes)
                

                
                for track in self.tracks_:
                    track_ids.append(track.track_id)
                    self.tracker.draw_bbx(track, frame)
                    bbox = track.to_tlbr()
                    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    x_mid = (x_min + x_max) //2
                    y_mid = (y_min + y_max) //2
                    cv2.circle(frame, (x_mid, y_mid), radius=5, color=(0, 255, 255), thickness=-1)
                    human_distance = self.pcl_uts.convert_pixel_to_distance(depth_frame, x_mid, y_mid)
                    human_distance_str = "{:.3f}".format(human_distance)
                    cv2.putText(frame, human_distance_str,(int(x_mid), int(y_mid-50)),0, 1.0, (255,255,255),1, lineType=cv2.LINE_AA)

                    if human_distance<self.hand_distance_th:
                        hand_tracking_frame = frame[y_min:y_max, x_min:x_max]
                        finger_count = self.hand_tracking(hand_tracking_frame, depth_frame)
                        if finger_count == 2 and not self.isFollowing:
                            self.unique_id = track.track_id
                        if finger_count == 3 and self.unique_id == track.track_id:
                            self.unique_id = None
                            self.isFollowing = False

                    if self.unique_id is not None and self.unique_id == track.track_id:
                        self.isFollowing = True
                        hm_midpoint = (int(x_mid), int(y_mid // 2.0))
                        im_midpoint = (int(frame.shape[1] // 2.0), int(frame.shape[0] // 2.0))

                        cv2.circle(frame, im_midpoint, radius=5, color=(0, 255, 255), thickness=-1)
                        cv2.putText(frame, "follow_"+str(self.unique_id),(int(x_mid), int(y_mid-11)),0, 1.0, (0,255,0),1, lineType=cv2.LINE_AA)

                        # angle_rad = self.calculate_angle(im_midpoint, hm_midpoint)
                        # cv2.putText(frame, str(angle_rad),(int(x_mid), int(y_mid+50)),0, 1.0, (255,255,255),1, lineType=cv2.LINE_AA)

                        self.pcl_uts.target_gp(depth_frame, im_midpoint, hm_midpoint)

                        # print(self.pcl_uts.linear_x, self.pcl_uts.angular_z)
                        linear_x_str = "{:.3f}".format(self.pcl_uts.linear_x)
                        angular_z_str = "{:.3f}".format(self.pcl_uts.angular_z)
                        self.cmd_vel(self.pcl_uts.linear_x, self.pcl_uts.angular_z)
                        cv2.putText(frame, "linear_x: "+ linear_x_str +" angular_z: " + angular_z_str ,(int(x_mid), int(y_mid+50)),0, 1.0, (255,255,255),1, lineType=cv2.LINE_AA)

                        
                        

                    elif self.unique_id is None:
                        # stop the robot
                        self.isFollowing = False
                        self.stop_robot()

            except Exception as e:
                # handle the exception and print information
                print(f"Error: {e}")
                # pass
            
            rclpy.spin_once(self, timeout_sec=0.0000001)

            


            cv2.imshow("YOLOv8 and MediaPipe Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main(args=None):
    rclpy.init(args=args)
    real_sense_followme = RealSenseFollowme()
    try:
        real_sense_followme.start_pipeline()
        real_sense_followme.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        real_sense_followme.stop_pipeline()
        real_sense_followme.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
