import math
# import rclpy
# from geometry_msgs.msg import Twist
# from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs

class Pcl_utils():
    def __init__(self):
        self.ox_human_bbx_2d_rr = 200
        self.img_ox = 100
        self.img_oy = 100
        self.num_point_obs = 10
        self.obstracle_distance_th = 1.0
        self.stop_flag = False

    def convert_pixel_to_distance(self, depth, x, y):
        upixel = np.array([float(x), float(y)], dtype=np.float32)
        distance = depth.get_distance(x, y)
        return distance

    def convert_pixel_to_3d_world(self, depth, x, y):
        upixel = np.array([float(x), float(y)], dtype=np.float32)
        distance = depth.get_distance(x, y)
        intrinsics = depth.get_profile().as_video_stream_profile().get_intrinsics()
        pcd = rs.rs2_deproject_pixel_to_point(intrinsics, upixel, distance)
        return pcd[2], -pcd[0], -pcd[1] # ros coordinates X,Y,Z
    
    def dis_fun(self, pcd):
        return math.hypot(pcd[0],pcd[1]) # x,y
    
    def cosine(self, a, b, c):
        if a == 0.0 or b == 0.0:
            return float('nan')
        else:
            return math.acos((a**2 + b**2 - c**2) / (2 * a * b))

    def gamma_sign_correction(self, gamma, y):
        if y < 0:
            gamma = -abs(gamma)
        if y > 0:
            gamma = abs(gamma)
        return gamma

    def target_gp(self, depth_frame, refe_point, target_point):
        refe_point_pcd = self.convert_pixel_to_3d_world(depth_frame, refe_point[0], refe_point[1])
        target_point_pcd = self.convert_pixel_to_3d_world(depth_frame, target_point[0], target_point[1])
        self.find_gpxy(refe_point_pcd, target_point_pcd)
    
    def find_gpxy(self, refe_point_pcd, target_point_pcd):
        dis_refe_pcd = self.dis_fun(refe_point_pcd)
        dis_targ_pcd = self.dis_fun(target_point_pcd)
        dis_refe_targ_pcd = self.dis_fun([refe_point_pcd[0]-target_point_pcd[0], refe_point_pcd[1]-target_point_pcd[1]])

        gamma = self.cosine(dis_refe_pcd, dis_targ_pcd, dis_refe_targ_pcd)
        gamma_corr = self.gamma_sign_correction(gamma, target_point_pcd[1])

        gpx = dis_targ_pcd*math.cos(gamma_corr)
        gpy = dis_targ_pcd*math.sin(gamma_corr)

        self.vel_gen(gpx, gpy)
        
    def vel_gen(self, x, y):
        if math.isfinite(x) and math.isfinite(y):
            currentDistance = math.hypot(x,y)
            error = abs(self.desiredDistance - currentDistance)
            l_v = error * self.speed_
            linear_x = min(l_v, self.vel_max)
            angular_z = math.atan2(y, x)
            # self.cmd_vel(linear_x, angular_z)
            # print("linearX --> ", linear_x, "angilar_z ---> ", angular_z)
            return linear_x, angular_z
        else:
            linear_x = 0.0
            angular_z = 0.0
            return linear_x, angular_z

    def calculate_pix_distance(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx**2 + dy**2)
    
    def obstracle_layer(self, depth, frame):
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
                    try:
                        obstracle_distance = self.convert_pixel_to_distance(depth, x_int, y_int)
                        obstracle_pcd = self.convert_pixel_to_3d_world(depth, x_int, y_int)
                        print("obstracle_distance ---> ",obstracle_distance)
                        
                        if(obstracle_distance < self.obstracle_distance_th):
                            self.stop_flag = True
                            

                    except Exception as e:
                        print(i,j,f"Error: {e}")
