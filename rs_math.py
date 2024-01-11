import numpy as np
import pyrealsense2 as rs
import math
import cv2

class VelocityGenerator:
    def __init__(self, desired_distance=1.0, vel_max=1.5, speed=0.95):
        self.desired_distance = desired_distance
        self.vel_max = vel_max
        self.speed = speed
        self.follow_distance_th = 0.6
        self.current_distance = 0.0
        self.linear_x = 0.0
        self.angular_z = 0.0

    def dis_fun(self, pcd):
        return math.hypot(pcd[0], pcd[1])  # x, y

    def cosine(self, a, b, c):
        if a == 0.0 or b == 0.0:
            return None
        else:
            return math.acos((a**2 + b**2 - c**2) / (2 * a * b))

    def gamma_sign_correction(self, gamma, y):
        if y < 0:
            gamma = -abs(gamma)
        if y > 0:
            gamma = abs(gamma)
        return gamma

    def find_gpxy(self, refe_point_pcd, target_point_pcd):
        dis_refe_pcd = self.dis_fun(refe_point_pcd)
        dis_targ_pcd = self.dis_fun(target_point_pcd)
        dis_refe_targ_pcd = self.dis_fun([refe_point_pcd[0] - target_point_pcd[0], refe_point_pcd[1] - target_point_pcd[1]])

        gamma = self.cosine(dis_refe_pcd, dis_targ_pcd, dis_refe_targ_pcd)

        if gamma is not None:
            gamma_corr = self.gamma_sign_correction(gamma, target_point_pcd[1])
            gpx = dis_targ_pcd * math.cos(gamma_corr)
            gpy = dis_targ_pcd * math.sin(gamma_corr)
            current_distance = math.hypot(gpx, gpy)
            if current_distance>self.follow_distance_th:
                error = abs(self.desired_distance - current_distance)
                l_v = error * self.speed
                self.linear_x = min(l_v, self.vel_max)
                self.angular_z = math.atan2(gpy, gpx)
                self.current_distance = current_distance
            else:
                self.linear_x = 0.0
                self.angular_z = 0.0


    def generate_velocity(self, refe_point, target_point):
        self.find_gpxy(refe_point, target_point)
        return self.linear_x, self.angular_z, self.current_distance




class PixeltoPcl:
    def __init__(self, depth_frame):
        self.depth_frame = depth_frame
        self.intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

    def convert_pixel_to_distance(self, x, y):
        upixel = np.array([float(x), float(y)], dtype=np.float32)
        distance = self.depth_frame.get_distance(x, y)
        return distance

    def convert_pixel_to_3d_world(self, x, y):
        upixel = np.array([float(x), float(y)], dtype=np.float32)
        distance = self.depth_frame.get_distance(x, y)
        pcd = rs.rs2_deproject_pixel_to_point(self.intrinsics, upixel, distance)
        return pcd[2], -pcd[0], -pcd[1]  # ros coordinates X, Y, Z


class PixelToVelocityGenerator_rs:
    def __init__(self, depth_frame, desired_distance=1.5, vel_max=1.2, speed=0.85):
        self.pixel_to_pcl_converter = PixeltoPcl(depth_frame)
        self.velocity_generator = VelocityGenerator(desired_distance, vel_max, speed)

    def generate_velocity_from_pixels(self, refe_point_pixel, target_point_pixel):
        # Convert pixels to 3D world coordinates
        refe_point_pcd = self.pixel_to_pcl_converter.convert_pixel_to_3d_world(*refe_point_pixel)
        target_point_pcd = self.pixel_to_pcl_converter.convert_pixel_to_3d_world(*target_point_pixel)

        # Generate velocity using VelocityGenerator
        linear_velocity, angular_velocity, current_distance = self.velocity_generator.generate_velocity(refe_point_pcd, target_point_pcd)

        return linear_velocity, angular_velocity, current_distance


class Keypoints:
    def __init__(self, frame, hip, shoulder, elbow):
        self.hip = hip
        self.shoulder = shoulder
        self.elbow = elbow

        kps = [hip, shoulder, elbow]
        self.draw_kp(frame, kps)
        # self.distance()
    
    def draw_kp(self, frame, kps):
        for kp in kps:
            x = int(kp[0])
            y = int(kp[1])
            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def distance(self):
        distance_hip_shoulder = math.sqrt((self.hip[0]-self.shoulder[0])**2 + (self.hip[1]-self.shoulder[1])**2)
        distance_hip_elbow = math.sqrt((self.hip[0]-self.elbow[0])**2 + (self.hip[1]-self.elbow[1])**2)
        distance_shoulder_elbow = math.sqrt((self.shoulder[0]-self.elbow[0])**2 + (self.shoulder[1]-self.elbow[1])**2)

        gamma = self.cosine(distance_hip_shoulder, distance_shoulder_elbow, distance_hip_elbow)
        if gamma is not None:
            angle = math.degrees(gamma)
            return angle
        else:
            return None

    
    def cosine(self, a, b, c):
        if a == 0.0 or b == 0.0:
            return None
        else:
            return math.acos((a**2 + b**2 - c**2) / (2 * a * b))
        

