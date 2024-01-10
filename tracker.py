from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepSortTracker
from tracking_helpers import read_class_names, create_box_encoder
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pcl_utils import Pcl_utils

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
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

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
        
    

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = './weights/mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = create_box_encoder(encoder_model_filename, batch_size=1)

        cmap = plt.get_cmap('tab20b') #initialize color map
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        self.detection_threshold = 0.80
        self.pcl_uts = Pcl_utils()
        self.human_distance_th=6.0
        

    def update(self, frame, bboxes, scores, classes):
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # self.draw_bbx(track, frame)

        return self.tracker.tracks
    
    def draw_bbx(self, track, frame):
        bbox = track.to_tlbr()
        class_name = track.get_class()
        track_id = track.track_id

        color = self.colors[int(track.track_id) % len(self.colors)]  # draw bbox on screen
        color = [i * 255 for i in color]

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)
   
    def bbx_utils(self, results, depth_frame):
        bboxes = []
        classes = []
        scores = []
        class_names = ['person']
        for res in results[0].boxes.data.tolist():
            # print(res)
            x1 = int(res[0])
            y1 = int(res[1])
            x2 = int(res[2])
            y2 = int(res[3])
            score = res[4]
            class_id = int(res[5])

            label = class_names[class_id]
            # cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)

            w = (x2-x1)
            h = (y2-y1)

            box = [x1,y1,w,h]

            x_mid = (x1 + x2) //2
            y_mid = (y1 + y1) //2
            human_distance = self.pcl_uts.convert_pixel_to_distance(depth_frame, x_mid, y_mid)

            # print("box ---> ",box)
            if(human_distance < self.human_distance_th and score > self.detection_threshold):
                bboxes.append(box)
                classes.append(label)
                scores.append(score)

        return (bboxes, scores, classes)
    