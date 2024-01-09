from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepSortTracker
from tracking_helpers import read_class_names, create_box_encoder
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

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
        self.tracking_activate = False
        self.tracking_index = None


    def update(self, frame, bboxes, scores, classes):
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

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


    def keypoints_utils(self, frame, results):
        keypoints_tensor = results[0].keypoints.data.tolist()
        idx = 0
        for kps in keypoints_tensor:
            right_kps = Keypoints(frame, kps[12],kps[6], kps[8])
            left_kps = Keypoints(frame, kps[11],kps[5], kps[7])

            try:
                right_angle = right_kps.distance()
                left_angle = left_kps.distance()

                # print("Right angle between hip, shoulder, and elbow:", right_angle, idx)
                # print("Left angle between hip, shoulder, and elbow:", left_angle, idx)

                if right_angle is not None and right_angle > 70 and right_angle < 95:
                    self.tracking_activate = True
                    self.tracking_index = idx


                if left_angle is not None and left_angle > 70 and left_angle < 95:
                    self.tracking_activate = False
                    self.tracking_index = None

            except:
                pass
            
            idx +=1



            

        # for res in results[0].keypoints.data.tolist():
        #     print(res)
    
    def bbx_utils(self, results):
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
            # print("box ---> ",box)
            if(score > self.detection_threshold):
                bboxes.append(box)
                classes.append(label)
                scores.append(score)
        return (bboxes, classes, scores)
    
    def bbx_util(self, yolo_results, track_idx):
        bboxes = []
        classes = []
        scores = []
        class_names = ['person']

        res = yolo_results[0].boxes.data.tolist()[track_idx]
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
        # print("box ---> ",box)
        if(score > self.detection_threshold):
            bboxes.append(box)
            classes.append(label)
            scores.append(score)
        return (bboxes, classes, scores)

