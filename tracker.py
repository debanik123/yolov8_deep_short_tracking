from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepSortTracker
from tracking_helpers import read_class_names, create_box_encoder
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    def update(self, frame, bboxes, scores, classes):
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            track_id = track.track_id
            print("Tracker ID: {}, Class: {}".format(track_id, class_name))

            color = self.colors[int(track.track_id) % len(self.colors)]  # draw bbox on screen
            color = [i * 255 for i in color]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)
    
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
            bboxes.append(box)
            classes.append(label)
            scores.append(score)
        return (bboxes, classes, scores)

