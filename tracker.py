from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepSortTracker
from tracking_helpers import read_class_names, create_box_encoder
import numpy as np

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = 'mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, bboxes, scores, classes):
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
        self.tracker.predict()
        self.tracker.update(detections)

