import numpy as np
import foolbox
import OffenseDefense.distance_tools as distance_tools

class Detector:
    def get_score(self, image):
        pass
    def get_scores(self, images):
        pass

class DistanceDetector(Detector):
    def __init__(self,
                 foolbox_model : foolbox.models.Model,
                 distance_tool : distance_tools.DistanceTool,
                 p : np.float):
        self.foolbox_model = foolbox_model
        self.distance_tool = distance_tool
        self.p = p

    def get_score(self, image):
        label = self.foolbox_model.predictions(image)
        return self.distance_tool.get_distance(image, label, self.p)

    def get_scores(self, images):
        labels = self.foolbox_model.batch_predictions(images)
        return self.distance_tool.get_distances(images, labels, self.p)
