import numpy as np
import foolbox
import OffenseDefense.distance_tools as distance_tools

class Detector:
    def get_score(self, image):
        raise NotImplementedError()
    def get_scores(self, images):
        raise NotImplementedError()

class DistanceDetector(Detector):
    def __init__(self,
                 distance_tool : distance_tools.DistanceTool):
        self.distance_tool = distance_tool

    def get_score(self, image):
        return self.distance_tool.get_distance(image)

    def get_scores(self, images):
        return self.distance_tool.get_distances(images)
