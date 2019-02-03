import numpy as np
import foolbox
from . import distance_tools


class Detector:
    """Base class for detectors. Each detector
    outputs a "trustworthiness score" for each input.
    """

    def get_score(self, image) -> float:
        raise NotImplementedError()

    def get_scores(self, images) -> np.ndarray:
        raise NotImplementedError()


class DistanceDetector(Detector):
    def __init__(self,
                 distance_tool: distance_tools.DistanceTool):
        self.distance_tool = distance_tool

    def get_score(self, image):
        return self.distance_tool.get_distance(image)

    def get_scores(self, images):
        return self.distance_tool.get_distances(images)
