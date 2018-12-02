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


class CompositeDetectorModel(foolbox.models.Model):
    def __init__(self, classifier, detector, threshold, undetected_value=-np.Infinity):
        super().__init__(classifier.bounds(),
                         classifier.channel_axis(), classifier._preprocessing)
        self.classifier = classifier
        self.detector = detector
        self.threshold = threshold
        self.undetected_value = undetected_value

    def num_classes(self):
        return self.classifier.num_classes() + 1

    def predictions(self, image):
        detector_score = self.detector.get_score(image)

        if detector_score >= self.threshold:
            # Return [y0, y1, y2... yN, undetected_value]
            classifier_predictions = self.classifier.predictions(image)

            assert np.all(classifier_predictions > self.undetected_value)
            output = np.append(classifier_predictions,
                               [self.undetected_value], axis=0)

            return output
        else:
            # Return [0, 0, 0... 0, 1]
            output = np.zeros(self.classifier.num_classes())
            output[-1] = 1.
            return output

    def batch_predictions(self, images):
        detector_scores = np.array(self.detector.get_scores(images))

        valid_indices = np.where(detector_scores >= self.threshold)[0]
        rejected_indices = np.where(detector_scores < self.threshold)[0]

        outputs = np.zeros([len(images), self.num_classes()])

        # Rejected samples are assigned [0, 0, 0... 0, 1]
        if len(rejected_indices) > 0:
            rejected_output = np.zeros(self.num_classes())
            rejected_output[-1] = 1.
            outputs[rejected_indices] = rejected_output

        # Valid samples are assigned [y0, y1, y2... yN, undetected_value]
        if len(valid_indices) > 0:
            batch_classifier_predictions = self.classifier.batch_predictions(
                images[valid_indices])
            assert np.all(batch_classifier_predictions > self.undetected_value)
            valid_outputs = np.insert(batch_classifier_predictions,
                                      batch_classifier_predictions.shape[1], self.undetected_value, axis=1)
            # print(outputs)
            # print(valid_indices)
            # print(valid_outputs)
            outputs[valid_indices] = valid_outputs

        return outputs


class Undetected(foolbox.criteria.Criterion):
    def is_adversarial(self, predictions, label):
        top_label = np.argmax(predictions)
        detected_label = len(predictions) - 1
        assert label != detected_label

        return top_label != detected_label
