import logging

import foolbox
import numpy as np

from . import utils

logger = logging.getLogger(__name__)


class Rejector:
    def valid(self, image):
        raise NotImplementedError()

    def batch_valid(self, image):
        raise NotImplementedError()


class DetectorRejector(Rejector):
    def __init__(self, detector, threshold):
        self.detector = detector
        self.threshold = threshold

    def valid(self, image):
        detector_score = self.detector.get_score(image)

        return detector_score >= self.threshold

    def batch_valid(self, images):
        detector_scores = np.array(self.detector.get_scores(images))

        return detector_scores >= self.threshold

#TODO: Check the cache

class CacheRejector(Rejector):
    def __init__(self, distance_tool, threshold, distance_measure, bounds, cache_size):
        self.distance_tool = distance_tool
        self.threshold = threshold
        self.distance_measure = distance_measure
        self.bounds = bounds
        self.cache_size = cache_size

        self.cache = []

    def valid(self, image):
        """import matplotlib.pyplot as plt
        plt.title('Initial')
        plt.imshow(np.transpose(image, [1, 2, 0]))
        plt.show()"""

        for cache_image, cache_distance in self.cache:
            image_distance = self.distance_measure.compute(image, cache_image, False, self.bounds)

            """plt.title(image_distance)
            plt.imshow(np.transpose(cache_image, [1, 2, 0]))
            plt.show()
            logger.debug('Cache distance: {}'.format(cache_distance))
            logger.debug('Distance from cache: {}'.format(image_distance))"""

            # If the cache is not valid and the image is close enough to the cache, reject
            if image_distance < self.threshold - cache_distance and cache_distance < self.threshold:
                logger.debug('Auto-reject')
                return False
            # We cannot be sure that a valid cache is actually valid

        # If no cache image was able to automatically reject or accept, evaluate normally

        logger.debug('Evaluating')
        distance = self.distance_tool.get_distance(image)

        # Add the current image to the cache (if it is not infinite)
        if not np.isinf(distance):
            self.cache.insert(0, (image, distance))

        # If the cache is too big, remove the last image (which is the oldest)
        if len(self.cache) > self.cache_size:
            del self.cache[-1]

        return distance >= self.threshold

    def batch_valid(self, images):
        auto_accept_indices = []
        auto_reject_indices = []

        for i, image in enumerate(images):
            for cache_image, cache_distance in self.cache:
                image_distance = self.distance_measure.compute(image, cache_image, False, self.bounds)

                # If the cache is not valid and the image is close enough to the cache, reject
                if image_distance < self.threshold - cache_distance and cache_distance < self.threshold:
                    auto_reject_indices.append(i)
                    break
                # We cannot be sure that a valid cache is actually valid

        # Initialize the responses to False
        responses = np.zeros([len(images)], dtype=np.bool)

        # Set the auto accept
        auto_accept_indices = np.array(auto_accept_indices, dtype=np.int32)
        responses[auto_accept_indices] = True

        # We don't set the auto rejects because they are already False

        # Set the evaluated indices
        evaluate_indices = np.array([i for i in range(
            len(images)) if i not in auto_accept_indices and i not in auto_reject_indices])

        if len(evaluate_indices) > 0:
            evaluate_images = images[evaluate_indices]
            evaluate_distances = np.array(
                self.distance_tool.get_distances(evaluate_images))
            evaluate_responses = evaluate_distances >= self.threshold

            responses[evaluate_indices] = evaluate_responses

            # Add the samples to the cache, excluding infinities

            finite_images, finite_distances = utils.filter_lists(
                lambda _, distance: not np.isinf(distance), evaluate_images, evaluate_distances)

            assert len(finite_images) == len(finite_distances)

            self.cache.extend(zip(finite_images, finite_distances))

            # If the cache is too big, remove the first ones
            # Note: If len(evaluate_indices) > self.cache_size, some samples never
            # end up in the cache
            self.cache = self.cache[:self.cache_size]

        return responses


class RejectorModel(foolbox.models.Model):
    def __init__(self, classifier, rejector):
        super().__init__(classifier.bounds(),
                         classifier.channel_axis(), classifier._preprocessing)
        self.classifier = classifier
        self.rejector = rejector

    def num_classes(self):
        return self.classifier.num_classes() + 1

    def predictions(self, image):
        valid = self.rejector.valid(image)

        if valid:
            # Return [y0, y1, y2... yN, -inf]
            classifier_predictions = self.classifier.predictions(image)

            assert not np.any(np.isneginf(classifier_predictions))
            output = np.append(classifier_predictions,
                               [-np.Infinity], axis=0)

            return output
        else:
            # Return [0, 0, 0... 0, 1]
            output = np.zeros(self.classifier.num_classes())
            output[-1] = 1.
            return output

    def batch_predictions(self, images):
        batch_valid = self.rejector.batch_valid(images)

        valid_indices = np.where(batch_valid)[0]
        rejected_indices = np.where(batch_valid)[0]

        outputs = np.zeros([len(images), self.num_classes()])

        # Rejected samples are assigned [0, 0, 0... 0, 1]
        if len(rejected_indices) > 0:
            rejected_output = np.zeros(self.num_classes())
            rejected_output[-1] = 1.
            outputs[rejected_indices] = rejected_output

        # Valid samples are assigned [y0, y1, y2... yN, -inf]
        if len(valid_indices) > 0:
            batch_classifier_predictions = self.classifier.batch_predictions(
                images[valid_indices])
            assert not np.any(np.isneginf(batch_classifier_predictions))
            valid_outputs = np.insert(batch_classifier_predictions,
                                      batch_classifier_predictions.shape[1], -np.Infinity, axis=1)

            outputs[valid_indices] = valid_outputs

        return outputs


class Unrejected(foolbox.criteria.Criterion):
    def is_adversarial(self, predictions, label):
        top_label = np.argmax(predictions)
        detected_label = len(predictions) - 1
        assert label != detected_label

        return top_label != detected_label
