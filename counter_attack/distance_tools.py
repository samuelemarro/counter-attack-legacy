import copy
import logging

from abc import ABC, abstractmethod

import foolbox
import numpy as np

from . import batch_attack, utils

logger = logging.getLogger(__name__)

class DistanceMeasure(ABC):
    @abstractmethod
    def compute(self, x, y, batch, bounds):
        pass
    

class LpDistanceMeasure(DistanceMeasure):
    def __init__(self, p, mean):
        if p < 0:
            raise ValueError('p must be positive or zero')

        if np.isposinf(p) and mean:
            raise ValueError('Averaging is not supported for L-inf.')
            
        self.p = p
        self.mean = mean

    def compute(self, x, y, batch, bounds):
        # If x or y are empty, we return an empty array
        empty_x = hasattr(x, '__len__') and len(x) == 0
        empty_y = hasattr(y, '__len__') and len(y) == 0
        if batch and (empty_x or empty_y):
            return np.array([], dtype=np.float)


        def single_image(diff):
            if bounds is not None:
                _min, _max = bounds
                assert _max > _min

                bound_normalization = _max - _min

                diff = diff / bound_normalization

            # L_infinity: Maximum difference
            if np.isinf(self.p):
                value = np.max(np.abs(diff))
            # L_0: Count of different values
            elif self.p == 0:
                value = len(np.nonzero(np.reshape(diff, -1))[0])
            # L_p: p-root of the sum of diff^p
            else:
                value = np.power(np.sum(np.power(np.abs(diff), self.p)), 1 / self.p)

            if self.mean:
                value = value / diff.size
            
            return value

        if batch:
            return np.array([single_image(_x - _y) for _x, _y in zip(x, y)])
        else:
            return single_image(x - y)

    def __str__(self):
        _name = ''

        if self.mean:
            _name += 'Mean '

        _name += 'L-{} Bound-normalized Distance Measure'.format(self.p)

        return _name

class DistanceTool:
    def __init__(self, name):
        self.name = name

    def get_distance(self, image):
        raise NotImplementedError()

    def get_distances(self, images):
        raise NotImplementedError()


class AdversarialDistance(DistanceTool):
    """
    Finds the distance using an adversarial attack.
    Returns a failure value if it can't find an adversarial
    sample
    """

    def __init__(self,
                 foolbox_model: foolbox.models.Model,
                 attack: foolbox.attacks.Attack,
                 distance_measure : DistanceMeasure,
                 failure_value: np.float,
                 cuda: bool,
                 num_workers: int = 50,
                 name: str = None):
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.distance_measure = distance_measure
        self.failure_value = failure_value
        self.cuda = cuda
        self.num_workers = num_workers

        if name is None:
            name = attack.name()

        self.name = name

    def get_distance(self, image):
        predictions = self.foolbox_model.predictions(image)
        logger.debug('Requested single distance estimate.')

        label = np.argmax(predictions)

        attack = copy.copy(self.attack)
        attack._default_model = self.foolbox_model
        adversarial = attack(image, label)

        if adversarial is None:
            return self.failure_value

        distance = self.distance_measure.compute(adversarial, image, False, self.foolbox_model.bounds())

        logger.debug('Distance : {}'.format(distance))

        return distance

    def get_distances(self, images):
        logger.debug('Requested {} distance estimates'.format(len(images)))

        batch_predictions = self.foolbox_model.batch_predictions(images)
        labels = np.argmax(batch_predictions, axis=1)

        adversarials, _, _ = batch_attack.get_adversarials(self.foolbox_model,
                                                           images,
                                                           labels,
                                                           self.attack,
                                                           False,
                                                           self.cuda,
                                                           num_workers=self.num_workers)

        assert len(adversarials) == len(images)

        distances = [self.failure_value] * len(images)

        # Fill in the distances computed by the attack, while leaving the failed attacks with failure_value

        successful_adversarial_indices = [i for i in range(
            len(images)) if adversarials[i] is not None]

        # If there are no successful adversarial samples, return early
        if len(successful_adversarial_indices) == 0:
            return distances

        successful_adversarial_indices = np.array(
            successful_adversarial_indices)

        successful_adversarials = [
            adversarials[i] for i in successful_adversarial_indices]
        successful_images = [images[i]
                             for i in successful_adversarial_indices]
        successful_adversarials = np.array(successful_adversarials)
        successful_images = np.array(successful_images)

        successful_distances = self.distance_measure.compute(successful_adversarials, successful_images, True, self.foolbox_model.bounds())

        for i, original_index in enumerate(successful_adversarial_indices):
            distances[original_index] = successful_distances[i]

        logger.debug('Distances: {}'.format(distances))

        return distances

class FoolboxDistance(foolbox.distances.Distance):
    """Foolbox-compatible distance measure

    If you're wondering what's going on: foolbox attacks accept
    a distance type instead of an actual instance. So, if you want
    to use the MSE distance, you have to pass foolbox.distances.MSE.
    foolbox then calls the type and builds the distance (e.g. MSE(...)).
    This usually works well, but it means that we can't pass any other
    arguments to the distance like the distance measure. We therefore
    use the following wrapper trick: we init and pass the LpDistance instance;
    foolbox will attempt to create the distance by calling distance(...).
    However, since it's an instance, calls are handled by __call__.
    In __call__, we init WrappedLpDistance with the provided arguments
    (in addition to distance_measure) and return it.
    """
    class WrappedDistance(foolbox.distances.Distance):
        def __init__(self, distance_measure, reference=None, other=None, bounds=None, value=None):
            self.distance_measure = distance_measure
            super().__init__(reference, other, bounds, value)

        def _calculate(self):
            assert self.other.shape == self.reference.shape
            value = self.distance_measure.compute(self.other, self.reference, False, self._bounds)

            gradient = None
            return value, gradient

        @property
        def gradient(self):
            raise NotImplementedError

        def name(self):
            return 'Wrapped Distance ({})'.format(self.distance_measure)

    def __init__(self, distance_measure):
        self.distance_measure = distance_measure

    def __call__(self,
                 reference=None,
                 other=None,
                 bounds=None,
                 value=None):
        return FoolboxDistance.WrappedDistance(self.distance_measure, reference, other, bounds, value)

    def _calculate(self):
        raise NotImplementedError()
