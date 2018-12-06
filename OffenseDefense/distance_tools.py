import foolbox
import numpy as np
from . import batch_attack, batch_processing, utils


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
                 p: np.float,
                 failure_value: np.float,
                 batch_worker: batch_processing.BatchWorker = None,
                 num_workers: int = 50,
                 name: str = None):
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.p = p
        self.failure_value = failure_value
        self.batch_worker = batch_worker
        self.num_workers = num_workers

        if name is None:
            name = attack.name()

        self.name = name

    def get_distance(self, image):
        predictions = self.foolbox_model.predictions(image)
        label = np.argmax(predictions)

        adversarial = self.attack(image, label)

        if adversarial is None:
            return self.failure_value

        distance = utils.lp_distance(adversarial, image, self.p, False)
        return distance

    def get_distances(self, images):
        batch_predictions = self.foolbox_model.batch_predictions(images)
        labels = np.argmax(batch_predictions, axis=1)

        adversarial_filter = batch_attack.get_adversarials(self.foolbox_model,
                                                           images,
                                                           labels,
                                                           self.attack,
                                                           False,
                                                           False,
                                                           batch_worker=self.batch_worker,
                                                           num_workers=self.num_workers)

        assert len(adversarial_filter['adversarials']) == len(images)

        distances = [self.failure_value] * len(images)

        # Fill in the distances computed by the attack, while leaving the failed attacks with failure_value

        successful_adversarial_indices = [i for i in range(
            len(images)) if adversarial_filter['adversarials'][i] is not None]

        # If there are no successful adversarial samples, return early
        if len(successful_adversarial_indices) == 0:
            return distances

        successful_adversarial_indices = np.array(
            successful_adversarial_indices)

        successful_adversarials = [
            adversarial_filter['adversarials'][i] for i in successful_adversarial_indices]
        successful_images = [adversarial_filter['images'][i]
                             for i in successful_adversarial_indices]
        successful_adversarials = np.array(successful_adversarials)
        successful_images = np.array(successful_images)

        successful_distances = utils.lp_distance(
            successful_adversarials, successful_images, self.p, True)

        for i, original_index in enumerate(successful_adversarial_indices):
            distances[original_index] = successful_distances[i]

        return distances


"""
Returns the unnormalized L_p distance.

If you're wondering what's going on: foolbox attacks accept
a distance type instead of an actual object. So, if you want
to use the MSE distance, you have to pass foolbox.distances.MSE.
foolbox then calls the type and builds the distance (e.g. MSE(...)).
This usually works well, but it means that we can't pass any other
arguments to the distance. We therefore use this wrapper trick to 
pass the argument 'p': we init and pass the LpDistance object.
foolbox will attempt to create the distance by calling distance(...)
However, since it's an instance, calls to the class are handled by
__call__. In __call__, we init WrappedLpDistance with the provided
arguments (in addition to p) and return it.
"""


class LpDistance(foolbox.distances.Distance):
    class WrappedLpDistance(foolbox.distances.Distance):
        def __init__(self, p, reference=None, other=None, bounds=None, value=None):
            self.p = p
            super().__init__(reference, other, bounds, value)

        def _calculate(self):
            value = utils.lp_distance(
                self.other, self.reference, self.p, False)
            gradient = None
            return value, gradient

        @property
        def gradient(self):
            raise NotImplementedError

        def name(self):
            return 'L{} Distance'.format(self.p)

    def __init__(self, p):
        self.p = p

    def __call__(self,
                 reference=None,
                 other=None,
                 bounds=None,
                 value=None):
        return LpDistance.WrappedLpDistance(self.p, reference, other, bounds, value)

    def _calculate(self):
        raise NotImplementedError()
