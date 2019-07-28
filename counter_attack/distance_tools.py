import copy
import logging

import foolbox
import numpy as np

from . import batch_attack, utils

logger = logging.getLogger(__name__)



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
                 p : float,
                 failure_value: np.float,
                 cuda: bool,
                 num_workers: int = 50,
                 name: str = None):
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.p = p
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

        distance = utils.lp_distance(adversarial, image, self.p, False)

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

        successful_distances = utils.lp_distance(successful_adversarials, successful_images, self.p, True)

        for i, original_index in enumerate(successful_adversarial_indices):
            distances[original_index] = successful_distances[i]

        logger.debug('Distances: {}'.format(distances))

        return distances
