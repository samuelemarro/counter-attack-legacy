import logging
from typing import Tuple

import foolbox
import numpy as np

from counter_attack import batch_attack, distance_measures, loaders, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def attack_test(foolbox_model: foolbox.models.Model,
                loader: loaders.Loader,
                attack: foolbox.attacks.Attack,
                lp_distance : distance_measures.LpDistance,
                cuda: bool,
                num_workers: int = 50,
                save_adversarials: bool = False,
                name: str = 'Attack Test') -> Tuple[float, np.ndarray]:

    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []
    adversarials = [] if save_adversarials else None
    adversarial_ground_truths = [] if save_adversarials else None

    for images, labels in test_utils.get_iterator(name, logger, loader):
        samples_count += len(images)
        logger.debug('Received {} images'.format(len(images)))

        # Remove misclassified samples
        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        correct_count += len(correct_images)
        logger.debug('{} correct images ({} removed)'.format(
            correct_count, samples_count - correct_count))

        successful_adversarials, successful_images, successful_labels = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True, cuda, num_workers)

        successful_attack_count += len(successful_adversarials)
        logger.debug('{} successful attacks ({} removed)'.format(
            successful_attack_count, correct_count - successful_attack_count))

        # Update the distances and/or the adversarials (if there are successful adversarials)
        if len(successful_adversarials) > 0:
            distances += list(lp_distance.compute(
                successful_adversarials, successful_images, True, foolbox_model.bounds()))

            if save_adversarials:
                adversarials += list(successful_adversarials)
                adversarial_ground_truths += list(successful_labels)

        failure_count = correct_count - successful_attack_count

        average_distance, median_distance, adjusted_median_distance = utils.distance_statistics(
            distances, failure_count)

        logger.debug('Average Distance: {:2.2e}'.format(average_distance))
        logger.debug('Median Distance: {:2.2e}'.format(median_distance))
        logger.debug('Success Rate: {:2.2f}%'.format(
            successful_attack_count / float(correct_count) * 100.0))
        logger.debug('Adjusted Median Distance: {:2.2e}'.format(
            adjusted_median_distance))

        logger.debug('\n============\n')

    if save_adversarials:
        adversarials = np.array(adversarials)
        adversarial_ground_truths = np.array(adversarial_ground_truths)

    assert samples_count >= correct_count
    assert correct_count >= successful_attack_count
    assert len(distances) == successful_attack_count

    if adversarials is not None:
        assert len(distances) == len(adversarials)
        assert len(adversarials) == len(adversarial_ground_truths)

    return samples_count, correct_count, successful_attack_count, distances, adversarials, adversarial_ground_truths