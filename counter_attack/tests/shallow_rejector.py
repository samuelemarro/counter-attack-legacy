import logging

import foolbox
import numpy as np

from counter_attack import batch_attack, distance_measures
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def shallow_rejector_test(standard_model: foolbox.models.Model,
                          loader,
                          attack,
                          lp_distance : distance_measures.LpDistance,
                          rejector,
                          cuda: bool,
                          num_workers: int = 50,
                          name: str = 'Shallow Rejector Attack'):
    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []

    for images, labels in test_utils.get_iterator(name, logger, loader):
        samples_count += len(images)

        # First step: Remove the samples misclassified by the standard model
        correct_images, correct_labels = batch_attack.get_correct_samples(
            standard_model, images, labels)

        # Second step: Remove samples that are wrongfully detected as adversarial
        correct_images, correct_labels = batch_attack.get_approved_samples(
            standard_model, correct_images, correct_labels, rejector)

        correct_count += len(correct_images)
        images, labels = correct_images, correct_labels

        # Third step: Generate adversarial samples against the standard model (removing failed adversarials)
        assert len(images) == len(labels)
        adversarials, images, labels = batch_attack.get_adversarials(
            standard_model, images, labels, attack, True, cuda, num_workers)

        # Fourth step: Remove adversarial samples that are detected as such
        batch_valid = rejector.batch_valid(adversarials)
        adversarials = adversarials[batch_valid]
        images = images[batch_valid]
        labels = labels[batch_valid]

        successful_attack_count += len(adversarials)

        # Fifth step: Compute the distances
        batch_distances = lp_distance.compute(images, adversarials, True, standard_model.bounds())
        distances += list(batch_distances)

        accuracy = correct_count / samples_count
        success_rate = successful_attack_count / correct_count
        logger.debug('Accuracy: {:2.2f}%'.format(accuracy * 100.0))
        logger.debug('Success Rate: {:2.2f}%'.format(success_rate * 100.0))

    assert samples_count >= correct_count
    assert correct_count >= successful_attack_count
    assert len(distances) == successful_attack_count

    return samples_count, correct_count, successful_attack_count, np.array(distances)