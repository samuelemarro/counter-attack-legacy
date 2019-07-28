import logging

import foolbox
import numpy as np

from counter_attack import batch_attack, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def shallow_defense_test(standard_model: foolbox.models.Model,
                       loader,
                       attack,
                       p : float,
                       defended_model: foolbox.models.Model,
                       cuda: bool,
                       num_workers: int = 50,
                       name: str = 'Shallow Model Attack'):

    assert standard_model.bounds() == defended_model.bounds()

    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []

    for images, labels in test_utils.get_iterator(name, logger, loader):
        samples_count += len(images)

        # First step: Remove samples misclassified by the defended model
        correct_images, correct_labels = batch_attack.get_correct_samples(
            defended_model, images, labels)

        correct_count += len(correct_images)
        images, labels = correct_images, correct_labels

        # Second step: Generate adversarial samples against the standard model (removing failed adversarials)
        assert len(images) == len(labels)
        adversarials, images, labels = batch_attack.get_adversarials(
            standard_model, images, labels, attack, True, cuda, num_workers)

        # Third step: Remove adversarial samples that are correctly classified by the defended model
        adversarial_predictions = defended_model.batch_predictions(
            adversarials)
        adversarial_labels = np.argmax(adversarial_predictions, axis=1)
        successful_attack = np.not_equal(labels, adversarial_labels)
        images = images[successful_attack]
        labels = labels[successful_attack]
        adversarials = adversarials[successful_attack]
        adversarial_labels = adversarial_labels[successful_attack]

        successful_attack_count += len(adversarials)

        # Fourth step: Compute the distances
        batch_distances = utils.lp_distance(images, adversarials, p, True)
        distances += list(batch_distances)

        accuracy = correct_count / samples_count
        success_rate = successful_attack_count / correct_count
        logger.debug('Accuracy: {:2.2f}%'.format(accuracy * 100.0))
        logger.debug('Success Rate: {:2.2f}%'.format(success_rate * 100.0))

    assert samples_count >= correct_count
    assert correct_count >= successful_attack_count
    assert len(distances) == successful_attack_count

    return samples_count, correct_count, successful_attack_count, np.array(distances)
