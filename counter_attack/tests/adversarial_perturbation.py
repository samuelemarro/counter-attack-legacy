import logging

import numpy as np

from counter_attack import batch_attack, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def adversarial_perturbation_test(foolbox_model, loader, attack, distance_tool, cuda, num_workers, name='Adversarial Perturbation Test'):
    p = distance_tool.p

    samples_count = 0
    correct_count = 0
    successful_count = 0
    correct_estimate_count = 0

    final_boundary_distances = []
    final_adversarial_distances = []

    for images, labels in test_utils.get_iterator(name, logger, loader):
        samples_count += len(images)

        # Remove misclassified samples
        images, labels = batch_attack.get_correct_samples(foolbox_model, images, labels)
        correct_count += len(images)

        adversarials, images, labels = batch_attack.get_adversarials(foolbox_model, images, labels, attack, True, cuda, num_workers)
        successful_count += len(images)

        boundary_distances = distance_tool.get_distances(adversarials)
        adversarial_distances = utils.lp_distance(images, adversarials, p, True)

        correct_estimates = adversarial_distances >= boundary_distances
        correct_estimate_count += np.count_nonzero(correct_estimates)

        final_boundary_distances += list(boundary_distances)
        final_adversarial_distances += list(adversarial_distances)

    return samples_count, correct_count, successful_count, correct_estimate_count, final_boundary_distances, final_adversarial_distances
