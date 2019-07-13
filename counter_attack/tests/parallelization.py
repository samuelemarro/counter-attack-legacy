import logging

import foolbox

from counter_attack import batch_attack, distance_measures, loaders, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def parallelization_test(foolbox_model: foolbox.models.Model,
                         loader: loaders.Loader,
                         attack: foolbox.attacks.Attack,
                         lp_distance: distance_measures.LpDistance,
                         cuda: bool,
                         num_workers: int = 50,
                         name: str = 'Parallelization Test'):

    samples_count = 0
    correct_count = 0
    standard_attack_count = 0
    parallel_attack_count = 0
    standard_distances = []
    parallel_distances = []

    for images, labels in test_utils.get_iterator(name, logger, loader):
        samples_count += len(images)

        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        correct_count += len(correct_images)

        # Run the parallel attack
        parallel_adversarials, parallel_images, _ = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True, cuda, num_workers=num_workers)

        parallel_attack_count += len(parallel_adversarials)

        parallel_distances += list(lp_distance.compute(parallel_adversarials, parallel_images, True))

        # Run the standard attack
        standard_adversarials, standard_images, _ = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True, cuda)

        standard_attack_count += len(standard_adversarials)

        standard_distances += list(lp_distance.compute(standard_adversarials, standard_images, True))

        # Compute the statistics, treating failures as samples with distance=Infinity
        standard_failure_count = correct_count - standard_attack_count
        parallel_failure_count = correct_count - parallel_attack_count

        standard_average_distance, standard_median_distance, standard_adjusted_median_distance = utils.distance_statistics(
            standard_distances, standard_failure_count)
        parallel_average_distance, parallel_median_distance, parallel_adjusted_median_distance = utils.distance_statistics(
            parallel_distances, parallel_failure_count)

        standard_success_rate = standard_attack_count / correct_count
        parallel_success_rate = parallel_attack_count / correct_count

        average_distance_difference = (
            parallel_average_distance - standard_average_distance) / standard_average_distance
        median_distance_difference = (
            parallel_median_distance - standard_median_distance) / standard_median_distance
        success_rate_difference = (
            parallel_success_rate - standard_success_rate) / standard_success_rate
        adjusted_median_distance_difference = (
            parallel_adjusted_median_distance - standard_adjusted_median_distance) / standard_adjusted_median_distance

        logger.debug('Average Distance Relative Difference: {:2.5f}%'.format(
            average_distance_difference * 100.0))
        logger.debug('Median Distance Relative Difference: {:2.5f}%'.format(
            median_distance_difference * 100.0))
        logger.debug('Success Rate Relative Difference: {:2.5f}%'.format(
            success_rate_difference * 100.0))
        logger.debug('Adjusted Median Distance Relative Difference: {:2.5f}%'.format(
            adjusted_median_distance_difference * 100.0))

        logger.debug('\n============\n')

    assert samples_count >= correct_count
    assert correct_count >= standard_attack_count
    assert correct_count >= parallel_attack_count
    assert len(standard_distances) == standard_attack_count
    assert len(parallel_distances) == parallel_attack_count

    return samples_count, correct_count, standard_attack_count, parallel_attack_count, standard_distances, parallel_distances