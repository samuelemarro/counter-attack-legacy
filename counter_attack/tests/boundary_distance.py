import logging

import foolbox
import numpy as np

from counter_attack import attacks, batch_attack, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

# Also known as Consistency Test
def boundary_distance_test(foolbox_model, loader, distance_tool, max_radius, cuda, num_workers, name='Boundary Distance Test'):
    p = distance_tool.p

    # This attack uses binary search to find samples on the boundary
    foolbox_distance = utils.p_to_foolbox(p)

    initialization_attack = None

    # If possible, use DeepFool to generate the starting points
    if isinstance(foolbox_model, foolbox.models.DifferentiableModel):
        if p == 2:
            initialization_attack = foolbox.attacks.DeepFoolL2Attack
        elif np.isposinf(p):
            initialization_attack = foolbox.attacks.DeepFoolLinfinityAttack

    # Otherwise, use Salt and Pepper
    if initialization_attack is None:
        initialization_attack = foolbox.attacks.SaltAndPepperNoiseAttack

    attack = attacks.ImageBlendingAttack(foolbox_model, foolbox.criteria.Misclassification(), foolbox_distance, None)

    samples_count = 0
    consistent_count = 0
    failure_count = 0
    inconsistent_differences = []

    for images, labels in test_utils.get_iterator(name, logger, loader):
        # Find samples exactly on the boundary
        # These will be the original images
        adversarials, images, labels = batch_attack.get_adversarials(foolbox_model, images, labels, attack, True, cuda, num_workers)

        original_images = adversarials
        del images, labels, adversarials

        logger.debug('Generated {} starting points.'.format(len(original_images)))

        if len(original_images) == 0:
            continue

        movements = []
        shape = original_images[0].shape
        
        for _ in original_images:
            #print(utils.lp_norm(utils.sample_lp_hypersphere(2, shape), 2))
            movement = utils.random_unit_vector(shape) * max_radius
            movements.append(movement)

        movements = np.array(movements)
        new_images = original_images + movements

        # Clip to the accepted bounds
        min_, max_ = foolbox_model.bounds()
        new_images = new_images.clip(min_, max_)

        original_boundary_distances = distance_tool.get_distances(original_images)
        new_boundary_distances = distance_tool.get_distances(new_images)
        original_new_distances = utils.lp_distance(original_images, new_images, p, True)

        logger.debug('Original boundary distances: {}'.format(original_boundary_distances))
        logger.debug('New boundary distances: {}'.format(new_boundary_distances))
        logger.debug('Distances between original and new samples: {}'.format(original_new_distances))

        samples_count += len(original_images)

        # Remove the cases where the original or the new distance is infinity
        failed_measures = np.logical_or(np.isinf(original_boundary_distances), np.isinf(new_boundary_distances))
        logger.debug('Removed {} failed samples'.format(np.count_nonzero(failed_measures)))

        failure_count += np.count_nonzero(failed_measures)
        successful_measures = np.logical_not(failed_measures)
        
        original_boundary_distances = np.array(original_boundary_distances)[successful_measures]
        new_boundary_distances = np.array(new_boundary_distances)[successful_measures]
        original_new_distances = np.array(original_new_distances)[successful_measures]

        # The difference between the original boundary distance and the new
        # computed distance must be less than or equal to the distance between
        # the original sample and the new one. In other words, moving a sample
        # by k must change the boundary distance by at most k
        differences = original_new_distances - np.abs(original_boundary_distances - new_boundary_distances)
        consistent = differences >= 0 # original_new_distances >= np.abs(...)
        inconsistent = np.logical_not(consistent)

        consistent_count += np.count_nonzero(consistent)
        inconsistent_differences.extend(np.abs(differences[inconsistent]))

        failure_rate = failure_count / samples_count
        consistency_rate = consistent_count / samples_count
        effective_consistency_rate = consistent_count / (samples_count - failure_count)
        average_difference = np.average(inconsistent_differences)
        median_difference = np.median(inconsistent_differences)

        logger.debug('Failure Rate: {:.2f}%'.format(failure_rate * 100.0))
        logger.debug('Consistency Rate: {:.2f}%'.format(consistency_rate * 100.0))
        logger.debug('Effective Consistency Rate: {:.2f}%'.format(effective_consistency_rate * 100.0))
        logger.debug('Average Difference: {:2.2e}'.format(average_difference))
        logger.debug('Median Difference: {:2.2e}'.format(median_difference))

    #inconsistent_differences = np.array(inconsistent_differences)

    return samples_count, consistent_count, failure_count, inconsistent_differences