import logging

import numpy as np

from counter_attack import distance_measures, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def radius_test(foolbox_model, loader, distance_tool, sampling_count, name='Radius Test'):
    lp_distance = distance_tool.lp_distance
    p = lp_distance.p

    if p != 2 and not np.isposinf(p):
        raise NotImplementedError('Radius test supports only L2 and L-Inf measures')

    min_, max_ = foolbox_model.bounds()
    
    total_count = 0
    consistent_count = 0
    failures = 0

    for images, _ in test_utils.get_iterator(name, logger, loader):
        predicted_labels = np.argmax(foolbox_model.batch_predictions(images), axis=-1)
        estimated_distances = distance_tool.get_distances(images)

        for image, predicted_label, estimated_distance in zip(images, predicted_labels, estimated_distances):
            if np.isinf(estimated_distance):
                # No distance estimate, ignore
                failures += 1
                continue

            sampled_points = []

            for _ in range(sampling_count):
                perturbations = test_utils.sample_lp_ball(p, estimated_distance, image.shape)
                logger.debug('Perturbation magnitudes: {}'.format([utils.lp_norm(perturbation, p) for perturbation in perturbations]))
                point = image + perturbations
                point = np.clip(point, min_, max_)
                sampled_points.append(point)

            sampled_points = np.array(sampled_points).astype(np.float32)

            #logger.debug('Sampled magnitudes: {}'.format([utils.lp_norm(point, p) for point in sampled_points]))

            sampled_labels = np.argmax(foolbox_model.batch_predictions(sampled_points), axis=-1)

            total_count += sampling_count

            # Same labels as the original one
            consistent_labels = np.equal(sampled_labels, predicted_label)

            consistent_count += np.count_nonzero(consistent_labels)

    return total_count, consistent_count, failures

