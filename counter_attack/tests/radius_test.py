import logging

import numpy as np

from counter_attack import distance_measures
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def radius_test(foolbox_model, loader, distance_tool, sampling_count, name='Radius Test'):
    lp_distance = distance_tool.lp_distance

    if lp_distance != 2 and not np.isposinf(lp_distance):
        raise NotImplementedError('Radius test supports only L2 and L-Inf measures')

    for images, _ in test_utils.get_iterator(name, logger, loader):
        predicted_labels = np.argmax(foolbox_model.batch_predictions(images), axis=-1)
        estimated_distances = distance_tool.get_distances(images)

        for image, predicted_label, estimated_distance in zip(images, predicted_labels, estimated_distances):
            new_points = []

