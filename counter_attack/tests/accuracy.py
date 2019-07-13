import logging
from typing import List

import foolbox
import numpy as np

from counter_attack import loaders, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)


def accuracy_test(foolbox_model: foolbox.models.Model,
                  loader: loaders.Loader,
                  top_ks: List[int],
                  name: str = 'Accuracy Test'):
    accuracies = [utils.AverageMeter() for _ in range(len(top_ks))]

    for images, labels in test_utils.get_iterator(name, logger, loader):
        batch_predictions = foolbox_model.batch_predictions(images)

        for i, top_k in enumerate(top_ks):
            correct_count = utils.top_k_count(
                batch_predictions, labels, top_k)
            accuracies[i].update(1, correct_count)
            accuracies[i].update(0, len(images) - correct_count)

        for i in np.argsort(top_ks):
            logger.debug(
                'Top-{} Accuracy: {:2.2f}%'.format(top_ks[i], accuracies[i].avg * 100.0))

        logger.debug('\n============\n')

    # Return accuracies instead of AverageMeters
    return [accuracy.avg for accuracy in accuracies]