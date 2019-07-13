import logging

import foolbox
import sklearn.metrics as metrics

from counter_attack import detectors, loaders, utils
from counter_attack.tests import test_utils

logger = logging.getLogger(__name__)

def roc_curve_test(foolbox_model: foolbox.models.Model,
                    genuine_loader: loaders.Loader,
                    adversarial_loader: loaders.Loader,
                    detector: detectors.Detector,
                    save_samples: bool,
                    name: str = 'Detection Test'):
    """
        Computes the ROC of a shallow detector.
    """
    genuine_samples = [] if save_samples else None
    genuine_scores = []
    adversarial_samples = [] if save_samples else None
    adversarial_scores = []

    for images, _ in test_utils.get_iterator(name + ' (Genuine)', logger, genuine_loader):
        if save_samples:
            genuine_samples += list(images)
        genuine_scores += list(detector.get_scores(images))

    for adversarials, _ in test_utils.get_iterator(name + ' (Adversarial)', logger, adversarial_loader):
        if save_samples:
            adversarial_samples += list(adversarials)

        adversarial_scores += list(detector.get_scores(adversarials))

        fpr, tpr, thresholds = utils.roc_curve(
            genuine_scores, adversarial_scores)
        best_threshold, best_tpr, best_fpr = utils.get_best_threshold(
            tpr, fpr, thresholds)
        area_under_curve = metrics.auc(fpr, tpr)

        logger.debug('Detector AUC: {:2.2f}%'.format(area_under_curve * 100.0))
        logger.debug('Best Threshold: {:2.2e}'.format(best_threshold))
        logger.debug('Best TPR: {:2.2f}%'.format(best_tpr * 100.0))
        logger.debug('Best FPR: {:2.2f}%'.format(best_fpr * 100.0))

        logger.debug('\n============\n')

    return genuine_scores, adversarial_scores, genuine_samples, adversarial_samples