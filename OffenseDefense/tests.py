from typing import List, Tuple
import numpy as np
import torch
import foolbox
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from progress.bar import IncrementalBar
import logging
from . import batch_attack, batch_processing, detectors, distance_tools, loaders, utils

logger = logging.getLogger(__name__)


def _get_iterator(name, loader):
    if logger.getEffectiveLevel() == logging.INFO:
        return IncrementalBar(name).iter(loader)
    else:
        return loader


def accuracy_test(foolbox_model: foolbox.models.Model,
                  loader: loaders.Loader,
                  top_ks: List[int],
                  name: str = 'Accuracy Test'):
    accuracies = [utils.AverageMeter() for _ in range(len(top_ks))]

    for images, labels in _get_iterator(name, loader):
        batch_predictions = foolbox_model.batch_predictions(images)

        for i, top_k in enumerate(top_ks):
            correct_samples_count = utils.top_k_count(
                batch_predictions, labels, top_k)
            accuracies[i].update(1, correct_samples_count)
            accuracies[i].update(0, len(images) - correct_samples_count)

        for i in np.argsort(top_ks):
            logger.debug(
                'Top-{} Accuracy: {:2.2f}%'.format(top_ks[i], accuracies[i].avg * 100.0))

        logger.debug('\n============\n')

    # Return accuracies instead of AverageMeters
    return [accuracy.avg for accuracy in accuracies]


def attack_test(foolbox_model: foolbox.models.Model,
                loader: loaders.Loader,
                attack: foolbox.attacks.Attack,
                p: int,
                batch_worker: batch_processing.BatchWorker = None,
                num_workers: int = 50,
                save_adversarials: bool = False,
                name: str = 'Attack Test') -> Tuple[float, np.ndarray]:

    success_rate = utils.AverageMeter()
    distances = []
    adversarials = [] if save_adversarials else None
    adversarial_ground_truths = [] if save_adversarials else None

    for images, labels in _get_iterator(name, loader):
        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        successful_adversarials, successful_images, successful_labels = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True, batch_worker, num_workers)

        success_rate.update(1, len(successful_adversarials))
        success_rate.update(0, len(correct_images) -
                            len(successful_adversarials))

        # If there are no successful adversarials, don't update the distances or the adversarials
        if len(successful_adversarials) > 0:
            distances += list(utils.lp_distance(
                successful_adversarials, successful_images, p, True))

            if save_adversarials:
                adversarials += list(successful_adversarials)
                adversarial_ground_truths += list(successful_labels)

        failure_count = success_rate.count - \
            success_rate.sum
        average_distance, median_distance, _, adjusted_median_distance = utils.distance_statistics(
            distances, failure_count)

        logger.debug('Average Distance: {:2.2e}'.format(average_distance))
        logger.debug('Median Distance: {:2.2e}'.format(median_distance))
        logger.debug('Success Rate: {:2.2f}%'.format(success_rate.avg * 100.0))
        logger.debug('Adjusted Median Distance: {:2.2e}'.format(
            adjusted_median_distance))

        logger.debug('\n============\n')

    failure_count = success_rate.count - success_rate.sum

    if save_adversarials:
        adversarials = np.array(adversarials)
        adversarial_ground_truths = np.array(adversarial_ground_truths)

    return distances, failure_count, adversarials, adversarial_ground_truths


# TODO: Foolbox model -> standard model. Add optional 'defended model' (preprocessor or model foolbox)
def shallow_attack_test(foolbox_model: foolbox.models.Model,
                        loader,
                        attack,
                        p,
                        detector=None,
                        threshold=None,
                        batch_worker: batch_processing.BatchWorker = None,
                        num_workers: int = 50,
                        name: str = 'Shallow Attack'):
    samples_count = 0
    correct_samples_count = 0
    successful_attack_count = 0
    distances = []

    for images, labels in _get_iterator(name, loader):
        samples_count += len(images)

        # First step: Remove misclassified samples
        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        # Second step (optional): Remove samples that are wrongfully detected as adversarial
        if detector is not None:
            if threshold is None:
                raise ValueError(
                    'threshold must be set if detector is not None.')
            correct_images, correct_labels = batch_attack.get_approved_samples(
                foolbox_model, correct_images, correct_labels, detector, threshold)

        correct_samples_count += len(correct_images)
        images, labels = correct_images, correct_labels

        # Third step: Generate adversarial samples from the remaining samples (removing failed adversarials)
        adversarials, images, labels = batch_attack.get_adversarials(
            foolbox_model, images, labels, attack, True, batch_worker, num_workers)

        # Fourth step (optional): Remove adversarial samples that are detected as such
        if detector is not None:
            scores = np.array(detector.get_scores(adversarials))
            is_valid = scores >= threshold
            valid_indices = [i for i in range(
                len(adversarials)) if is_valid[i]]
            valid_indices = np.array(valid_adversarials)
            valid_adversarials = adversarials[valid_indices]
            adversarials = valid_adversarials

        successful_attack_count += len(adversarials)

        # Fifth step: Compute the distances
        batch_distances = utils.lp_distance(images, adversarials, p, True)
        distances += list(batch_distances)

        accuracy = correct_samples_count / samples_count
        success_rate = successful_attack_count / correct_samples_count
        logger.debug('Accuracy: {:2.2f}%'.format(accuracy * 100.0))
        logger.debug('Success Rate: {:2.2f}%'.format(success_rate * 100.0))

    accuracy = correct_samples_count / samples_count
    success_rate = successful_attack_count / correct_samples_count

    return accuracy, success_rate, np.array(distances)


def standard_detector_test(foolbox_model: foolbox.models.Model,
                           genuine_loader: loaders.Loader,
                           adversarial_loader: loaders.Loader,
                           detector: detectors.Detector,
                           save_samples: bool,
                           name: str = 'Detection Test'):
    genuine_samples = [] if save_samples else None
    genuine_scores = []
    adversarial_samples = [] if save_samples else None
    adversarial_scores = []

    for images, _ in _get_iterator(name + ' (Genuine)', genuine_loader):
        if save_samples:
            genuine_samples += list(images)
        genuine_scores += list(detector.get_scores(images))

        logger.debug('\n============\n')

    for adversarials, _ in _get_iterator(name + ' (Adversarial)', adversarial_loader):
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


def parallelization_test(foolbox_model: foolbox.models.Model,
                         loader: loaders.Loader,
                         attack: foolbox.attacks.Attack,
                         p: int,
                         batch_worker: batch_processing.BatchWorker,
                         num_workers: int = 50,
                         name: str = 'Parallelization Test'):

    standard_success_rate = utils.AverageMeter()
    parallel_success_rate = utils.AverageMeter()
    standard_distances = []
    parallel_distances = []

    for images, labels in _get_iterator(name, loader):
        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        # Run the parallel attack
        parallel_adversarials, parallel_images, _ = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True, batch_worker=batch_worker, num_workers=num_workers)

        parallel_success_rate.update(1, len(parallel_adversarials))
        parallel_success_rate.update(
            0, len(correct_images) - len(parallel_adversarials))

        parallel_distances += list(utils.lp_distance(
            parallel_adversarials, parallel_images, p, True))

        # Run the standard attack
        standard_adversarials, standard_images, _ = batch_attack.get_adversarials(
            foolbox_model, correct_images, correct_labels, attack, True)

        standard_success_rate.update(1, len(standard_adversarials))
        standard_success_rate.update(
            0, len(correct_images) - len(standard_adversarials))

        standard_distances += list(utils.lp_distance(
            standard_adversarials, standard_images, p, True))

        # Compute the statistics, treating failures as samples with distance=Infinity
        standard_failure_count = standard_success_rate.count - standard_success_rate.sum
        parallel_failure_count = parallel_success_rate.count - parallel_success_rate.sum

        standard_average_distance, standard_median_distance, _, standard_adjusted_median_distance = utils.distance_statistics(
            standard_distances, standard_failure_count)
        parallel_average_distance, parallel_median_distance, _, parallel_adjusted_median_distance = utils.distance_statistics(
            parallel_distances, parallel_failure_count)

        average_distance_difference = (
            parallel_average_distance - standard_average_distance) / standard_average_distance
        median_distance_difference = (
            parallel_median_distance - standard_median_distance) / standard_median_distance
        success_rate_difference = (
            parallel_success_rate.avg - standard_success_rate.avg) / standard_success_rate.avg
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

    standard_failure_count = standard_success_rate.count - standard_success_rate.sum
    parallel_failure_count = parallel_success_rate.count - parallel_success_rate.sum

    return standard_distances, standard_failure_count, parallel_distances, parallel_failure_count
