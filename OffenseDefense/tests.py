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


def attack_test(foolbox_model: foolbox.models.Model,
                loader: loaders.Loader,
                attack: foolbox.attacks.Attack,
                p: int,
                parallel_pooler = False,
                save_adversarials: bool = False,
                name: str = 'Attack Test') -> Tuple[float, np.ndarray]:

    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []
    adversarials = [] if save_adversarials else None
    adversarial_ground_truths = [] if save_adversarials else None

    for images, labels in _get_iterator(name, loader):
        samples_count += len(images)
        logger.debug('Received {} images'.format(len(images)))

        # Remove misclassified samples
        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        correct_count += len(correct_images)
        logger.debug('{} correct images ({} removed)'.format(
            correct_count, samples_count - correct_count))

        successful_adversarials, successful_images, successful_labels = batch_attack.get_adversarials(
            correct_images, correct_labels, attack, foolbox_model, parallel_pooler, True)

        successful_attack_count += len(successful_adversarials)
        logger.debug('{} successful attacks ({} removed)'.format(
            successful_attack_count, correct_count - successful_attack_count))

        # Update the distances and/or the adversarials (if there are successful adversarials)
        if len(successful_adversarials) > 0:
            distances += list(utils.lp_distance(
                successful_adversarials, successful_images, p, True))

            if save_adversarials:
                adversarials += list(successful_adversarials)
                adversarial_ground_truths += list(successful_labels)

        failure_count = correct_count - successful_attack_count

        average_distance, median_distance, adjusted_median_distance = utils.distance_statistics(
            distances, failure_count)

        logger.debug('Average Distance: {:2.2e}'.format(average_distance))
        logger.debug('Median Distance: {:2.2e}'.format(median_distance))
        logger.debug('Success Rate: {:2.2f}%'.format(
            successful_attack_count / float(correct_count) * 100.0))
        logger.debug('Adjusted Median Distance: {:2.2e}'.format(
            adjusted_median_distance))

        logger.debug('\n============\n')

    if save_adversarials:
        adversarials = np.array(adversarials)
        adversarial_ground_truths = np.array(adversarial_ground_truths)

    assert samples_count >= correct_count
    assert correct_count >= successful_attack_count
    assert len(distances) == successful_attack_count

    if adversarials is not None:
        assert len(distances) == len(adversarials)
        assert len(adversarials) == len(adversarial_ground_truths)

    return samples_count, correct_count, successful_attack_count, distances, adversarials, adversarial_ground_truths


def shallow_rejector_test(standard_model: foolbox.models.Model,
                          loader,
                          attack,
                          p,
                          rejector,
                          standard_parallel_pooler = None,
                          name: str = 'Shallow Rejector Attack'):
    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []

    for images, labels in _get_iterator(name, loader):
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
            images, labels, attack, standard_model, standard_parallel_pooler, True)

        # Fourth step: Remove adversarial samples that are detected as such
        batch_valid = rejector.batch_valid(adversarials)
        adversarials = adversarials[batch_valid]
        images = images[batch_valid]
        labels = labels[batch_valid]

        successful_attack_count += len(adversarials)

        # Fifth step: Compute the distances
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


def transfer_test(standard_model: foolbox.models.Model,
                       loader,
                       attack,
                       p,
                       defended_model: foolbox.models.Model,
                       standard_parallel_pooler = None,
                       name: str = 'Shallow Model Attack'):
    samples_count = 0
    correct_count = 0
    successful_attack_count = 0
    distances = []

    for images, labels in _get_iterator(name, loader):
        samples_count += len(images)

        # First step: Remove samples misclassified by the defended model
        correct_images, correct_labels = batch_attack.get_correct_samples(
            defended_model, images, labels)

        correct_count += len(correct_images)
        images, labels = correct_images, correct_labels

        # Second step: Generate adversarial samples against the standard model (removing failed adversarials)
        assert len(images) == len(labels)
        adversarials, images, labels = batch_attack.get_adversarials(
            images, labels, attack, standard_model, standard_parallel_pooler, True)

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

    for images, _ in _get_iterator(name + ' (Genuine)', genuine_loader):
        if save_samples:
            genuine_samples += list(images)
        genuine_scores += list(detector.get_scores(images))

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
                         parallel_pooler = None,
                         name: str = 'Parallelization Test'):

    samples_count = 0
    correct_count = 0
    standard_attack_count = 0
    parallel_attack_count = 0
    standard_distances = []
    parallel_distances = []

    for images, labels in _get_iterator(name, loader):
        samples_count += len(images)

        correct_images, correct_labels = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        correct_count += len(correct_images)

        # Run the parallel attack
        parallel_adversarials, parallel_images, _ = batch_attack.get_adversarials(
            correct_images, correct_labels, attack, foolbox_model, parallel_pooler, True)

        parallel_attack_count += len(parallel_adversarials)

        parallel_distances += list(utils.lp_distance(
            parallel_adversarials, parallel_images, p, True))

        # Run the standard attack
        standard_adversarials, standard_images, _ = batch_attack.get_adversarials(
            correct_images, correct_labels, attack, foolbox_model, None, True)

        standard_attack_count += len(standard_adversarials)

        standard_distances += list(utils.lp_distance(
            standard_adversarials, standard_images, p, True))

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
