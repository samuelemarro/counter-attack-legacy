import warnings
from typing import List, Tuple
import numpy as np
import torch
import foolbox
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from . import batch_attack, batch_processing, detectors, distance_tools, loaders, utils


def accuracy_test(foolbox_model: foolbox.models.Model,
                  loader: loaders.Loader,
                  top_ks: List[int],
                  verbose: bool = True):
    accuracies = [utils.AverageMeter() for _ in range(len(top_ks))]

    for images, labels in loader:
        batch_predictions = foolbox_model.batch_predictions(images)

        for i, top_k in enumerate(top_ks):
            correct_samples_count = utils.top_k_count(
                batch_predictions, labels, top_k)
            accuracies[i].update(1, correct_samples_count)
            accuracies[i].update(0, len(images) - correct_samples_count)

        if verbose:
            for i in np.argsort(top_ks):
                print(
                    'Top-{} Accuracy: {:2.2f}%'.format(top_ks[i], accuracies[i].avg * 100.0))

            print('\n============\n')

    # Return accuracies instead of AverageMeters
    return [accuracy.avg for accuracy in accuracies]


def distance_comparison_test(foolbox_model: foolbox.models.Model,
                             test_distance_tools,
                             loader: loaders.Loader,
                             verbose: bool = True):

    final_estimated_distances = {}
    success_rates = {}

    for distance_tool in test_distance_tools:
        final_estimated_distances[distance_tool.name] = []
        success_rates[distance_tool.name] = utils.AverageMeter()

    for images, labels in loader:
        # Remove misclassified samples
        correct_classification_filter = batch_attack.get_correct_samples(
            foolbox_model, images, labels)

        images = correct_classification_filter['images']
        labels = correct_classification_filter['image_labels']

        for distance_tool in test_distance_tools:
            estimated_distances = distance_tool.get_distances(images)
            successful_estimated_distances = [
                distance for distance in estimated_distances if distance is not None]

            success_rates[distance_tool.name].update(
                1, len(successful_estimated_distances))
            success_rates[distance_tool.name].update(
                0, len(images) - len(successful_estimated_distances))

            final_estimated_distances[distance_tool.name] += successful_estimated_distances

            tool_estimated_distances = final_estimated_distances[distance_tool.name]

            if verbose:
                failure_count = success_rates[distance_tool.name].count - \
                    success_rates[distance_tool.name].sum
                average_distance, median_distance, _, adjusted_median_distance = utils.distance_statistics(
                    tool_estimated_distances, failure_count)

                print('{}:'.format(distance_tool.name))
                print('Average Distance: {:2.2e}'.format(average_distance))
                print('Median Distance: {:2.2e}'.format(median_distance))
                print('Success Rate: {:2.2f}%'.format(
                    success_rate.avg * 100.0))
                print('Adjusted Median Distance: {:2.2e}'.format(
                    adjusted_median_distance))

                print('\n============\n')

        if verbose:
            print('\n====================\n')

    # Replace AverageMeters with the final averages
    for key, value in success_rates.items():
        success_rates[key] = value.avg

    return final_estimated_distances, success_rates


def attack_test(foolbox_model: foolbox.models.Model,
                loader: loaders.Loader,
                attack: foolbox.attacks.Attack,
                p: int,
                batch_worker: batch_processing.BatchWorker = None,
                num_workers: int = 50,
                verbose: bool = True) -> Tuple[float, np.ndarray]:

    success_rate = utils.AverageMeter()
    distances = []

    for images, labels in loader:
        adversarial_filter = batch_attack.get_adversarials(
            foolbox_model, images, labels, attack, True, True, batch_worker=batch_worker, num_workers=num_workers)
        adversarials = adversarial_filter['adversarials']

        success_rate.update(1, len(adversarials))
        success_rate.update(0, len(images) - len(adversarials))

        # If there are no successful adversarials, don't update the distances
        if len(adversarials) > 0:
            distances += list(utils.lp_distance(
                adversarial_filter['adversarials'], adversarial_filter['images'], p, True))

        if verbose:
            failure_count = success_rate.count - \
                success_rate.sum
            average_distance, median_distance, _, adjusted_median_distance = utils.distance_statistics(
                distances, failure_count)

            print('Average Distance: {:2.2e}'.format(average_distance))
            print('Median Distance: {:2.2e}'.format(median_distance))
            print('Success Rate: {:2.2f}%'.format(success_rate.avg * 100.0))
            print('Adjusted Median Distance: {:2.2e}'.format(
                adjusted_median_distance))

            print('\n============\n')

    failure_count = success_rate.count - success_rate.sum

    return distances, failure_count


def standard_detector_test(foolbox_model: foolbox.models.Model,
                           loader: loaders.Loader,
                           attack: foolbox.attacks.Attack,
                           detector: detectors.Detector,
                           batch_worker: batch_processing.BatchWorker = None,
                           num_workers: int = 50,
                           verbose: bool = True):
    attack_success_rate = utils.AverageMeter()
    genuine_scores = []
    adversarial_scores = []

    for images, labels in loader:
        adversarial_filter = batch_attack.get_adversarials(
            foolbox_model, images, labels, attack, True, True, batch_worker=batch_worker, num_workers=num_workers)

        adversarials = adversarial_filter['adversarials']

        attack_success_rate.update(1, len(adversarials))
        attack_success_rate.update(0, len(images) - len(adversarials))

        # Note: We evaluate the detector on all the genuine samples,
        # even the ones that could not be attacked
        genuine_scores += list(detector.get_scores(images))
        adversarial_scores += list(detector.get_scores(adversarials))

        if verbose:
            fpr, tpr, thresholds = utils.roc_curve(
                genuine_scores, adversarial_scores)
            best_threshold, best_tpr, best_fpr = utils.get_best_threshold(
                tpr, fpr, thresholds)
            area_under_curve = metrics.auc(fpr, tpr)

            print('Attack Success Rate: {:2.2f}%'.format(
                attack_success_rate.avg * 100.0))
            print('Detector AUC: {:2.2f}%'.format(area_under_curve * 100.0))
            print('Best Threshold: {:2.2e}'.format(best_threshold))
            print('Best TPR: {:2.2f}%'.format(best_tpr * 100.0))
            print('Best FPR: {:2.2f}%'.format(best_fpr * 100.0))

            print('\n============\n')

    return genuine_scores, adversarial_scores, attack_success_rate.avg


def parallelization_test(foolbox_model: foolbox.models.Model,
                         loader: loaders.Loader,
                         attack: foolbox.attacks.Attack,
                         p: int,
                         batch_worker: batch_processing.BatchWorker,
                         num_workers: int = 50,
                         verbose: bool = True):

    standard_success_rate = utils.AverageMeter()
    parallel_success_rate = utils.AverageMeter()
    standard_distances = []
    parallel_distances = []

    for images, labels in loader:
        # Run the parallel attack
        parallel_adversarial_filter = batch_attack.get_adversarials(
            foolbox_model, images, labels, attack, True, True, batch_worker=batch_worker, num_workers=num_workers)
        parallel_adversarials = parallel_adversarial_filter['adversarials']

        parallel_success_rate.update(1, len(parallel_adversarials))
        parallel_success_rate.update(
            0, len(images) - len(parallel_adversarials))

        parallel_distances += list(utils.lp_distance(
            parallel_adversarial_filter['adversarials'], parallel_adversarial_filter['images'], p, True))

        # Run the standard attack
        standard_adversarial_filter = batch_attack.get_adversarials(
            foolbox_model, images, labels, attack, True, True)
        standard_adversarials = standard_adversarial_filter['adversarials']

        standard_success_rate.update(1, len(standard_adversarials))
        standard_success_rate.update(
            0, len(images) - len(standard_adversarials))

        standard_distances += list(utils.lp_distance(
            standard_adversarial_filter['adversarials'], standard_adversarial_filter['images'], p, True))

        if verbose:
            # Compute the statistics, treating failures as samples with distance=Infinity
            standard_failure_count = standard_success_rate.count - standard_success_rate.sum
            parallel_failure_count = parallel_success_rate.count - parallel_success_rate.sum

            standard_average_distance, standard_median_distance, _, standard_adjusted_median_distance = utils.distance_statistics(
                standard_distances, standard_failure_count)
            print('Average Standard Distance: {:2.5e}'.format(
                standard_average_distance))
            print('Median Standard Distance: {:2.5e}'.format(
                standard_median_distance))
            print('Standard Success Rate: {:2.2f}%'.format(
                standard_success_rate.avg * 100.0))
            print('Standard Adjusted Median Distance: {:2.5e}'.format(
                standard_adjusted_median_distance))

            print('\n')

            parallel_average_distance, parallel_median_distance, _, parallel_adjusted_median_distance = utils.distance_statistics(
                parallel_distances, parallel_failure_count)
            print('Parallel Average Distance: {:2.5e}'.format(
                parallel_average_distance))
            print('Parallel Median Distance: {:2.5e}'.format(
                parallel_median_distance))
            print('Parallel Success Rate: {:2.2f}%'.format(
                parallel_success_rate.avg * 100.0))
            print('Parallel Adjusted Median Distance: {:2.5e}'.format(
                parallel_adjusted_median_distance))

            print('\n')

            average_distance_difference = (
                parallel_average_distance - standard_average_distance) / standard_average_distance
            median_distance_difference = (
                parallel_median_distance - standard_median_distance) / standard_median_distance
            success_rate_difference = (
                parallel_success_rate.avg - standard_success_rate.avg) / standard_success_rate.avg
            adjusted_median_distance_difference = (
                parallel_adjusted_median_distance - standard_adjusted_median_distance) / standard_adjusted_median_distance
            print('Average Distance Relative Difference: {:2.5f}%'.format(
                average_distance_difference * 100.0))
            print('Median Distance Relative Difference: {:2.5f}%'.format(
                median_distance_difference * 100.0))
            print('Success Rate Relative Difference: {:2.5f}%'.format(
                success_rate_difference * 100.0))
            print('Adjusted Median Distance Relative Difference: {:2.5f}%'.format(
                adjusted_median_distance_difference * 100.0))

            print('\n============\n')

    return standard_success_rate.avg, standard_distances, parallel_success_rate.avg, parallel_distances
