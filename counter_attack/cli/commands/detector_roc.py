import logging

import click
import numpy as np
import sklearn.metrics

from counter_attack import tests, utils
from counter_attack.cli import options

logger = logging.getLogger(__name__)

@click.command()
@options.global_options
@options.dataset_options('test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('detector-roc')
@options.distance_options
@options.counter_attack_options(False)
@options.detector_options
@options.adversarial_dataset_options
@click.option('--score_dataset_path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the .zip file where the scores will be saved with their corresponding images. If unspecified, no scores will be saved.')
@click.option('--no-test-warning', is_flag=True,
              help='Disables the warning for running this test on the test set.')
def detector_roc(options, score_dataset_path, no_test_warning):
    """
    Uses a detector to identify adversarial samples and computes the ROC curve.

    \b
    Stores the following results:
        ROC Area Under Curve (ROC-AUC)
        Best Threshold: The threshold with the best Youden Index (TPR - FPR)
        Best Threshold True Positive Rate: The TPR at the best threshold
        Best Threshold False Positive Rate: The FPR at the best threshold

        Genuine Scores: All the scores computed for the genuine samples
        Adversarial Scores: All the scores computed for the adversarial samples

    The last three columns contain the data to build the ROC curve. These are:
        Thresholds
        True Positive Rates
        False Positive Rates

    Each threshold has a corresponding TPR and FPR.
    """

    adversarial_loader = options['adversarial_loader']
    command = options['command']
    dataset_type = options['dataset_type']
    detector = options['detector']
    failure_value = options['failure_value']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    results_path = options['results_path']

    save_scores = score_dataset_path is not None

    if dataset_type == 'test' and not no_test_warning:
        logger.warning('Remember to use \'--dataset-type train\' if you plan to use the results '
                       'to pick a threshold for other tests. You can disable this warning by passing '
                       '\'--no-test-warning\'.')

    genuine_scores, adversarial_scores, genuine_samples, adversarial_samples = tests.roc_curve_test(
        foolbox_model, genuine_loader, adversarial_loader, detector, save_scores)

    false_positive_rates, true_positive_rates, thresholds = utils.roc_curve(
        genuine_scores, adversarial_scores)

    best_threshold, best_tpr, best_fpr = utils.get_best_threshold(
        true_positive_rates, false_positive_rates, thresholds)
    area_under_curve = sklearn.metrics.auc(
        false_positive_rates, true_positive_rates)

    info = [['ROC AUC', '{:2.2f}%'.format(area_under_curve * 100.0)],
            ['Best Threshold', '{:2.2e}'.format(best_threshold)],
            ['Best Threshold True Positive Rate', '{:2.2f}%'.format(best_tpr * 100.0)],
            ['Best Threshold False Positive Rate', '{:2.2f}%'.format(best_fpr * 100.0)]]

    header = ['Genuine Scores', 'Adversarial Scores',
              'Thresholds', 'True Positive Rates', 'False Positive Rates']

    true_positive_rates = ['{:2.2f}%'.format(
        true_positive_rate * 100.0) for true_positive_rate in true_positive_rates]
    false_positive_rates = ['{:2.2f}%'.format(
        false_positive_rate * 100.0) for false_positive_rate in false_positive_rates]

    columns = [genuine_scores, adversarial_scores,
               thresholds, true_positive_rates, false_positive_rates]

    utils.save_results(results_path, table=columns, command=command,
                       info=info, header=header)

    if save_scores:
        # Remove failures

        genuine_not_failed = np.not_equal(genuine_scores, failure_value)
        genuine_samples = genuine_samples[genuine_not_failed]
        genuine_scores = genuine_scores[genuine_not_failed]

        adversarial_not_failed = np.not_equal(
            adversarial_scores, failure_value)
        adversarial_samples = adversarial_samples[adversarial_not_failed]
        adversarial_scores = adversarial_scores[adversarial_not_failed]

        genuine_list = zip(genuine_samples, genuine_scores)
        adversarial_list = zip(adversarial_samples, adversarial_scores)

        dataset = (genuine_list, adversarial_list)

        utils.save_zip(dataset, score_dataset_path)