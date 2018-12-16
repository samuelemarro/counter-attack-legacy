import logging

import foolbox
import sklearn
import click

import OffenseDefense
import OffenseDefense.attacks as attacks
import OffenseDefense.detectors as detectors
import OffenseDefense.distance_tools as distance_tools
import OffenseDefense.parsing as parsing
import OffenseDefense.tests as tests
import OffenseDefense.training as training
import OffenseDefense.utils as utils

logger = logging.getLogger('OffenseDefense')


@click.group()
def main(*args):
    logging.basicConfig()


@main.command()
@parsing.attack_options(parsing.supported_attacks)
def attack(options):
    """
    Runs an attack against the model.

    Stores the following results:
        Success Rate: The success rate of the attack.
        Average Distance: The average L_p distance of the successful adversarial samples from their original samples.
        Median Distance: The median L_p distance of the successful adversarial samples from their original samples.
        Adjusted Median Distance: The median L_p distance of the adversarial samples from their original samples, treating failed attacks as samples with distance Infinity.
    """

    command = options['command']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    batch_worker = options['batch_worker']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    p = options['p']
    results_path = options['results_path']

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    distances, failure_count = tests.attack_test(foolbox_model, loader, attack, p,
                                                 batch_worker, attack_workers)

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, [distances], command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.global_options
@click.option('-tk', '--top-ks', nargs=2, type=int, default=(1, 5), show_default=True,
              help='The two top-k accuracies that will be computed.')
def accuracy(options, top_ks):
    """
    Computes the accuracy of the model.

    Stores the following results:
        Top-K Accuracies: The accuracies, where k values are configurable with --top-ks.
    """

    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    accuracies = tests.accuracy_test(
        foolbox_model, loader, top_ks)
    accuracies = ['{:2.2f}%'.format(accuracy * 100.0)
                  for accuracy in accuracies]

    header = ['Top-{}'.format(top_k) for top_k in top_ks]
    utils.save_results(results_path, [accuracies], command, header=header)


@main.command()
@parsing.detector_options(parsing.supported_attacks)
@click.option('--no-detector-warning', '-ndw', is_flag=True,
              help='Disables the warning for running this test on the test set.')
def detect(options, no_detector_warning):
    """
    Uses a detector to identify adversarial samples.
    """

    command = options['command']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    batch_worker = options['batch_worker']
    dataset_type = options['dataset_type']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    p = options['p']
    results_path = options['results_path']

    if dataset_type == 'test' and not no_detector_warning:
        logger.warning('Remember to use \'-dt train\' if you plan to use the results '
                       'to pick a threshold for other tests. You can disable this warning by passing '
                       '\'--no-detector-warning\' (alias: \'-ndw\').')

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    genuine_scores, adversarial_scores, success_rate = tests.standard_detector_test(
        foolbox_model, loader, attack, detector, batch_worker, attack_workers)

    false_positive_rates, true_positive_rates, thresholds = utils.roc_curve(
        genuine_scores, adversarial_scores)

    best_threshold, best_tpr, best_fpr = utils.get_best_threshold(
        true_positive_rates, false_positive_rates, thresholds)
    area_under_curve = sklearn.metrics.auc(
        false_positive_rates, true_positive_rates)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['ROC AUC', '{:2.2f}%'.format(area_under_curve * 100.0)],
            ['Best Threshold', '{:2.2e}'.format(best_threshold)],
            ['Best True Positive Rate', '{:2.2f}%'.format(best_tpr * 100.0)],
            ['Best False Positive Rate', '{:2.2f}%'.format(best_fpr * 100.0)]]

    header = ['Genuine Scores', 'Adversarial Scores',
              'Thresholds', 'True Positive Rates', 'False Positive Rates']

    true_positive_rates = ['{:2.2}%'.format(
        true_positive_rate) for true_positive_rate in true_positive_rates]
    false_positive_rates = ['{:2.2}%'.format(
        false_positive_rate) for false_positive_rate in false_positive_rates]

    columns = [genuine_scores, adversarial_scores,
               thresholds, true_positive_rates, false_positive_rates]

    utils.save_results(results_path, columns, command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.distance_options(parsing.supported_attacks)
@click.argument('distance-tool', type=click.Choice(parsing.supported_detectors))
def distance(options, distance_tool):
    """
    Estimates the distance of the samples from the decision boundary.
    """

    command = options['command']
    foolbox_model = options['foolbox_model']
    results_path = options['results_path']
    loader = options['loader']
    distance_tool = parsing.parse_distance_tool(distance_tool, options)

    distances, failure_count = tests.distance_test(
        foolbox_model, distance_tool, loader)

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Attack Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, [distances], command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.evasion_options(parsing.black_box_attacks)
def black_box_evasion(options):
    """
    Treats the model and the detector as a black box.
    """

    attack_name = options['attack_name']
    batch_worker = options['batch_worker']
    command = options['command']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    attack_workers = options['attack_workers']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']
    threshold = options['threshold']

    composite_model = detectors.CompositeDetectorModel(
        foolbox_model, detector, threshold)

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        composite_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    distances, failure_count = tests.attack_test(composite_model, loader, attack,
                                                 p, batch_worker, attack_workers, name='Black Box Evasion Test')

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, [distances], command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.evasion_options(parsing.differentiable_attacks)
def differentiable_evasion(options):
    attack_name = options['attack_name']
    batch_worker = options['batch_worker']
    command = options['command']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    attack_workers = options['attack_workers']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']
    threshold = options['threshold']

    # composite_model = detectors.CompositeDetectorModel(
    #    foolbox_model, detector, threshold)

    #attack_constructor = get_attack_constructor(attack_name, p)
    # attack = attack_constructor(
    #    composite_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    composite_model = None
    distances, failure_count = tests.attack_test(composite_model, loader, attack,
                                                 p, batch_worker, attack_workers, name='Differentiable Evasion Test')

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, [distances], command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.attack_options(parsing.parallelizable_attacks)
def parallelization(options):
    """
    Compares parallelized attacks with standard ones.
    """

    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    batch_worker = options['batch_worker']
    command = options['command']
    foolbox_model = options['foolbox_model']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    standard_distances, standard_failure_count, parallel_distances, parallel_failure_count = tests.parallelization_test(
        foolbox_model, loader, attack, p, batch_worker, attack_workers)

    standard_average_distance, standard_median_distance, standard_success_rate, standard_adjusted_median_distance = utils.distance_statistics(
        standard_distances, standard_failure_count)
    parallel_average_distance, parallel_median_distance, parallel_success_rate, parallel_adjusted_median_distance = utils.distance_statistics(
        parallel_distances, parallel_failure_count)

    average_distance_difference = (
        parallel_average_distance - standard_average_distance) / standard_average_distance
    median_distance_difference = (
        parallel_median_distance - standard_median_distance) / standard_median_distance
    success_rate_difference = (
        parallel_success_rate - standard_success_rate) / standard_success_rate
    adjusted_median_distance_difference = (
        parallel_adjusted_median_distance - standard_adjusted_median_distance) / standard_adjusted_median_distance

    info = [['Average Distance Relative Difference', average_distance_difference],
            ['Median Distance Relative Difference', median_distance_difference],
            ['Success Rate Difference', success_rate_difference],
            ['Adjusted Median Distance Difference', adjusted_median_distance_difference]]

    header = ['Standard Distances', 'Parallel Distances']

    utils.save_results(results_path, [standard_distances, parallel_distances], command,
                       info=info, header=header, transpose=True)


@main.command
@parsing.global_options
def train_model(options):
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    torch_model = options['torch_model']
    results_path = options['results_path']

    training.train_torch(torch_model, loader)


if __name__ == '__main__':
    main()
