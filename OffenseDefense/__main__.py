import logging

import click
import foolbox
import numpy as np
import sklearn
import torch

import OffenseDefense
import OffenseDefense.attacks as attacks
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.defenses as defenses
import OffenseDefense.detectors as detectors
import OffenseDefense.distance_tools as distance_tools
import OffenseDefense.loaders as loaders
import OffenseDefense.model_tools as model_tools
import OffenseDefense.parsing as parsing
import OffenseDefense.tests as tests
import OffenseDefense.training as training
import OffenseDefense.utils as utils

logger = logging.getLogger('OffenseDefense')


@click.group()
def main(*args):
    logging.basicConfig()


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('attack')
@parsing.parallelization_options
@parsing.attack_options(parsing.supported_attacks)
@click.option('-sdp', '--saved_dataset_path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the .zip file where the adversarial samples will be saved. If unspecified, no adversarial samples will be saved.')
@click.option('--no-test-warning', '-ntw', is_flag=True,
              help='Disables the warning for running this test on the test set.')
def attack(options, saved_dataset_path, no_test_warning):
    """
    Runs an attack against the model.

    \b
    Stores the following results:
        Success Rate: The success rate of the attack.
        Average Distance: The average L_p distance of the successful adversarial samples from their original samples.
        Median Distance: The median L_p distance of the successful adversarial samples from their original samples.
        Adjusted Median Distance: The median L_p distance of the adversarial samples from their original samples, treating failed attacks as samples with distance Infinity.
    """

    command = options['command']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    dataset_type = options['dataset_type']
    model_batch_worker = options['model_batch_worker']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    p = options['p']
    results_path = options['results_path']
    torch_model = options['torch_model']

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    save_adversarials = saved_dataset_path is not None

    if dataset_type == 'test' and save_adversarials and not no_test_warning:
        logger.warning('Remember to use \'-dt train\' if you plan to use the generated adversarials '
                       'to train or calibrate an adversarial detector. You can disable this warning by passing '
                       '\'--no-test-warning\' (alias: \'-ntw\').')

    distances, failure_count, adversarials, adversarial_ground_truths = tests.attack_test(foolbox_model, loader, attack, p,
                                                                                          model_batch_worker, attack_workers, save_adversarials=save_adversarials)

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header, transpose=True)

    if save_adversarials:
        dataset = list(
            zip(adversarials, adversarial_ground_truths)), success_rate
        utils.save_zip(dataset, saved_dataset_path)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('accuracy')
@click.option('-tk', '--top-ks', nargs=2, type=click.Tuple([int, int]), default=(1, 5), show_default=True,
              help='The two top-k accuracies that will be computed.')
def accuracy(options, top_ks):
    """
    Computes the accuracy of the model.

    \b
    Stores the following results:
        Top-K Accuracies: The accuracies, where k values are configurable with --top-ks.
    """

    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    torch_model = options['torch_model']

    accuracies = tests.accuracy_test(
        foolbox_model, loader, top_ks)

    info = [['Top-{} Accuracy:'.format(top_k), accuracy]
            for top_k, accuracy in zip(top_ks, accuracies)]
    utils.save_results(results_path, command=command, info=info)


@main.command()
@parsing.global_options
@parsing.dataset_options('test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('detect')
@parsing.parallelization_options
@parsing.detector_options(-np.Infinity)
@parsing.adversarial_dataset_options
@click.option('-sdp', '--saved_dataset_path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the .zip file where the scores will be saved with their corresponding images. If unspecified, no scores will be saved.')
@click.option('--no-test-warning', '-ntw', is_flag=True,
              help='Disables the warning for running this test on the test set.')
def detect(options, saved_dataset_path, no_test_warning):
    """
    Uses a detector to identify adversarial samples.
    """

    adversarial_loader = options['adversarial_loader']
    batch_size = options['batch_size']
    command = options['command']
    dataset_type = options['dataset_type']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    results_path = options['results_path']
    shuffle = options['shuffle']
    torch_model = options['torch_model']

    save_scores = saved_dataset_path is not None

    if dataset_type == 'test' and not no_test_warning:
        logger.warning('Remember to use \'-dt train\' if you plan to use the results '
                       'to pick a threshold for other tests. You can disable this warning by passing '
                       '\'--no-test-warning\' (alias: \'-ntw\').')

    genuine_scores, adversarial_scores, genuine_samples, adversarial_samples = tests.standard_detector_test(
        foolbox_model, genuine_loader, adversarial_loader, detector, save_scores)

    false_positive_rates, true_positive_rates, thresholds = utils.roc_curve(
        genuine_scores, adversarial_scores)

    best_threshold, best_tpr, best_fpr = utils.get_best_threshold(
        true_positive_rates, false_positive_rates, thresholds)
    area_under_curve = sklearn.metrics.auc(
        false_positive_rates, true_positive_rates)

    info = [['ROC AUC', '{:2.2f}%'.format(area_under_curve * 100.0)],
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

    utils.save_results(results_path, table=columns, command=command,
                       info=info, header=header, transpose=True)

    if save_scores:
        # Remove failures
        failure_value = -np.Infinity

        genuine_samples, genuine_scores = utils.filter_lists(
            lambda _, score: score is not failure_value, genuine_samples, genuine_scores)

        adversarial_samples, adversarial_scores = utils.filter_lists(
            lambda _, score: score is not failure_value, adversarial_samples, adversarial_scores)

        genuine_list = zip(genuine_samples, genuine_scores)
        adversarial_list = zip(adversarial_samples, adversarial_scores)

        dataset = (genuine_list, adversarial_list)

        utils.save_zip(dataset, saved_dataset_path)


@main.group()
def defense():
    pass


@defense.group(name='detector')
def detector_defense():
    pass


@detector_defense.command(name='shallow')
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/detector/shallow')
@parsing.parallelization_options
@parsing.detector_options(-np.Infinity)
@parsing.adversarial_dataset_options
@click.argument('threshold', type=float)
def shallow_detector(options, threshold):
    adversarial_loader = options['adversarial_loader']
    adversarial_generation_success_rate = options['adversarial_generation_success_rate']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    results_path = options['results_path']

    evasion_success_rate = tests.shallow_attack_test(
        foolbox_model, adversarial_loader, detector, threshold)

    success_rate = adversarial_generation_success_rate * evasion_success_rate

    info = [
        ['Adversarial Generation Success Rate', '{:2.2f}%'.format(
            adversarial_generation_success_rate * 100.0)],
        ['Evasion Success Rate', '{:2.2f}%'.format(
            evasion_success_rate * 100.0)],
        ['Final Success Rate', '{:2.2f}%'.format(success_rate * 100.0)]
    ]

    utils.save_results(results_path, info=info)


@defense.group(name='model')
def model_defense():
    pass


@model_defense.command(name='shallow')
@parsing.global_options
@parsing.custom_model_options
@parsing.test_options('defense/model/shallow')
@parsing.adversarial_dataset_options
def shallow_model(options):
    adversarial_loader = options['adversarial_loader']
    adversarial_generation_success_rate = options['adversarial_generation_success_rate']
    device = options['device']
    foolbox_model = options['foolbox_model']
    num_classes = options['num_classes']
    results_path = options['results_path']

    evasion_success_rate = tests.shallow_attack_test(
        foolbox_model, adversarial_loader)

    success_rate = adversarial_generation_success_rate * evasion_success_rate

    info = [
        ['Adversarial Generation Success Rate', '{:2.2f}%'.format(
            adversarial_generation_success_rate * 100.0)],
        ['Evasion Success Rate', '{:2.2f}%'.format(
            evasion_success_rate * 100.0)],
        ['Final Success Rate', '{:2.2f}%'.format(success_rate * 100.0)]
    ]

    utils.save_results(results_path, info=info)


@defense.group(name='preprocessor')
def preprocessor_defense(name='preprocessor'):
    pass

# TODO: batch_attack's parallelization uses the Torch Model instead of the foolbox one (which might contain the detector and the preprocessors)


@preprocessor_defense.command(name='shallow')
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/preprocessor/shallow')
@parsing.preprocessor_options
@parsing.adversarial_dataset_options
def shallow_preprocessor(options):
    adversarial_loader = options['adversarial_loader']
    adversarial_generation_success_rate = options['adversarial_generation_success_rate']
    foolbox_model = options['foolbox_model']
    results_path = options['results_path']
    preprocessor = options['preprocessor']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    evasion_success_rate = tests.shallow_attack_test(
        defended_model, adversarial_loader)

    success_rate = adversarial_generation_success_rate * evasion_success_rate

    info = [
        ['Adversarial Generation Success Rate', '{:2.2f}%'.format(
            adversarial_generation_success_rate * 100.0)],
        ['Evasion Success Rate', '{:2.2f}%'.format(
            evasion_success_rate * 100.0)],
        ['Final Success Rate', '{:2.2f}%'.format(success_rate * 100.0)]
    ]

    utils.save_results(results_path, info=info)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('black_box_evasion')
@parsing.parallelization_options
@parsing.attack_options(parsing.black_box_attacks)
@parsing.detector_options(-np.Infinity)
def black_box_evasion(options):
    """
    Treats the model and the detector as a black box.
    """

    attack_name = options['attack_name']
    model_batch_worker = options['model_batch_worker']
    command = options['command']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    attack_workers = options['attack_workers']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']
    threshold = options['threshold']
    torch_model = options['torch_model']

    composite_model = detectors.CompositeDetectorModel(
        foolbox_model, detector, threshold)

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        composite_model, foolbox.criteria.Misclassification() and detectors.Undetected(), distance_tools.LpDistance(p))

    distances, failure_count, _, _ = tests.attack_test(composite_model, loader, attack,
                                                       p, model_batch_worker, attack_workers, name='Black Box Evasion Test')

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('differentiable_evasion')
@parsing.parallelization_options
@parsing.attack_options(parsing.differentiable_attacks)
@parsing.detector_options(-np.Infinity)
def differentiable_evasion(options):
    attack_name = options['attack_name']
    model_batch_worker = options['model_batch_worker']
    command = options['command']
    detector = options['detector']
    foolbox_model = options['foolbox_model']
    attack_workers = options['attack_workers']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']
    threshold = options['threshold']
    torch_model = options['torch_model']

    # composite_model = detectors.CompositeDetectorModel(
    #    foolbox_model, detector, threshold)

    # attack_constructor = get_attack_constructor(attack_name, p)
    # attack = attack_constructor(
    #    composite_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    #TODO: Complete

    composite_model = None
    distances, failure_count, _, _ = tests.attack_test(composite_model, loader, attack,
                                                       p, model_batch_worker, attack_workers, name='Differentiable Evasion Test')

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(adjusted_median_distance)]]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('parallelization')
@parsing.set_parameters({'enable_attack_parallelization': True})
@parsing.attack_options(parsing.parallelizable_attacks)
@click.option('-aw', '--attack-workers', default=5, show_default=True, type=click.IntRange(1, None),
              help='The number of parallel workers that will be used to speed up the attack.')
def parallelization(options, attack_workers):
    """
    Compares parallelized attacks with standard ones.
    """

    attack_name = options['attack_name']
    command = options['command']
    foolbox_model = options['foolbox_model']
    p = options['p']
    results_path = options['results_path']
    loader = options['loader']
    torch_model = options['torch_model']

    model_batch_worker = batch_attack.TorchWorker(torch_model)

    attack_constructor = parsing.parse_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    standard_distances, standard_failure_count, parallel_distances, parallel_failure_count = tests.parallelization_test(
        foolbox_model, loader, attack, p, model_batch_worker, attack_workers)

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

    utils.save_results(results_path, table=[standard_distances, parallel_distances], command=command,
                       info=info, header=header, transpose=True)


@main.command()
@parsing.global_options
@parsing.dataset_options('train')
@parsing.train_options
@click.option('-tmp', '--trained-model-path', type=click.Path(file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the model will be saved. If unspecified, it defaults to \'./train_model/$dataset$ $start_time$.pth.tar\'')
def train_model(options, trained_model_path):
    command = options['command']
    cuda = options['cuda']
    dataset = options['dataset']
    epochs = options['epochs']
    loader = options['loader']
    optimizer = options['optimizer']
    start_time = options['start_time']

    if trained_model_path is None:
        trained_model_path = parsing.get_training_default_path(
            'train_model', dataset, start_time)

    torch_model = parsing._get_torch_model(dataset)
    torch_model.train()

    loss = torch.nn.CrossEntropyLoss()

    training.train_torch(torch_model, loader, loss,
                         optimizer, epochs, cuda, classification=True)

    model_tools.save_state_dict(torch_model, trained_model_path)


@main.command()
@parsing.global_options
@parsing.train_options
@click.option('-l', '--loss', type=click.Choice(['l1', 'l2', 'smooth_l1']), default='l2', show_default=True)
@click.option('-tap', '--trained-approximator-path', type=click.Path(file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the approximator will be saved. If unspecified, it defaults to \'./train_approximator/$dataset$ $start_time$.pth.tar\'')
def train_approximator(options, loss, trained_approximator_path):
    cuda = options['cuda']
    dataset = options['dataset']
    detector = options['detector']
    epochs = options['epochs']
    loader = options['loader']
    optimizer = options['optimizer']
    start_time = options['start_time']
    torch_model = options['torch_model']

    if trained_approximator_path is None:
        trained_approximator_path = parsing.get_training_default_path(
            'train_approximator', dataset, start_time)

    if loss == 'l1':
        loss = torch.nn.L1Loss()
    elif loss == 'l2':
        loss = torch.nn.MSELoss()
    elif loss == 'smooth_l1':
        loss = torch.nn.SmoothL1Loss()
    else:
        raise ValueError('Loss not supported.')

    training.train_torch(torch_model, loader, loss,
                         optimizer, epochs, cuda, classification=False)

    model_tools.save_state_dict(torch_model, trained_approximator_path)


if __name__ == '__main__':
    main()
