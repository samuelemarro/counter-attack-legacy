import logging

import click
import foolbox
import numpy as np
import pathlib
import sklearn
import sys
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
import OffenseDefense.rejectors as rejectors
import OffenseDefense.tests as tests
import OffenseDefense.training as training
import OffenseDefense.utils as utils

logger = logging.getLogger('OffenseDefense')

# TODO: Test preprocessing options
# TODO: Allow for optional model weights?
# TODO: Check that the pretrained model does not contain normalisation inside?
# TODO: British vs American spelling
# TODO: Download the cifar100 weights for densenet-bc-100-12 (when available)
# TODO: Upload both of them and update the links in config.ini
# TODO: Sanity check: Difference between the original model and its trained approximator
# TODO: Verify that the transfer is successful
# TODO: Complete the substitutes
# TODO: In pretrained_model, you are passing the model path, not the weights one
# TODO: standard and parallel are treated completely differently, and might have different models or attacks
# TODO: Not all rejectors use [2|inf] from distance_tool_options. Merge distance_tool with counter_attack?
# TODO: preprocessor and model, from a defense point of view, are the same. The only difference is the arguments.
# I could theoretically load a foolbox model and use it, no matter what it contains.
# TODO: MSE(x) is (L2(x)^2)/n, but foolbox uses MSE when it talks about L2.
# TODO: Foolbox uses normalized distances in the bounds (d / (max_ - min_)) which are often averaged (d / n, L-inf is an exception)
# Should I use the same approach? I don't like averaging, as it has nothing to do with the norm, but it is useful for comparing
# different image sizes. I think I will have to add both normalization and averaging.
# TODO: RandomDirectionAttack passes arbitrary distance and threshold
# TODO: Can I make WrappedLpDistance less hacky?
# TODO: Finish bound normalization in LpDistance


# IMPORTANT:
# Shallow attacks the standard model, then it is evaluated on the defended model
# Substitute and Black-Box attack the defended model
# This means that you cannot write the sanity check "Shallow is the same as
# a Substitute that uses the original as gradient estimator"

# Note: Not all adversarial attacks are successful. This means that an approximation
# dataset will have slightly less adversarial samples. This unbalanced dataset should
# not cause problems when training approximators, but it might cause problems when
# training adversarial classifiers.

# Note: When preparing the adversarial dataset, the model accuracy will always be around 100%
# because you're running the attack on the train set

# Note: For ImageNet, we evaluate on the validation set

@click.group()
def main(*args):
    logging.basicConfig()
    logging.captureWarnings(True)

    # Print the messages to console
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    root.addHandler(handler)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('attack')
@parsing.attack_options(parsing.supported_attacks)
@click.option('--adversarial-dataset-path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the .zip file where the adversarial samples will be saved. If unspecified, no adversarial samples will be saved.')
@click.option('--no-test-warning', is_flag=True,
              help='Disables the warning for running this test on the test set.')
def attack(options, adversarial_dataset_path, no_test_warning):
    """
    Runs an attack against the model.

    \b
    Stores the following results:
        Success Rate: The success rate of the attack.
        Average Distance: The average L_p distance of the successful adversarial samples from their original samples.
        Median Distance: The median L_p distance of the successful adversarial samples from their original samples.
        Adjusted Median Distance: The median L_p distance of the adversarial samples from their original samples, treating failed attacks as samples with distance Infinity.
    """

    attack_p = options['attack_p']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    dataset_type = options['dataset_type']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    attack = parsing.parse_attack(
        attack_name, attack_p, foolbox_model, criterion)

    save_adversarials = adversarial_dataset_path is not None

    if dataset_type == 'test' and save_adversarials and not no_test_warning:
        logger.warning('Remember to use \'--dataset-type train\' if you plan to use the generated adversarials '
                       'to train or calibrate an adversarial detector. You can disable this warning by passing '
                       '\'--no-test-warning\'.')

    samples_count, correct_count, successful_attack_count, distances, adversarials, adversarial_ground_truths = tests.attack_test(foolbox_model, loader, attack, attack_p,
                                                                                                                                  attack_workers, save_adversarials=save_adversarials)

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    failure_count = correct_count - successful_attack_count
    average_distance, median_distance, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Base Accuracy', '{:2.2f}%'.format(accuracy * 100.0)],
            ['Success Rate', '{:2.2f}%'.format(success_rate * 100.0)],
            ['Average Distance', '{:2.2e}'.format(average_distance)],
            ['Median Distance', '{:2.2e}'.format(median_distance)],
            ['Adjusted Median Distance', '{:2.2e}'.format(
                adjusted_median_distance)],
            ['Samples Count', str(samples_count)],
            ['Correct Count', str(correct_count)],
            ['Successful Attack Count', str(successful_attack_count)]]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)

    if save_adversarials:
        dataset = list(
            zip(adversarials, adversarial_ground_truths)), success_rate
        utils.save_zip(dataset, adversarial_dataset_path)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test', 'test')
@parsing.test_options('accuracy')
@click.option('--top-ks', nargs=2, type=click.Tuple([int, int]), default=(1, 5), show_default=True,
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

    accuracies = tests.accuracy_test(
        foolbox_model, loader, top_ks)

    info = [['Top-{} Accuracy:'.format(top_k), '{:2.2f}%'.format(accuracy * 100.0)]
            for top_k, accuracy in zip(top_ks, accuracies)]
    utils.save_results(results_path, command=command, info=info)


@main.command()
@parsing.global_options
@parsing.dataset_options('test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('detector-roc')
@parsing.distance_tool_options
@parsing.counter_attack_options(False)
@parsing.detector_options
@parsing.adversarial_dataset_options
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

    true_positive_rates = ['{:2.2}%'.format(
        true_positive_rate) for true_positive_rate in true_positive_rates]
    false_positive_rates = ['{:2.2}%'.format(
        false_positive_rate) for false_positive_rate in false_positive_rates]

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


@main.group()
def defense():
    """
    Uses defenses against various attack strategies.
    """
    pass


@defense.group(name='rejector')
def rejector_defense():
    """
    Defends using a "rejector", which is a tool (usually a detector) that rejects adversarial samples.
    """
    pass


@rejector_defense.command(name='shallow')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/rejector/shallow')
@parsing.attack_options(parsing.supported_attacks)
@parsing.distance_tool_options
@parsing.counter_attack_options(False)
@parsing.detector_options
@parsing.rejector_options
def shallow_rejector(options):
    """
    Simply evaluates the effectiveness of the rejector defense, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the undefended model.
    """
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    attack_p = options['attack_p']
    rejector = options['rejector']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the undefended model
    attack = parsing.parse_attack(
        attack_name, attack_p, foolbox_model, criterion)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_rejector_test(
        foolbox_model, loader, attack, attack_p, rejector, attack_workers)

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@rejector_defense.command(name='black-box')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/rejector/black-box')
@parsing.attack_options(parsing.black_box_attacks)
@parsing.distance_tool_options
@parsing.counter_attack_options(False)
@parsing.detector_options
@parsing.rejector_options
def black_box_rejector(options):
    """
    Uses a black box attack to evade the rejector defense.

    Adversarial samples are generated to fool the defended model,
    which only provides the labels when queried.
    Note: Models with rejectors also have a special label 'reject',
    which does not represent a valid misclassification (i.e. the attack
    does not considered being rejected a success).
    """
    attack_name = options['attack_name']
    attack_p = options['attack_p']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    rejector = options['rejector']
    results_path = options['results_path']

    # The defended_model returns [y1, y2 ... yN, -inf] if it believes
    # that the sample is valid, otherwise it returns [0, 0 ... 0, 1]
    # This means that if the top label is the last one, it was classified as adversarial.
    # On a genuine dataset, this should never happen (if the rejector is perfect).

    defended_model = rejectors.RejectorModel(
        foolbox_model, rejector)

    # detectors.Undetected() adds the condition that the top label must not be the last
    # Note: (foolbox.Criterion and foolbox.Criterion) should give a combined criterion, but
    # apparently it doesn't work. The documentation recommends using "&"

    criterion = foolbox.criteria.CombinedCriteria(
        foolbox.criteria.Misclassification(), rejectors.Unrejected())

    # The attack will be against the defended model
    attack = parsing.parse_attack(
        attack_name, attack_p, defended_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(
        defended_model, loader, attack, attack_p, attack_workers, name='Black-Box Rejector Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@defense.group(name='model')
def model_defense():
    """
    Defends using a custom model, for example a model that
    has been trained to be more robust.
    """
    pass


@model_defense.command(name='shallow')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/model/shallow')
@parsing.custom_model_options
@parsing.attack_options(parsing.supported_attacks)
def shallow_model(options):
    """
    Simply evaluates the effectiveness of the custom model, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the standard model.
    """
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    attack_p = options['attack_p']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the undefended model
    attack = parsing.parse_attack(
        attack_name, attack_p, foolbox_model, criterion)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_defense_test(
        foolbox_model, loader, attack, attack_p, custom_foolbox_model, attack_workers, name='Shallow Model Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@model_defense.command(name='substitute')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.test_options('defense/model/substitute')
@parsing.custom_model_options
@parsing.attack_options(parsing.supported_attacks)
@parsing.substitute_options
def substitute_model(options):
    """
    Uses BPDA with a substitute model to attack the custom model.

    BPDA uses predictions from the defended model and gradients
    from the substitute model.
    Note: We could technically attack the custom model directly,
    since most models support gradient computation, but we are
    assuming that we do not have access to the gradients. 
    """
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    loader = options['loader']
    attack_p = options['attack_p']
    results_path = options['results_path']
    substitute_foolbox_model = options['substitute_foolbox_model']

    composite_model = foolbox.models.CompositeModel(custom_foolbox_model, substitute_foolbox_model)

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the substitute model with estimated gradients
    attack = parsing.parse_attack(
        attack_name, attack_p, composite_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(composite_model, loader, attack, attack_p,
                                                                                               attack_workers, name='Substitute Model Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@model_defense.command(name='black-box')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.test_options('defense/model/black-box')
@parsing.custom_model_options
@parsing.attack_options(parsing.black_box_attacks)
def black_box_model(options):
    """
    Uses a black box attack against the custom model.

    Adversarial samples are generated to fool the custom model,
    which only provides the labels when queried.

    Note: We could technically use the gradients,
    since most models support gradient computation, but we are
    assuming that we do not have access to them. 
    """
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    loader = options['loader']
    attack_p = options['attack_p']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the defended (custom) model
    attack = parsing.parse_attack(
        attack_name, attack_p, custom_foolbox_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(
        custom_foolbox_model, loader, attack, attack_p, attack_workers, name='Black-Box Model Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@defense.group(name='preprocessor')
def preprocessor_defense():
    """
    Defend using a "preprocessor", which modifies the image before passing it to the model.
    """
    pass


@preprocessor_defense.command(name='shallow')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/preprocessor/shallow')
@parsing.preprocessor_options
@parsing.attack_options(parsing.supported_attacks)
def shallow_preprocessor(options):
    """
    Simply evaluates the effectiveness of the preprocessor defense, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the undefended model.
    """
    attack_p = options['attack_p']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    preprocessor = options['preprocessor']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the undefended model
    attack = parsing.parse_attack(
        attack_name, attack_p, foolbox_model, criterion)

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_defense_test(foolbox_model, loader, attack, attack_p,
                                                                                                defended_model, attack_workers,
                                                                                                name='Shallow Preprocessor Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)

@preprocessor_defense.command(name='substitute')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/preprocessor/substitute')
@parsing.preprocessor_options
@parsing.attack_options(parsing.supported_attacks)
@parsing.substitute_options
def substitute_preprocessor(options):
    """
    Uses BPDA with a substitute model to evade the preprocessor defense.

    BPDA uses predictions from the defended model and gradients
    from the substitute model.
    """
    attack_p = options['attack_p']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    preprocessor = options['preprocessor']
    substitute_foolbox_model = options['substitute_foolbox_model']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    composite_model = foolbox.models.CompositeModel(defended_model, substitute_foolbox_model)

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the defended model with estimated gradients
    attack = parsing.parse_attack(
        attack_name, attack_p, composite_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(composite_model, loader, attack, attack_p,
                                                                                               attack_workers, name='Substitute Preprocessor Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)

@preprocessor_defense.command(name='black-box')
@parsing.global_options
@parsing.dataset_options('test', 'test')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.test_options('defense/preprocessor/black-box')
@parsing.preprocessor_options
@parsing.attack_options(parsing.black_box_attacks)
def black_box_preprocessor(options):
    """
    Uses a black box attack to evade the preprocessor defense.

    Adversarial samples are generated to fool the defended model,
    which only provides the labels when queried.
    """
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    attack_p = options['attack_p']
    preprocessor = options['preprocessor']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the defended model
    attack = parsing.parse_attack(
        attack_name, attack_p, defended_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(defended_model, loader, attack, attack_p,
                                                                                               attack_workers, name='Black-Box Preprocessor Attack')

    accuracy = correct_count / samples_count
    success_rate = successful_attack_count / correct_count

    info = [
        ['Base Accuracy', '{:2.2f}%'.format(
            accuracy * 100.0)],
        ['Base Attack Success Rate', '{:2.2f}%'.format(
            success_rate * 100.0)],
        ['Samples Count', str(samples_count)],
        ['Correct Count', str(correct_count)],
        ['Successful Attack Count', str(successful_attack_count)]
    ]

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@main.command()
@parsing.global_options
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.dataset_options('test')
@parsing.test_options('parallelization')
@parsing.attack_options(parsing.parallelizable_attacks, mandatory_parallelization=True)
def parallelization(options):
    """
    Compares parallelized attacks with standard ones.

    This is a sanity check to verify that attack parallelization does not seriously
    affect the results.
    """

    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    attack_p = options['attack_p']
    results_path = options['results_path']
    loader = options['loader']

    criterion = foolbox.criteria.Misclassification()

    attack = parsing.parse_attack(
        attack_name, attack_p, foolbox_model, criterion)

    samples_count, correct_count, standard_attack_count, parallel_attack_count, standard_distances, parallel_distances = tests.parallelization_test(
        foolbox_model, loader, attack, attack_p, attack_workers)

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

    info = [['Average Distance Relative Difference', average_distance_difference],
            ['Median Distance Relative Difference', median_distance_difference],
            ['Success Rate Difference', success_rate_difference],
            ['Adjusted Median Distance Difference',
                adjusted_median_distance_difference],
            ['Samples Count', str(samples_count)],
            ['Correct Count', str(correct_count)],
            ['Standard Attack Count', str(standard_attack_count)],
            ['Parallel Attack Count', str(parallel_attack_count)]]

    header = ['Standard Distances', 'Parallel Distances']

    utils.save_results(results_path, table=[standard_distances, parallel_distances], command=command,
                       info=info, header=header)


@main.command()
@parsing.global_options
@parsing.dataset_options('train', 'train')
@parsing.train_options
@click.option('--trained-model-path', type=click.Path(file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the model will be saved. If unspecified, it defaults to \'./train_model/$dataset $start_time.pth.tar\'')
def train_model(options, trained_model_path):
    cuda = options['cuda']
    dataset = options['dataset']
    epochs = options['epochs']
    loader = options['loader']
    optimiser_name = options['optimiser_name']
    start_time = options['start_time']

    if trained_model_path is None:
        trained_model_path = parsing.get_training_default_path(
            'train_model', dataset, start_time)

    torch_model = parsing.get_torch_model(dataset)
    torch_model.train()

    if cuda:
        torch_model.cuda()

    optimiser = parsing.build_optimiser(optimiser_name, torch_model.parameters(), options)

    loss = torch.nn.CrossEntropyLoss()

    training.train_torch(torch_model, loader, loss,
                         optimiser, epochs, cuda, classification=True)

    torch.save(torch_model, trained_model_path)

@main.group(name='approximation-dataset')
def approximation_dataset(*args, **kwargs):
    pass

@approximation_dataset.command(name='preprocessor')
@parsing.global_options
@parsing.dataset_options('train', 'train')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.preprocessor_options
@parsing.adversarial_dataset_options
@parsing.approximation_dataset_options('preprocessor')
def approximation_dataset_preprocessor(options):
    """
    Generates the dataset to train a substitute model for models
    with preprocessors.

    Saves the labels predicted by the defended model, using the genuine
    dataset + an adversarial dataset. 
    """
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    preprocessor = options['preprocessor']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)


    genuine_approximation_dataset = training.generate_approximation_dataset(defended_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(defended_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)

@approximation_dataset.command(name='model')
@parsing.global_options
@parsing.dataset_options('train', 'train')
@parsing.standard_model_options
@parsing.custom_model_options
@parsing.adversarial_dataset_options
@parsing.approximation_dataset_options('model')
def approximation_dataset_model(options):
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    custom_foolbox_model = options['custom_foolbox_model']
    genuine_loader = options['loader']

    genuine_approximation_dataset = training.generate_approximation_dataset(custom_foolbox_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(custom_foolbox_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)

@approximation_dataset.command(name='rejector')
@parsing.global_options
@parsing.dataset_options('train', 'train')
@parsing.standard_model_options
@parsing.pretrained_model_options
@parsing.distance_tool_options
@parsing.counter_attack_options(False)
@parsing.detector_options
@parsing.rejector_options
@parsing.adversarial_dataset_options
@parsing.approximation_dataset_options('rejector')
def approximation_dataset_rejector(options):
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    rejector = options['rejector']

    defended_model = rejectors.RejectorModel(foolbox_model, rejector)

    genuine_approximation_dataset = training.generate_approximation_dataset(defended_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(defended_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)

@main.command()
@parsing.global_options
@parsing.train_options
@click.argument('defense_type', type=click.Choice(['model', 'preprocessor', 'rejector']))
@click.argument('approximation_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--base-weights-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the file where the base weights are stored. If unspecified, it defaults to the pretrained model for the dataset.')
@click.option('--normalisation', default=None,
            help='The normalisation that will be applied by the model. Supports both dataset names ({}) and '
            'channel stds-means (format: "red_mean green_mean blue_mean red_stdev green_stdev blue_stdev" including quotes).'.format(', '.join(parsing.datasets)))
@click.option('--trained-approximator-path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the approximator will be saved. If unspecified, it defaults to \'./trained_models/train_approximator/$defense_type/$dataset $start_time.pth.tar\'')
def train_approximator(options, defense_type, approximation_dataset_path, base_weights_path, normalisation, trained_approximator_path):
    batch_size = options['batch_size']
    cuda = options['cuda']
    dataset = options['dataset']
    epochs = options['epochs']
    max_batches = options['max_batches']
    optimiser_name = options['optimiser_name']
    shuffle = options['shuffle']
    start_time = options['start_time']

    if base_weights_path is None:
        base_weights_path = './pretrained_models/' + dataset + '_weights.pth.tar'

    if trained_approximator_path is None:
        trained_approximator_path = parsing.get_training_default_path(
            'train_approximator/' + defense_type, dataset, start_time)

    is_rejector = defense_type == 'rejector'
    model = parsing.get_torch_model(dataset, is_rejector)
    model_tools.load_partial_state_dict(model, base_weights_path)
    last_layer = model_tools.get_last_layer(model)

    model = parsing.apply_normalisation(model, normalisation, 'model', '--normalisation')

    model.train()

    if cuda:
        model.cuda()

    # Reinitialise the last layer
    last_layer.reset_parameters()

    optimiser = parsing.build_optimiser(optimiser_name, last_layer.parameters(), options)

    # Build the pretrained model and the final model (which might be n+1)
    # Transfer some layers (which? how?)
    # Train the remaining layers of the final model
    # Save the model

    approximation_data = utils.load_zip(approximation_dataset_path)
    loader = loaders.ListLoader(approximation_data, batch_size, shuffle)

    if max_batches is not None:
        loader = loaders.MaxBatchLoader(loader, max_batches)

    training.train_torch(model, loader, torch.nn.CrossEntropyLoss(),
                         optimiser, epochs, cuda, classification=True)

    pathlib.Path(trained_approximator_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model, trained_approximator_path)


if __name__ == '__main__':
    main()
