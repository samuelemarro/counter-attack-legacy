import logging

import click
import foolbox

from counter_attack import tests, utils
from counter_attack.cli import definitions, parsing, options

logger = logging.getLogger(__name__)

@click.command()
@options.global_options
@options.standard_model_options
@options.pretrained_model_options
@options.dataset_options('test')
@options.test_options('attack')
@options.attack_options(definitions.supported_attacks)
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
    cuda = options['cuda']
    dataset_type = options['dataset_type']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    attack = parsing.parse_attack(
        attack_name, attack_p, criterion)

    save_adversarials = adversarial_dataset_path is not None

    if dataset_type == 'test' and save_adversarials and not no_test_warning:
        logger.warning('Remember to use \'--dataset-type train\' if you plan to use the generated adversarials '
                       'to train or calibrate an adversarial detector. You can disable this warning by passing '
                       '\'--no-test-warning\'.')

    samples_count, correct_count, successful_attack_count, distances, adversarials, adversarial_ground_truths = tests.attack_test(foolbox_model, loader, attack, attack_p,
                                                                                                                                  cuda, attack_workers, save_adversarials=save_adversarials)

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