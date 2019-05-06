import click
import foolbox

from counter_attack import tests, utils
from counter_attack.cli import definitions, options, parsing


@click.group(name='check')
def check():
    pass

@check.command(name='parallelization')
@options.global_options
@options.standard_model_options
@options.pretrained_model_options
@options.dataset_options('test')
@options.test_options('check/parallelization')
@options.attack_options(definitions.parallelizable_attacks, mandatory_parallelization=True)
def check_parallelization(options):
    """
    Compares parallelized attacks with standard ones.

    This is a sanity check to verify that attack parallelization does not seriously
    affect the results.
    """

    attack_distance_measure = options['attack_distance_measure']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    cuda = options['cuda']
    foolbox_model = options['foolbox_model']
    results_path = options['results_path']
    loader = options['loader']

    criterion = foolbox.criteria.Misclassification()

    attack = parsing.parse_attack(
        attack_name, attack_distance_measure, criterion)

    samples_count, correct_count, standard_attack_count, parallel_attack_count, standard_distances, parallel_distances = tests.parallelization_test(
        foolbox_model, loader, attack, attack_distance_measure, cuda, attack_workers)

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