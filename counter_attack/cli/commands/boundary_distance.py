import logging

import click
import numpy as np

from counter_attack import distance_tools, tests, utils
from counter_attack.cli import definitions, parsing, options

logger = logging.getLogger(__name__)

@click.command()
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('boundary_distance')
@options.distance_options
@options.counter_attack_options(True)
@click.argument('max_radius', type=float)
@click.option('--generation-workers', type=click.IntRange(0, None), default=0, show_default=True,
                help='The number of attack workers that will be used to generate the starting points. 0 disables parallelisation.')
def boundary_distance(options, max_radius, generation_workers):
    command = options['command']
    cuda = options['cuda']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    distance_tool = parsing.parse_distance_tool('counter-attack', options, np.inf)

    samples_count, consistent_count, failure_count, inconsistent_differences = tests.boundary_distance_test(foolbox_model, loader, distance_tool, max_radius, cuda, generation_workers, name='Boundary Distance Consistency Test')

    failure_rate = failure_count / samples_count
    consistency_rate = consistent_count / samples_count
    effective_consistency_rate = consistent_count / (samples_count - failure_count)
    average_difference = np.average(inconsistent_differences)
    median_difference = np.median(inconsistent_differences)

    info = [
        ['Total Samples', samples_count],
        ['Failed Measure Samples', failure_count],
        ['Consistent Sample', consistent_count],
        ['Failure Rate', '{:.2f}%'.format(failure_rate * 100.0)],
        ['Consistency Rate', '{:.2f}%'.format(consistency_rate * 100.0)],
        ['Effective Consistency Rate', '{:.2f}%'.format(effective_consistency_rate * 100.0)],
        ['Average Difference', '{:2.2e}'.format(average_difference)],
        ['Median Difference', '{:2.2e}'.format(median_difference)]
        ]

    header = ['Inconsistent Differences']
    utils.save_results(results_path, table=[inconsistent_differences], header=header, command=command, info=info)