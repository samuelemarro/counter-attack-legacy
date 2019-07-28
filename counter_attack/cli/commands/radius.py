import click
import numpy as np

from counter_attack import tests, utils
from counter_attack.cli import options, parsing

@click.command()
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('radius')
@options.distance_options
@options.counter_attack_options(True)
@click.argument('sampling_count', type=click.IntRange(1, None))
def radius(options, sampling_count):
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    distance_tool = parsing.parse_distance_tool('counter-attack', options, np.inf)

    total_count, consistent_count, failures = tests.radius_test(foolbox_model, loader, distance_tool, sampling_count)
    consistency_rate = consistent_count / total_count

    info = [
        ['Total Samples', total_count],
        ['Failures', failures],
        ['Consistent Sample', consistent_count],
        ['Consistency Rate', '{:.2f}%'.format(consistency_rate * 100.0)]
        ]

    utils.save_results(results_path, command=command, info=info)