import click
import foolbox
import numpy as np

from counter_attack import tests, utils
from counter_attack.cli import definitions, options, parsing


@click.command()
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('adversarial_perturbation')
@options.attack_options(definitions.supported_attacks)
@options.distance_options
@options.counter_attack_options(True)
def adversarial_perturbation(options):
    attack_name = options['attack_name']
    attack_p = options['attack_p']
    attack_workers = options['attack_workers']
    command = options['command']
    cuda = options['cuda']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    attack = parsing.parse_attack(
        attack_name, attack_p, criterion)

    distance_tool = parsing.parse_distance_tool('counter-attack', options, np.inf)

    samples_count, correct_count, successful_count, \
    correct_estimate_count, boundary_distances,\
    adversarial_distances = tests.adversarial_perturbation_test(foolbox_model, loader, attack, distance_tool, cuda, attack_workers)

    correct_estimate_rate = correct_estimate_count / successful_count
    effective_correct_estimate_rate = correct_estimate_count / correct_count

    info = [
        ['Total Count', samples_count],
        ['Correctly Classified Count', correct_count],
        ['Successful Attack Count', successful_count],
        ['Correct Estimate Count', correct_estimate_count],
        ['Correct Estimate Rate (correct_estimate / successful_attack)', '{:2.2f}%'.format(correct_estimate_rate * 100.0)],
        ['Effective Correct Estimate Rate (correct_estimate / correct_classification)', '{:2.2f}%'.format(effective_correct_estimate_rate * 100.0)]
    ]

    header = ['Boundary Distances', 'Adversarial Distances']

    utils.save_results(results_path, table=[boundary_distances, adversarial_distances], header=header, command=command, info=info)
