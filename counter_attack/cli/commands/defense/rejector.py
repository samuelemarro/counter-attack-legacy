import click
import foolbox

from counter_attack import rejectors, tests, utils
from counter_attack.cli import definitions, parsing, options

@click.group(name='rejector')
def rejector_defense():
    """
    Defends using a "rejector", which is a tool (usually a detector) that rejects suspected adversarial samples.
    """
    pass


@rejector_defense.command(name='shallow')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/rejector/shallow')
@options.attack_options(definitions.supported_attacks)
@options.distance_options
@options.counter_attack_options(False)
@options.detector_options
@options.rejector_options
def shallow_rejector(options):
    """
    Simply evaluates the effectiveness of the rejector defense, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the undefended model.
    """
    attack_lp_distance = options['attack_lp_distance']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    cuda = options['cuda']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    rejector = options['rejector']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the undefended model

    attack = parsing.parse_attack(
        attack_name, attack_lp_distance, criterion)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_rejector_test(
        foolbox_model, loader, attack, attack_lp_distance, rejector, cuda, attack_workers)

    info = utils.attack_statistics_info(samples_count, correct_count, successful_attack_count, distances)

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@rejector_defense.command(name='substitute')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/rejector/substitute')
@options.distance_options
@options.counter_attack_options(False)
@options.detector_options
@options.rejector_options
@options.attack_options(definitions.supported_attacks)
@options.substitute_options
def substitute_rejector(options):
    pass

@rejector_defense.command(name='black-box')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/rejector/black-box')
@options.attack_options(definitions.black_box_attacks)
@options.distance_options
@options.counter_attack_options(False)
@options.detector_options
@options.rejector_options
def black_box_rejector(options):
    """
    Uses a black box attack to evade the rejector defense.

    Adversarial samples are generated to fool the defended model,
    which only provides the labels when queried.
    Note: Models with rejectors also have a special label 'reject',
    which does not represent a valid misclassification (i.e. the attack
    does not considered being rejected a success).
    """
    attack_lp_distance = options['attack_lp_distance']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    cuda = options['cuda']
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
        attack_name, attack_lp_distance, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(
        defended_model, loader, attack, attack_lp_distance, cuda, attack_workers, name='Black-Box Rejector Attack')

    info = utils.attack_statistics_info(samples_count, correct_count, successful_attack_count, distances)

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)