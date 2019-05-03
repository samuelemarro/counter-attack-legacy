import click
import foolbox

from OffenseDefense import tests, utils
from OffenseDefense.cli import definitions, options, parsing

@click.group(name='model')
def model_defense():
    """
    Defends using a custom model, for example a model that
    has been trained to be more robust.
    """
    pass


@model_defense.command(name='shallow')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/model/shallow')
@options.custom_model_options
@options.attack_options(definitions.supported_attacks)
def shallow_model(options):
    """
    Simply evaluates the effectiveness of the custom model, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the standard model.
    """
    attack_distance_measure = options['attack_distance_measure']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the undefended model
    attack = parsing.parse_attack(
        attack_name, attack_distance_measure, foolbox_model, criterion)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_defense_test(
        foolbox_model, loader, attack, attack_distance_measure, custom_foolbox_model, attack_workers, name='Shallow Model Attack')

    info = utils.attack_statistics_info(samples_count, correct_count, successful_attack_count, distances)

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@model_defense.command(name='substitute')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.test_options('defense/model/substitute')
@options.custom_model_options
@options.attack_options(definitions.supported_attacks)
@options.substitute_options
def substitute_model(options):
    """
    Uses BPDA with a substitute model to attack the custom model.

    BPDA uses predictions from the defended model and gradients
    from the substitute model.
    Note: We could technically attack the custom model directly,
    since most models support gradient computation, but we are
    assuming that we do not have access to the gradients. 
    """
    attack_distance_measure = options['attack_distance_measure']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    substitute_foolbox_model = options['substitute_foolbox_model']

    if substitute_foolbox_model.num_classes() != custom_foolbox_model.num_classes():
        raise click.BadArgumentUsage('The substitute model ({} classes) must have the same '
        'number of classes as the custom model ({} classes)'.format(
            substitute_foolbox_model.num_classes(), custom_foolbox_model.num_classes()))

    composite_model = foolbox.models.CompositeModel(custom_foolbox_model, substitute_foolbox_model)

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the substitute model with estimated gradients
    attack = parsing.parse_attack(
        attack_name, attack_distance_measure, composite_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(composite_model, loader, attack, attack_distance_measure,
                                                                                               attack_workers, name='Substitute Model Attack')

    info = utils.attack_statistics_info(samples_count, correct_count, successful_attack_count, distances)

    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)


@model_defense.command(name='black-box')
@options.global_options
@options.dataset_options('test', 'test')
@options.test_options('defense/model/black-box')
@options.custom_model_options
@options.attack_options(definitions.black_box_attacks)
def black_box_model(options):
    """
    Uses a black box attack against the custom model.

    Adversarial samples are generated to fool the custom model,
    which only provides the labels when queried.

    Note: We could technically use the gradients,
    since most models support gradient computation, but we are
    assuming that we do not have access to them. 
    """
    attack_distance_measure = options['attack_distance_measure']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    custom_foolbox_model = options['custom_foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the defended (custom) model
    attack = parsing.parse_attack(
        attack_name, attack_distance_measure, custom_foolbox_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(
        custom_foolbox_model, loader, attack, attack_distance_measure, attack_workers, name='Black-Box Model Attack')

    info = utils.attack_statistics_info(samples_count, correct_count, successful_attack_count, distances)
    header = ['Distances']

    utils.save_results(results_path, table=[distances], command=command,
                       info=info, header=header)
