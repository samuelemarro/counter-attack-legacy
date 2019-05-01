import click
import foolbox

from OffenseDefense import defenses, tests, utils
from OffenseDefense.cli import definitions, options, parsing

@click.group(name='preprocessor')
def preprocessor_defense():
    """
    Defends using a "preprocessor", which modifies
    the image before passing it to the standard model.
    """
    pass


@preprocessor_defense.command(name='shallow')
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/preprocessor/shallow')
@options.preprocessor_options
@options.attack_options(definitions.supported_attacks)
def shallow_preprocessor(options):
    """
    Simply evaluates the effectiveness of the preprocessor defense, without additional
    attack strategies.
    
    Adversarial samples are generated to fool the undefended model.
    """
    attack_distance_measure = options['attack_distance_measure']
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
        attack_name, attack_distance_measure, foolbox_model, criterion)

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    samples_count, correct_count, successful_attack_count, distances = tests.shallow_defense_test(foolbox_model, loader, attack, attack_distance_measure,
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
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/preprocessor/substitute')
@options.preprocessor_options
@options.attack_options(definitions.supported_attacks)
@options.substitute_options
def substitute_preprocessor(options):
    """
    Uses BPDA with a substitute model to evade the preprocessor defense.

    BPDA uses predictions from the defended model and gradients
    from the substitute model.
    """
    attack_distance_measure = options['attack_distance_measure']
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
        attack_name, attack_distance_measure, composite_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(composite_model, loader, attack, attack_distance_measure,
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
@options.global_options
@options.dataset_options('test', 'test')
@options.standard_model_options
@options.pretrained_model_options
@options.test_options('defense/preprocessor/black-box')
@options.preprocessor_options
@options.attack_options(definitions.black_box_attacks)
def black_box_preprocessor(options):
    """
    Uses a black box attack to evade the preprocessor defense.

    Adversarial samples are generated to fool the defended model,
    which only provides the labels when queried.
    """
    attack_distance_measure = options['attack_distance_measure']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']
    preprocessor = options['preprocessor']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)

    criterion = foolbox.criteria.Misclassification()

    # The attack will be against the defended model
    attack = parsing.parse_attack(
        attack_name, attack_distance_measure, defended_model, criterion)

    samples_count, correct_count, successful_attack_count, distances, _, _ = tests.attack_test(defended_model, loader, attack, attack_distance_measure,
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