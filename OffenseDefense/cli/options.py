import datetime
import functools
import logging
import sys

import click
import foolbox
import numpy as np
import torch

from OffenseDefense import detectors, loaders, model_tools, rejectors, utils
from OffenseDefense.cli import parsing, definitions

logger = logging.getLogger(__name__)

def global_options(func):
    @click.argument('dataset', type=click.Choice(definitions.datasets))
    @click.option('--batch-size', default=5, show_default=True, type=click.IntRange(1, None),
                  help='The size of each batch.')
    @click.option('--max-model-batch-size', type=click.IntRange(0, None), default=0,
                  help='The maximum number of images passed in the same batch. 0 disables batch limiting (default).')
    @click.option('--max-batches', type=click.IntRange(1, None), default=None,
                  help='The maximum number of batches. If unspecified, no batch limiting is applied.')
    @click.option('--shuffle', type=bool, default=True, show_default=True,
                  help='Whether to shuffle the dataset.')
    @click.option('--config-path', default='./config.ini', type=click.Path(file_okay=True, exists=True),
                  help='The path to the configuration file.')
    @click.option('--no-cuda', is_flag=True)
    @click.option('--no-shuffle-warning', is_flag=True,
                  help='Disables the warning for limiting batches without shuffling.')
    @click.option('--log-level', default='info', show_default=True, type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
                  help='Sets the logging level.')
    @functools.wraps(func)
    def _parse_global_options(dataset, batch_size, max_model_batch_size, max_batches, shuffle, config_path, no_cuda, no_shuffle_warning, log_level, *args, **kwargs):
        start_time = datetime.datetime.now()

        command = ' '.join(sys.argv[1:])

        if max_batches is not None:
            if (not shuffle) and (not no_shuffle_warning):
                logger.warning('You are limiting the number of batches, but you aren\'t applying any shuffling. '
                               'This means that the last parts of your dataset will be never used. You can disable this '
                               'warning by passing \'--no-shuffle-warning\'.')

        num_classes = parsing.get_num_classes(dataset)

        logger.debug('CUDA is supported: {}'.format(torch.cuda.is_available()))

        cuda = torch.cuda.is_available() and not no_cuda

        device = torch.cuda.current_device() if cuda else 'cpu'

        logging.getLogger('OffenseDefense').setLevel(log_level.upper())

        logger.info('Batch size: {}'.format(batch_size))

        global_options = {
            'batch_size': batch_size,
            'command': command,
            'config_path': config_path,
            'cuda': cuda,
            'device': device,
            'dataset': dataset,
            'max_batches': max_batches,
            'max_model_batch_size' : max_model_batch_size,
            'no_shuffle_warning': no_shuffle_warning,
            'num_classes': num_classes,
            'shuffle': shuffle,
            'start_time': start_time
        }

        return func(global_options, *args, **kwargs)
    return _parse_global_options


def standard_model_options(func):
    @functools.wraps(func)
    def _parse_standard_model_options(options, *args, **kwargs):
        dataset = options['dataset']

        base_model = parsing.get_torch_model(dataset)

        options = dict(options)
        options['base_model'] = base_model

        return func(options, *args, **kwargs)
    return _parse_standard_model_options


def pretrained_model_options(func):
    """
    Loads the pretrained weights and saves
    the model in foolbox and torch format.

    Requires:
        base_model
        cuda
        dataset
        device
        num_classes

    Adds:
        foolbox_model
        torch_model
    """

    @click.option('--weights-path', type=click.Path(file_okay=True, dir_okay=False), default=None)
    @click.option('--download-model', is_flag=True,
                  help='If the model file does not exist, download the pretrained model for the corresponding dataset.')
    @functools.wraps(func)
    def _parse_pretrained_model_options(options, weights_path, download_model, *args, **kwargs):
        base_model = options['base_model']
        cuda = options['cuda']
        dataset = options['dataset']
        device = options['device']
        max_model_batch_size = options['max_model_batch_size']
        num_classes = options['num_classes']

        if weights_path is None:
            weights_path = './pretrained_models/' + dataset + '.pth.tar'

        logger.debug('Loading pretrained weights from {}'.format(weights_path))

        torch_model = parsing.get_pretrained_torch_model(
            dataset, base_model, weights_path, download_model)

        torch_model = torch.nn.Sequential(
            parsing.get_normalisation_by_name(dataset), torch_model)

        torch_model.eval()

        if cuda:
            torch_model.cuda()

        foolbox_model = foolbox.models.PyTorchModel(
            torch_model, (0, 1), num_classes, channel_axis=3, device=device, preprocessing=(0, 1))

        if max_model_batch_size > 0:
            foolbox_model = model_tools.MaxBatchModel(foolbox_model, max_model_batch_size)

        options = dict(options)

        options['foolbox_model'] = foolbox_model
        options['torch_model'] = torch_model

        return func(options, *args, **kwargs)
    return _parse_pretrained_model_options


def custom_model_options(func):
    @click.option('--custom-weights-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
    @click.option('--custom-model-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
    @click.option('--custom-model-normalisation', default=None,
                  help='The normalisation that will be applied by the custom model. Supports both dataset names ({}) and '
                  'channel stds-means (format: "red_mean green_mean blue_mean red_stdev green_stdev blue_stdev" including quotes).'.format(', '.join(definitions.datasets)))
    @functools.wraps(func)
    def _parse_custom_model_options(options, custom_weights_path, custom_model_path, custom_model_normalisation, *args, **kwargs):
        cuda = options['cuda']
        dataset = options['dataset']
        device = options['device']
        max_model_batch_size = options['max_model_batch_size']
        num_classes = options['num_classes']

        # NXOR between custom_weights_path and custom_model_path
        if (custom_weights_path is None) == (custom_model_path is None):
            raise click.BadOptionUsage('--custom-weights-path',
                'You must pass either \'--custom-weights-path [PATH]\' or \'--custom-model-path [PATH]\' (but not both).')

        if custom_model_path is None:
            logger.info('No custom architecture path passed. Using default architecture {}'.format(
                definitions.default_architecture_names[dataset]))
            custom_torch_model = parsing.get_torch_model(dataset)
            logger.debug('Loading weights from {}'.format(custom_weights_path))
            model_tools.load_state_dict(
                custom_torch_model, custom_weights_path, False, False)
        else:
            logger.debug('Loading model from {}'.format(custom_model_path))
            custom_torch_model = torch.load(custom_model_path)

        custom_torch_model = parsing.apply_normalisation(custom_torch_model, custom_model_normalisation, 'custom model', '--custom-model-normalisation')

        custom_torch_model.eval()

        if cuda:
            custom_torch_model.cuda()

        custom_foolbox_model = foolbox.models.PyTorchModel(
            custom_torch_model, (0, 1), num_classes, channel_axis=3, device=device, preprocessing=(0, 1))

        if max_model_batch_size > 0:
            logger.debug('Applying model batch limiting: {}'.format(max_model_batch_size))
            custom_foolbox_model = model_tools.MaxBatchModel(custom_foolbox_model, max_model_batch_size)

        options = dict(options)
        options['custom_foolbox_model'] = custom_foolbox_model
        options['custom_torch_model'] = custom_torch_model

        return func(options, *args, **kwargs)
    return _parse_custom_model_options


def dataset_options(default_dataset, recommended=None):
    def _dataset_options(func):
        @click.option('--data-folder', default=None, type=click.Path(file_okay=False, dir_okay=True),
                      help='The path to the folder where the dataset is stored (or will be downloaded). '
                      'If unspecified, it defaults to \'./data/genuine/$dataset\'.')
        @click.option('--dataset-type', default=default_dataset, show_default=True, type=click.Choice(['train', 'test']),
                      help='Sets the dataset (train or test) that will be used. For ImageNet, we use the validation set '
                      'as test dataset.')
        @click.option('--download-dataset', is_flag=True,
                      help='If the dataset files do not exist, download them.')
        @click.option('--loader-workers', default=2, show_default=True, type=click.IntRange(0, None),
                      help='The number of parallel workers that will load the samples from the dataset. '
                      '0 disables parallelization.')
        @functools.wraps(func)
        def _parse_dataset_options(options, data_folder, dataset_type, download_dataset, loader_workers, *args, **kwargs):
            batch_size = options['batch_size']
            config_path = options['config_path']
            dataset = options['dataset']
            max_batches = options['max_batches']
            shuffle = options['shuffle']

            if data_folder is None:
                data_folder = './data/genuine/' + dataset

            if recommended is not None and dataset_type != recommended:
                logger.warning('You are using the {} dataset. We recommend using the {} dataset for this command.'.format(
                    dataset_type, recommended))

            logger.info('Using {} {} dataset.'.format(dataset, dataset_type))

            train_loader, test_loader = parsing.get_genuine_loaders(
                dataset, data_folder, batch_size, shuffle, loader_workers, download_dataset, config_path)

            if dataset_type == 'train':
                loader = train_loader
            else:
                loader = test_loader

            if max_batches is not None:
                loader = loaders.MaxBatchLoader(loader, max_batches)

            options = dict(options)
            options['dataset_type'] = dataset_type
            options['loader'] = loader

            return func(options, *args, **kwargs)
        return _parse_dataset_options
    return _dataset_options


def train_options(func):
    @click.argument('epochs', type=click.IntRange(1, None))
    @click.option('--optimiser', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True,
            help='The optimiser that will be used for training.')
    @click.option('--learning_rate', type=float, default=1e-3, show_default=True,
            help='The learning rate for the optimiser.')
    @click.option('--weight-decay', type=float, default=0, show_default=True,
            help='The weight decay for the optimiser.')
    @click.option('--adam-betas', nargs=2, type=click.Tuple([float, float]), default=(0.9, 0.999), show_default=True,
            help='The two beta values. Ignored if the optimiser is not \'adam\'')
    @click.option('--adam-epsilon', type=float, default=1e-8, show_default=True,
            help='The value of epsilon. Ignored if the optimiser is not \'adam\'')
    @click.option('--adam-amsgrad', is_flag=True,
            help='Enables AMSGrad. Ignored if the optimiser is not \'adam\'')
    @click.option('--sgd-momentum', type=float, default=0, show_default=True,
            help='The intensity of momentum. Ignored if the optimiser is not \'sgd\'')
    @click.option('--sgd-dampening', type=float, default=0, show_default=True,
            help='The intensity of dampening. Ignored if the optimiser is not \'sgd\'')
    @click.option('--sgd-nesterov', is_flag=True,
            help='Enables Nesterov Accelerated Gradient. Ignored if the optimiser is not \'adam\'')
    @functools.wraps(func)
    def _parse_train_options(options, epochs, optimiser, learning_rate, weight_decay, adam_betas, adam_epsilon, adam_amsgrad, sgd_momentum, sgd_dampening, sgd_nesterov, *args, **kwargs):
        options = dict(options)

        options['adam_amsgrad'] = adam_amsgrad
        options['adam_betas'] = adam_betas
        options['adam_epsilon'] = adam_epsilon
        options['epochs'] = epochs
        options['learning_rate'] = learning_rate
        options['optimiser_name'] = optimiser
        options['sgd_dampening'] = sgd_dampening
        options['sgd_momentum'] = sgd_momentum
        options['sgd_nesterov'] = sgd_nesterov
        options['weight_decay'] = weight_decay

        return func(options, *args, **kwargs)
    return _parse_train_options


def test_options(test_name):
    def _test_options(func):
        @click.option('--results-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                      help='The path to the CSV file where the results will be saved. If unspecified '
                      'it defaults to \'./results/{}/$dataset $start_time.csv\''.format(test_name))
        @functools.wraps(func)
        def _parse_test_options(options, results_path, *args, **kwargs):
            dataset = options['dataset']
            start_time = options['start_time']

            if results_path is None:
                results_path = parsing.get_results_default_path(
                    test_name, dataset, start_time)

            options = dict(options)
            options['results_path'] = results_path

            return func(options, *args, **kwargs)
        return _parse_test_options
    return _test_options

def attack_options(attacks, mandatory_parallelization=False):
    def _attack_options(func):
        @click.argument('attack', type=click.Choice(attacks))
        @click.argument('attack_p', type=click.Choice(definitions.supported_ps))
        @functools.wraps(func)
        def _parse_attack_options(options, attack, attack_p, attack_workers, *args, **kwargs):
            attack_p = float(attack_p)

            mean = not np.isposinf(attack_p) # Don't average L-inf
            attack_distance_measure = parsing.distance_tools.LpDistanceMeasure(attack_p, mean)
            logger.info('Attack distance measure: {}'.format(attack_distance_measure))

            if attack in definitions.parallelizable_attacks:
                logger.debug('Attack supports parallelization.')
            else:
                logger.debug('Attack does not support parallelization.')

                if attack_workers > 0:
                    raise click.BadOptionUsage('--attack-workers', 'The chosen attack \'{}\' does not support parallelization.'.format(attack))

            logger.info('Attack workers: {}.'.format(attack_workers))

            options = dict(options)

            # We don't immediately parse 'attack' because every test needs a specific configuration
            options['attack_name'] = attack
            options['attack_distance_measure'] = attack_distance_measure
            options['attack_workers'] = attack_workers

            return func(options, *args, **kwargs)

        parse_func = _parse_attack_options

        if mandatory_parallelization:
            parse_func = click.argument('attack_workers', type=click.IntRange(1, None))(parse_func)
        else:
            parse_func = click.option('--attack-workers', type=click.IntRange(0, None), show_default=True,
            help='The number of parallel workers that will be used to speed up the attack. 0 disables parallelization.')(parse_func)

        return parse_func
    return _attack_options


def distance_tool_options(func):
    @click.argument('defense_p', type=click.Choice(definitions.supported_ps))
    @functools.wraps(func)
    def _parse_distance_tool_options(options, defense_p, *args, **kwargs):
        defense_p = float(defense_p)
        mean = not np.isposinf(defense_p)
        defense_distance_measure = parsing.distance_tools.LpDistanceMeasure(defense_p, mean)
        logger.info('Defense distance measure: {}'.format(defense_distance_measure))

        options = dict(options)

        options['defense_distance_measure'] = defense_distance_measure

        return func(options, *args, **kwargs)
    return _parse_distance_tool_options


def counter_attack_options(required):
    def _counter_attack_options(func):
        @click.option('--counter-attack-workers', type=click.IntRange(0, None), default=0, show_default=True,
                      help='The number of attack workers of the counter attack.')
        @functools.wraps(func)
        def _parse_counter_attack_options(options, counter_attack, counter_attack_workers, *args, **kwargs):
            defense_distance_measure = options['defense_distance_measure']
            max_model_batch_size = options['max_model_batch_size']

            if counter_attack in definitions.parallelizable_attacks:
                logger.debug('Counter attack supports parallelization.')
            else:
                logger.debug('Counter attack does not support parallelization.')

                if counter_attack_workers is not None and counter_attack_workers > 0:
                    raise click.BadOptionUsage('--counter-attack-workers', 'The chosen counter-attack \'{}\' does not support parallelization.'.format(counter_attack))

                counter_attack_workers = 0

            logger.info('Counter attack workers: {}.'.format(counter_attack_workers))

            if max_model_batch_size > 0 and counter_attack_workers > max_model_batch_size:
                raise click.BadOptionUsage('--counter-attack-workers',
                    'The number of counter attack workers must be at most the maximum model batch size. '
                    'Either increase the maximum model batch size, decrease the number of '
                    'counter attack workers, or disable model batch limiting.')

            counter_attack = parsing.parse_attack(
                counter_attack, defense_distance_measure, foolbox.criteria.Misclassification())

            options = dict(options)

            options['counter_attack'] = counter_attack
            options['counter_attack_workers'] = counter_attack_workers

            return func(options, *args, **kwargs)

        parse_func = _parse_counter_attack_options
        if required:
            parse_func = click.argument(
                'counter_attack', type=click.Choice(definitions.supported_attacks))(parse_func)
        else:
            parse_func = click.option('--counter-attack', default='deepfool', type=click.Choice(definitions.supported_attacks),
                                      help='The counter-attack that will be used (if required).')(parse_func)

        return parse_func
    return _counter_attack_options


def detector_options(func):
    @click.argument('detector', type=click.Choice(definitions.supported_detectors))
    @click.option('--reject-on-failure', type=bool, default=True, show_default=True,
                  help='If True, samples for which the detector cannot compute the score will be rejected. If False, they will be accepted.')
    @click.option('--cache-size', type=click.IntRange(0, None), default=0, show_default=True,
                  help='The size of the distance tool cache. 0 disables caching.')
    @functools.wraps(func)
    def _parse_detector_options(options, detector, reject_on_failure, cache_size, *args, **kwargs):

        # To avoid confusion
        detector_name = detector
        del detector

        if reject_on_failure:
            failure_value = -np.Infinity
        else:
            failure_value = np.Infinity

        if detector_name in definitions.supported_distance_tools:
            logger.debug('The detector is a distance tool.')
            detector_type = 'distance'

            if cache_size == 0:
                logger.debug('Caching disabled.')
                enable_caching = False
            else:
                if detector_name in definitions.cache_distance_tools:
                    logger.debug('Caching enabled')
                    enable_caching = True
                else:
                    logger.debug(
                        'Caching is enabled, but the distance tool does not support it.')
                    enable_caching = False

            distance_tool = parsing.parse_distance_tool(
                detector_name, options, failure_value)
            detector = detectors.DistanceDetector(distance_tool)

        elif detector_name in definitions.supported_standard_detectors:
            logger.debug('The detector is a standard detector.')
            detector_type = 'standard'

            enable_caching = False
            distance_tool = None
            detector = parsing.parse_standard_detector(
                detector_name, options, failure_value)
        else:
            raise ValueError('Detector not supported.')

        options = dict(options)

        options['cache_size'] = cache_size
        options['detector'] = detector
        options['detector_name'] = detector_name
        options['detector_type'] = detector_type
        options['distance_tool'] = distance_tool
        options['enable_caching'] = enable_caching
        options['failure_value'] = failure_value

        return func(options, *args, **kwargs)
    return _parse_detector_options


def rejector_options(func):
    @click.argument('threshold', type=float)
    @functools.wraps(func)
    def _parse_rejector_options(options, threshold, *args, **kwargs):
        cache_size = options['cache_size']
        defense_distance_measure = options['defense_distance_measure']
        detector = options['detector']
        detector_type = options['detector_type']
        distance_tool = options['distance_tool']
        enable_caching = options['enable_caching']
        foolbox_model = options['foolbox_model']

        if enable_caching:
            assert detector_type == 'distance'
            assert distance_tool is not None

            rejector = rejectors.CacheRejector(
                distance_tool, threshold, defense_distance_measure, foolbox_model.bounds(), cache_size)

        else:
            rejector = rejectors.DetectorRejector(detector, threshold)

        options = dict(options)

        options['rejector'] = rejector
        options['threshold'] = threshold

        return func(options, *args, **kwargs)

    return _parse_rejector_options


def preprocessor_options(func):
    @click.argument('preprocessor', type=click.Choice(definitions.supported_preprocessors))
    @click.option('--feature-squeezing-bit-depth', type=int, default=8, show_default=True,
                  help='The bit depth of feature squeezing (only applied if preprocessor is \'feature_squeezing\').')
    @click.option('--spatial-smoothing-window', type=int, default=3, show_default=True,
                  help='The size of the sliding window for spatial smoothing (only applied if preprocessor is \'spatial_smoothing\').')
    @functools.wraps(func)
    def _parse_preprocessor_options(options, preprocessor, feature_squeezing_bit_depth, spatial_smoothing_window, *args, **kwargs):

        options = dict(options)

        options['feature_squeezing_bit_depth'] = feature_squeezing_bit_depth
        options['spatial_smoothing_window'] = spatial_smoothing_window

        # preprocessor must be parsed last
        preprocessor = parsing.parse_preprocessor(
            preprocessor, options)
        options['preprocessor'] = preprocessor
        return func(options, *args, **kwargs)

    return _parse_preprocessor_options


def adversarial_dataset_options(func):
    @click.argument('adversarial_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
    @click.option('--max-adversarial-batches', type=click.IntRange(1, None), default=None,
                  help='The maximum number of batches. If unspecified, no batch limiting is applied.')
    @functools.wraps(func)
    def _parse_adversarial_dataset_options(options, adversarial_dataset_path, max_adversarial_batches, *args, **kwargs):
        batch_size = options['batch_size']
        shuffle = options['shuffle']

        adversarial_list, adversarial_generation_success_rate = utils.load_zip(
            adversarial_dataset_path)

        adversarial_loader = loaders.ListLoader(
            adversarial_list, batch_size, shuffle)

        if max_adversarial_batches is not None:
            if (not options['shuffle']) and (not options['no_shuffle_warning']):
                logger.warning('You are limiting the number of adversarial batches, but you are not applying any shuffling. '
                               'This means that the last parts of your adversarial dataset will be never used. You can disable this '
                               'warning by passing \'--no-shuffle-warning\'.')

            adversarial_loader = loaders.MaxBatchLoader(
                adversarial_loader, max_adversarial_batches)

        options = dict(options)
        options['adversarial_loader'] = adversarial_loader
        options['adversarial_generation_success_rate'] = adversarial_generation_success_rate

        return func(options, *args, **kwargs)

    return _parse_adversarial_dataset_options

def substitute_options(func):
    @click.argument('substitute_model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
    @click.option('--substitute-normalisation', default=None,
                  help='The normalisation that will be applied by the substitute model. Supports both dataset names ({}) and '
                  'channel stds-means (format: "red_mean green_mean blue_mean red_stdev green_stdev blue_stdev" including quotes).'.format(', '.join(definitions.datasets)))
    @functools.wraps(func)
    def _parse_substitute_options(options, substitute_model_path, substitute_normalisation, *args, **kwargs):
        cuda = options['cuda']
        device = options['device']
        max_model_batch_size = options['max_model_batch_size']
        num_classes = options['num_classes']

        substitute_torch_model = torch.load(substitute_model_path)

        substitute_torch_model = parsing.apply_normalisation(substitute_torch_model, substitute_normalisation, 'substitute model', '--substitute-normalisation')

        substitute_torch_model.eval()

        if cuda:
            substitute_torch_model.cuda()

        substitute_foolbox_model = foolbox.models.PyTorchModel(
            substitute_torch_model, (0, 1), num_classes, channel_axis=3, device=device, preprocessing=(0, 1))

        if max_model_batch_size > 0:
            substitute_foolbox_model = model_tools.MaxBatchModel(substitute_foolbox_model, max_model_batch_size)

        options = dict(options)
        options['substitute_foolbox_model'] = substitute_foolbox_model
        options['substitute_torch_model'] = substitute_torch_model

        return func(options, *args, **kwargs)
    
    return _parse_substitute_options

def approximation_dataset_options(defense_name):
    def _approximation_dataset_options(func):
        @click.option('--approximation-dataset-path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None)
        @functools.wraps(func)
        def _parse_approximation_dataset_options(options, approximation_dataset_path, *args, **kwargs):
            dataset = options['dataset']
            start_time = options['start_time']

            if approximation_dataset_path is None:
                approximation_dataset_path = parsing.get_custom_dataset_default_path('approximation/' + defense_name, dataset, start_time)

            options = dict(options)

            options['approximation_dataset_path'] = approximation_dataset_path

            return func(options, *args, **kwargs)
        return _parse_approximation_dataset_options
    return _approximation_dataset_options