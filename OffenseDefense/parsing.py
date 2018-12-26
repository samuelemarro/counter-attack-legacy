import datetime
import functools
import logging
import pathlib
import os
import shutil

import art.defences
import click
import foolbox
import numpy as np
import sys
import tarfile
import torch
import torchvision

from . import batch_attack, cifar_models, defenses, detectors, distance_tools, loaders, model_tools, training, utils

datasets = ['cifar10', 'cifar100', 'imagenet']
supported_attacks = ['boundary', 'deepfool', 'fgsm']
parallelizable_attacks = ['deepfool', 'fgsm']
differentiable_attacks = ['deepfool', 'fgsm']
black_box_attacks = [
    x for x in supported_attacks if x not in differentiable_attacks]

supported_distance_tools = ['anti-attack']
supported_detectors = list(supported_distance_tools)
supported_preprocessors = ['feature_squeezing', 'spatial_smoothing']

supported_ps = ['2', 'inf']


logger = logging.getLogger(__name__)


def _get_results_default_path(test_name, dataset, start_time):
    return './results/{}/{} {:%Y-%m-%d %H-%M-%S}.csv'.format(test_name, dataset, start_time)


def get_training_default_path(training_name, dataset, start_time):
    return './trained_models/{}/{} {:%Y-%m-%d %H-%M-%S}.pth.tar'.format(training_name, dataset, start_time)


def get_custom_dataset_default_path(name, original_dataset, start_time):
    return './data/{}/{} {:%Y-%m-%d %H-%M-%S}.pth.tar'.format(name, original_dataset, start_time)


def _cifar_loader(dataset, path, train, download, batch_size, shuffle, num_workers):
    if dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100
    else:
        raise ValueError('dataset must be either \'cifar10\' or \'cifar100\'.')

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset = data(root=path,
                       train=train,
                       download=download,
                       transform=torchvision.transforms.ToTensor())
    except RuntimeError:
        raise RuntimeError(
            'Dataset files not found. Use --download-dataset to automatically download missing files.')
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def _download_imagenet(path, config_path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    train_path = path / 'train'
    train_file_path = train_path / 'ILSVRC2012_img_train.tar'
    val_path = path / 'val'
    val_file_path = val_path / 'ILSVRC2012_img_val.tar'

    train_path.mkdir(parents=True, exist_ok=True)
    utils.download_from_config(
        config_path, train_file_path, 'dataset_links', 'imagenet_train')
    tarfile.open(train_file_path).extractall(train_path)
    os.remove(train_file_path)

    for file_name in os.listdir(train_path):
        logger.debug(file_name)
        # Skip files that are not tar files
        if not file_name.endswith('.tar'):
            continue

        class_file_path = train_path / file_name
        class_path = train_path / file_name[:-4]

        # Create /aaaaa
        os.mkdir(class_path)
        # Extract aaaaa.tar in /aaaaa
        tarfile.open(class_file_path).extractall(class_path)
        # Remove aaaaa.tar
        os.remove(class_file_path)

    val_path.mkdir(parents=True, exist_ok=True)
    utils.download_from_config(
        config_path, val_file_path, 'dataset_links', 'imagenet_val')
    tarfile.open(val_file_path).extractall(val_path)
    os.remove(val_file_path)

    ground_truths = utils.load_json('imagenet_val_ground_truth.txt')[0]
    classes = ground_truths['classes']
    labels = ground_truths['labels']

    for _class in classes:
        os.mkdir(val_path / _class)

    for file_name, label in labels.items():
        shutil.move(val_path / file_name, val_file_path / label)


def _imagenet_loader(path, train, download, batch_size, shuffle, num_workers, config_path):
    if not pathlib.Path(path).exists():
        if download:
            _download_imagenet(path, config_path)
        else:
            raise RuntimeError(
                'Dataset files not found. Use --download-dataset to automatically download missing files.')

    if train:
        data_dir = os.path.join(path, 'train')
    else:
        data_dir = os.path.join(path, 'val')

    transforms = [torchvision.transforms.Resize(256),
                  torchvision.transforms.CenterCrop(224),
                  torchvision.transforms.ToTensor()]

    dataset = torchvision.datasets.ImageFolder(
        data_dir,
        torchvision.transforms.Compose(transforms))

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)

    return loader


def _get_genuine_loaders(dataset, path, batch_size, shuffle, num_workers, download, config_path):
    if dataset in ['cifar10', 'cifar100']:
        train_loader = _cifar_loader(
            dataset, path, True, download, batch_size, shuffle, num_workers)
        test_loader = _cifar_loader(
            dataset, path, False, download, batch_size, shuffle, num_workers)

    elif dataset == 'imagenet':
        train_loader = _imagenet_loader(
            path, True, download, batch_size, shuffle, num_workers, config_path)
        test_loader = _imagenet_loader(
            path, False, download, batch_size, shuffle, num_workers, config_path)
    else:
        raise ValueError('Dataset not supported.')

    train_loader = loaders.TorchLoader(train_loader)
    test_loader = loaders.TorchLoader(test_loader)

    return train_loader, test_loader


def _download_pretrained_model(dataset, path):
    if dataset in ['cifar10', 'cifar100']:
        utils.download_from_config(
            'config.ini', path, 'model_links', dataset)
    elif dataset == 'imagenet':
        model = torchvision.models.densenet161(pretrained=True)
        model_tools.save_model(model, path)
    else:
        raise ValueError('Dataset not supported.')


def _get_torch_model(dataset: str) -> torch.nn.Module:
    """Returns the pretrained Torch model for a given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Currently supported values
        are ['cifar10', 'cifar100', 'imagenet']

    Raises
    ------
    ValueError
        If the dataset is not supported.

    Returns
    -------
    torch.nn.Module
        The pretrained Torch model for the given dataset.
    """

    # Use the models that have shown the best top-1 accuracy
    if dataset in ['cifar10', 'cifar100']:
        # For CIFAR10(0), we use a DenseNet with depth 190 and growth rate 40
        num_classes = 10 if dataset == 'cifar10' else 100
        model = cifar_models.densenet(
            depth=190, growthRate=40, num_classes=num_classes)
    elif dataset == 'imagenet':
        # For ImageNet, we use a Densenet with depth 161 and growth rate 48
        model = torchvision.models.densenet161(pretrained=False)
    else:
        raise ValueError('Dataset not supported.')

    return model


def _get_pretrained_torch_model(dataset: str, path: str, download: bool) -> torch.nn.Module:
    """Returns the pretrained Torch model for a given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Currently supported values
        are ['cifar10', 'cifar100', 'imagenet']
    path : str
        The path to the file where the pretrained model is
        stored (or will be downloaded).
    download : bool
        If True it will download the pretrained model if the specified
        file does not exist.

    Raises
    ------
    ValueError
        If the dataset is not supported.

    Returns
    -------
    torch.nn.Module
        The pretrained Torch model for the given dataset.
    """
    model = _get_torch_model(dataset)
    if not pathlib.Path(path).exists():
        if download:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            _download_pretrained_model(dataset, path)
        else:
            raise RuntimeError(
                'No pretrained model found: {}. Use --download-model to automatically download missing models.'.format(path))

    model = model_tools.load_model(model, path, False, False)
    return model


def _get_preprocessing(dataset: str) -> model_tools.Preprocessing:
    """Returns the preprocessing for a given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Currently supported values
        are ['cifar10', 'cifar100', 'imagenet']

    Raises
    ------
    ValueError
        If the dataset is not supported.

    Returns
    -------
    model_tools.Preprocessing
        The Preprocessing model with the means and standard
        deviations for the given dataset.
    """

    if dataset in ['cifar10', 'cifar100']:
        # The pretrained CIFAR models use the same preprocessing
        # for both versions
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2023, 0.1994, 0.2010)
    elif dataset == 'imagenet':
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    else:
        raise ValueError('Dataset not supported.')

    return model_tools.Preprocessing(means, stds)


def _get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError('Dataset not supported')


def parse_attack_constructor(attack_name, p):
    if attack_name == 'deepfool':
        if p == 2:
            return foolbox.attacks.DeepFoolL2Attack
        elif p == np.inf:
            return foolbox.attacks.DeepFoolLinfinityAttack
        else:
            raise ValueError('Deepfool supports L-2 and L-Infinity')
    elif attack_name == 'fgsm':
        return foolbox.attacks.FGSM
    elif attack_name == 'boundary':
        return foolbox.attacks.BoundaryAttack
    else:
        raise ValueError('Attack not supported.')


def parse_distance_tool(tool_name, options, failure_value=-np.Infinity):
    anti_attack = options['anti_attack']
    anti_attack_p = options['anti_attack_p']
    attack_workers = options['attack_workers']
    foolbox_model = options['foolbox_model']
    parallelize_anti_attack = options['parallelize_anti_attack']
    torch_model = options['torch_model']

    if tool_name == 'anti-attack':
        # We treat failures as -Infinity because failed
        # detection means that the sample is likely adversarial

        if parallelize_anti_attack:
            batch_worker = batch_attack.TorchWorker(torch_model)
            distance_tool = distance_tools.AdversarialDistance(foolbox_model, anti_attack,
                                                               anti_attack_p, failure_value, batch_worker, attack_workers)
        else:
            distance_tool = distance_tools.AdversarialDistance(foolbox_model, anti_attack,
                                                               anti_attack_p, failure_value)
    else:
        raise ValueError('Distance tool not supported.')

    return distance_tool


def parse_detector(detector, options, failure_value=-np.Infinity):
    if detector in supported_distance_tools:
        return detectors.DistanceDetector(parse_distance_tool(detector, options, failure_value))
    else:
        raise ValueError('Detector not supported.')


def parse_preprocessor(preprocessor, options):
    if preprocessor == 'spatial_smoothing':
        return art.defences.SpatialSmoothing(options['spatial_smoothing_window'])
    elif preprocessor == 'feature_squeezing':
        return art.defences.FeatureSqueezing(options['feature_squeezing_bit_depth'])
    else:
        raise ValueError('Preprocessor not supported.')


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def set_parameters(parameters):
    def _set_parameters(func):
        @functools.wraps(func)
        def _parse_set_parameters(options, *args, **kwargs):
            set_parameters_options = dict(options)
            for key, value in parameters.items():
                set_parameters_options[key] = value

            return func(set_parameters_options, *args, **kwargs)
        return _parse_set_parameters
    return _set_parameters


def global_options(func):
    @click.argument('dataset', type=click.Choice(datasets))
    @click.option('-b', '--batch-size', default=5, show_default=True, type=click.IntRange(1, None),
                  help='The size of each batch.')
    @click.option('-mb', '--max-batches', type=click.IntRange(1, None), default=None,
                  help='The maximum number of batches. If unspecified, no batch limiting is applied.')
    @click.option('-s', '--shuffle', type=bool, default=True, show_default=True,
                  help='Whether to shuffle the dataset.')
    @click.option('-cp', '--config-path', default='./config.ini', type=click.Path(file_okay=True, exists=True),
                  help='The path to the configuration file.')
    @click.option('-nc', '--no-cuda', is_flag=True)
    @click.option('-nsw', '--no-shuffle-warning', is_flag=True,
                  help='Disables the warning for limiting batches without shuffling.')
    @click.option('-v', '--verbosity', default='info', show_default=True, type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
                  help='Sets the level of verbosity.')
    @functools.wraps(func)
    def _parse_global_options(dataset, batch_size, max_batches, shuffle, config_path, no_cuda, no_shuffle_warning, verbosity, *args, **kwargs):
        start_time = datetime.datetime.now()

        command = ' '.join(sys.argv[1:])

        if max_batches is not None:
            if (not shuffle) and (not no_shuffle_warning):
                logger.warning('You are limiting the number of batches, but you aren\'t applying any shuffling. '
                               'This means that the last parts of your dataset will be never used. You can disable this '
                               'warning by passing \'--no-shuffle-warning\' (alias: \'-nsw\').')

        num_classes = _get_num_classes(dataset)

        cuda = torch.cuda.is_available() and not no_cuda

        logging.getLogger('OffenseDefense').setLevel(verbosity.upper())

        parsed_common_options = {
            'batch_size': batch_size,
            'command': command,
            'config_path': config_path,
            'cuda': cuda,
            'dataset': dataset,
            'max_batches': max_batches,
            'no_shuffle_warning': no_shuffle_warning,
            'num_classes': num_classes,
            'shuffle': shuffle,
            'start_time': start_time
        }

        return func(parsed_common_options, *args, **kwargs)
    return _parse_global_options


def pretrained_model_options(func):
    @click.option('-mp', '--model-path', type=click.Path(file_okay=True, dir_okay=False), default=None)
    @click.option('-dm', '--download-model', is_flag=True,
                  help='If the model file does not exist, download the pretrained model for the corresponding dataset.')
    @functools.wraps(func)
    def _parse_pretrained_model_options(options, model_path, download_model, *args, **kwargs):
        cuda = options['cuda']
        dataset = options['dataset']
        num_classes = options['num_classes']

        if model_path is None:
            model_path = './pretrained_models/' + dataset + '.pth.tar'

        torch_model = _get_pretrained_torch_model(
            dataset, model_path, download_model)
        torch_model = torch.nn.Sequential(
            _get_preprocessing(dataset), torch_model)

        torch_model.eval()

        if cuda:
            torch_model.cuda()

        device = torch.cuda.current_device() if cuda else 'cpu'

        foolbox_model = foolbox.models.PyTorchModel(
            torch_model, (0, 1), num_classes, channel_axis=3, device=device, preprocessing=(0, 1))

        parsed_pretrained_model_options = dict(options)

        parsed_pretrained_model_options['foolbox_model'] = foolbox_model
        parsed_pretrained_model_options['torch_model'] = torch_model

        return func(parsed_pretrained_model_options, *args, **kwargs)
    return _parse_pretrained_model_options


def dataset_options(recommended):
    def _dataset_options(func):
        @click.option('-df', '--data-folder', default=None, type=click.Path(file_okay=False, dir_okay=True),
                      help='The path to the folder where the dataset is stored (or will be downloaded). '
                      'If unspecified, it defaults to \'./data/genuine/$dataset$\'.')
        @click.option('-dt', '--dataset-type', default=recommended, show_default=True, type=click.Choice(['train', 'test']),
                      help='Sets the dataset (train or test) that will be used.')
        @click.option('-dd', '--download-dataset', is_flag=True,
                      help='If the dataset files do not exist, download them.')
        @click.option('-lw', '--loader-workers', default=2, show_default=True, type=click.IntRange(0, None),
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

                train_loader, test_loader = _get_genuine_loaders(
                    dataset, data_folder, batch_size, shuffle, loader_workers, download_dataset, config_path)

                if dataset_type == 'train':
                    loader = train_loader
                else:
                    loader = test_loader

                if max_batches is not None:
                    loader = loaders.MaxBatchLoader(loader, max_batches)

                parsed_dataset_options = dict(options)
                parsed_dataset_options['dataset_type'] = dataset_type
                parsed_dataset_options['loader'] = loader

                return func(parsed_dataset_options, *args, **kwargs)
        return _parse_dataset_options
    return _dataset_options


def train_options(func):
    @click.argument('epochs', type=click.IntRange(1, None))
    @click.option('-o', '--optimizer', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True)
    @click.option('-lr', '--learning_rate', type=float, default=1e-3, show_default=True)
    @click.option('-wd', '--weight-decay', type=float, default=0, show_default=True)
    @click.option('-ab', '--adam-betas', nargs=2, type=click.Tuple([float, float]), default=(0.9, 0.999), show_default=True)
    @click.option('-ae', '--adam-epsilon', type=float, default=1e-8, show_default=True)
    @click.option('-aa', '--adam-amsgrad', is_flag=True)
    @click.option('-sm', '--sgd-momentum', type=float, default=0, show_default=True)
    @click.option('-sd', '--sgd-dampening', type=float, default=0, show_default=True)
    @click.option('-sn', '--sgd-nesterov', is_flag=True)
    @functools.wraps(func)
    def _parse_train_options(options, epochs, optimizer, learning_rate, weight_decay, adam_betas, adam_epsilon, adam_amsgrad, sgd_momentum, sgd_dampening, sgd_nesterov, *args, **kwargs):
        torch_model = options['torch_model']
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(
                torch_model.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay, eps=adam_epsilon, amsgrad=adam_amsgrad)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                torch_model.parameters(), lr=learning_rate, momentum=sgd_momentum,
                dampening=sgd_dampening, weight_decay=weight_decay, nesterov=sgd_nesterov)
        else:
            raise ValueError('Optimizer not supported.')

        parsed_train_options = dict(options)

        parsed_train_options['adam_betas'] = adam_betas
        parsed_train_options['adam_epsilon'] = adam_epsilon
        parsed_train_options['epochs'] = epochs
        parsed_train_options['learning_rate'] = learning_rate
        parsed_train_options['optimizer'] = optimizer
        parsed_train_options['sgd_dampening'] = sgd_dampening
        parsed_train_options['sgd_momentum'] = sgd_momentum
        parsed_train_options['sgd_nesterov'] = sgd_nesterov
        parsed_train_options['weight_decay'] = weight_decay

        return func(parsed_train_options, *args, **kwargs)
    return _parse_train_options


def test_options(test_name):
    def _test_options(func):
        @click.option('-rp', '--results-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                      help='The path to the CSV file where the results will be saved. If unspecified '
                      'it defaults to \'./results/{}/$dataset$ $start_time$.csv\''.format(test_name))
        @functools.wraps(func)
        def _parse_test_options(options, results_path, *args, **kwargs):
            dataset = options['dataset']
            start_time = options['start_time']

            if results_path is None:
                results_path = _get_results_default_path(
                    test_name, dataset, start_time)

            parsed_test_options = dict(options)
            parsed_test_options['results_path'] = results_path

            return func(parsed_test_options, *args, **kwargs)
        return _parse_test_options
    return _test_options


def parallelization_options(func):
    @click.option('-nap', '--no-attack-parallelization', is_flag=True,
                  help='Disables attack parallelization. This might increase the execution time.')
    @click.option('-aw', '--attack-workers', default=5, show_default=True, type=click.IntRange(1, None),
                  help='The number of parallel workers that will be used to speed up the attack (if possible).')
    @functools.wraps(func)
    def parse_parallelization_options(options, no_attack_parallelization, attack_workers, *args, **kwargs):
        torch_model = options['torch_model']

        enable_attack_parallelization = not no_attack_parallelization
        model_batch_worker = batch_attack.TorchWorker(torch_model)

        parsed_parallelization_options = dict(options)
        parsed_parallelization_options['attack_workers'] = attack_workers
        parsed_parallelization_options['enable_attack_parallelization'] = enable_attack_parallelization
        parsed_parallelization_options['model_batch_worker'] = model_batch_worker

        return func(parsed_parallelization_options, *args, **kwargs)
    return parse_parallelization_options


def attack_options(attacks):
    def _attack_options(func):
        @click.argument('attack', type=click.Choice(attacks))
        @click.option('-p', default='inf', show_default=True, type=click.Choice(supported_ps),
                      help='The L_p distance of the attack.')
        @functools.wraps(func)
        def _parse_attack_options(options, attack, p, *args, **kwargs):
            enable_attack_parallelization = options['enable_attack_parallelization']
            foolbox_model = options['foolbox_model']
            torch_model = options['torch_model']
            loader = options['loader']

            if p == '2':
                p = 2
            elif p == 'inf':
                p = np.inf

            parallelize_attack = enable_attack_parallelization and attack in parallelizable_attacks

            parsed_attack_options = dict(options)

            # We don't immediately parse 'attack' because every test needs a specific configuration
            parsed_attack_options['attack_name'] = attack
            parsed_attack_options['enable_attack_parallelization'] = enable_attack_parallelization
            parsed_attack_options['p'] = p
            parsed_attack_options['parallelize_attack'] = parallelize_attack
            parsed_attack_options['loader'] = loader

            return func(parsed_attack_options, *args, **kwargs)
        return _parse_attack_options
    return _attack_options


def detector_options(failure_value):
    def _detector_options(func):
        @click.argument('detector', type=click.Choice(supported_detectors))
        @click.option('-aa', '--anti-attack', default='deepfool', type=click.Choice(supported_attacks),
                      help='The anti-attack that will be used (if required).')
        @click.option('-aap', '--anti-attack-p', default='inf', type=click.Choice(supported_ps),
                      help='The L_p distance of the anti-attack (if required).')
        @functools.wraps(func)
        def _parse_detector_options(options, detector, anti_attack, anti_attack_p, *args, **kwargs):
            foolbox_model = options['foolbox_model']
            enable_attack_parallelization = options['enable_attack_parallelization']

            if anti_attack_p == '2':
                anti_attack_p = 2
            elif anti_attack_p == 'inf':
                anti_attack_p = np.inf

            parallelize_anti_attack = (
                anti_attack in parallelizable_attacks) and enable_attack_parallelization

            anti_attack_constructor = parse_attack_constructor(
                anti_attack, anti_attack_p)
            anti_attack = anti_attack_constructor(
                foolbox_model, foolbox.criteria.Misclassification(),
                distance_tools.LpDistance(anti_attack_p))

            parsed_detector_options = dict(options)

            parsed_detector_options['anti_attack'] = anti_attack
            parsed_detector_options['anti_attack_p'] = anti_attack_p
            parsed_detector_options['parallelize_anti_attack'] = parallelize_anti_attack

            # detector must be parsed last
            detector = parse_detector(
                detector, parsed_detector_options, failure_value)
            parsed_detector_options['detector'] = detector

            return func(parsed_detector_options, *args, **kwargs)
        return _parse_detector_options
    return _detector_options


def preprocessor_options(func):
    @click.argument('preprocessor', type=click.Choice(supported_preprocessors))
    @click.option('-fsbd', '--feature-squeezing-bit-depth', type=int, default=8, show_default=True,
                  help='The bit depth of feature squeezing (only applied if preprocessor is \'feature_squeezing\').')
    @click.option('-ssw', '--spatial-smoothing-window', type=int, default=3, show_default=True,
                  help='The size of the sliding window for spatial smoothing (only applied if preprocessor is \'spatial_smoothing\').')
    @functools.wraps(func)
    def _parse_preprocessor_options(options, preprocessor, feature_squeezing_bit_depth, spatial_smoothing_window, *args, **kwargs):

        parsed_preprocessor_options = dict(options)

        parsed_preprocessor_options['feature_squeezing_bit_depth'] = feature_squeezing_bit_depth
        parsed_preprocessor_options['spatial_smoothing_window'] = spatial_smoothing_window
        # preprocessor must be parsed last
        preprocessor = parse_preprocessor(
            preprocessor, parsed_preprocessor_options)
        parsed_preprocessor_options['preprocessor'] = preprocessor
        return func(parsed_preprocessor_options, *args, **kwargs)

    return _parse_preprocessor_options


def adversarial_dataset_options(func):
    @click.argument('adversarial_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
    @click.option('--max-adversarial_batches', '-mab', type=click.IntRange(1, None), default=None,
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
                logger.warning('You are limiting the number of adversarial batches, but you aren\'t applying any shuffling. '
                               'This means that the last parts of your adversarial dataset will be never used. You can disable this '
                               'warning by passing \'--no-shuffle-warning\' (alias: \'-nsw\').')

            adversarial_loader = loaders.MaxBatchLoader(
                adversarial_loader, max_adversarial_batches)

        parsed_adversarial_dataset_options = dict(options)
        parsed_adversarial_dataset_options['adversarial_loader'] = adversarial_loader
        parsed_adversarial_dataset_options['adversarial_generation_success_rate'] = adversarial_generation_success_rate

        return func(parsed_adversarial_dataset_options, *args, **kwargs)

    return _parse_adversarial_dataset_options
