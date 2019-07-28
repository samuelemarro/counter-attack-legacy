import logging
import functools
import os
import pathlib
import tarfile

import art.defences
import click
import foolbox
import numpy as np
import shutil
import torch
import torchvision

from counter_attack import attacks, cifar_models, distance_measures, distance_tools, loaders, model_tools, utils

logger = logging.getLogger(__name__)

def get_results_default_path(test_name, dataset, start_time):
    return './results/{}/{} {:%Y-%m-%d %H-%M-%S}.csv'.format(test_name, dataset, start_time)


def get_training_default_path(training_name, dataset, start_time):
    return './trained_models/{}/{} {:%Y-%m-%d %H-%M-%S}.pth.tar'.format(training_name, dataset, start_time)


def get_custom_dataset_default_path(name, original_dataset, start_time):
    return './data/{}/{} {:%Y-%m-%d %H-%M-%S}.zip'.format(name, original_dataset, start_time)

def build_optimiser(optimiser_name, learnable_parameters, options):
    if optimiser_name == 'adam':
        optimiser = torch.optim.Adam(
            learnable_parameters, lr=options['learning_rate'], betas=options['adam_betas'], weight_decay=options['weight_decay'], eps=options['adam_epsilon'], amsgrad=options['adam_amsgrad'])
    elif optimiser_name == 'sgd':
        optimiser = torch.optim.SGD(
            learnable_parameters, lr=options['learning_rate'], momentum=options['sgd_momentum'],
            dampening=options['sgd_dampening'], weight_decay=options['weight_decay'], nesterov=options['sgd_nesterov'])
    else:
        raise ValueError('Optimiser not supported.')

    return optimiser

def _get_cifar_loader(dataset, path, train, download, batch_size, shuffle, num_workers):
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


def _get_imagenet_loader(path, train, download, batch_size, shuffle, num_workers, config_path):
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


def get_genuine_loaders(dataset, path, batch_size, shuffle, num_workers, download, config_path):
    if dataset in ['cifar10', 'cifar100']:
        train_loader = _get_cifar_loader(
            dataset, path, True, download, batch_size, shuffle, num_workers)
        test_loader = _get_cifar_loader(
            dataset, path, False, download, batch_size, shuffle, num_workers)

    elif dataset == 'imagenet':
        train_loader = _get_imagenet_loader(
            path, True, download, batch_size, shuffle, num_workers, config_path)
        test_loader = _get_imagenet_loader(
            path, False, download, batch_size, shuffle, num_workers, config_path)
    else:
        raise ValueError('Dataset not supported.')

    train_loader = loaders.TorchLoaderWrapper(train_loader)
    test_loader = loaders.TorchLoaderWrapper(test_loader)

    return train_loader, test_loader


def _download_pretrained_model(dataset, path):
    logger.info('Downloading pretrained model.')
    if dataset in ['cifar10', 'cifar100']:
        utils.download_from_config(
            'config.ini', path, 'model_links', dataset)
    elif dataset == 'imagenet':
        model = torchvision.models.densenet161(pretrained=True)

        # We save the model structure, too
        # model_tools.save_state_dict(model, path)
        torch.save(model, path)
    else:
        raise ValueError('Dataset not supported.')


def get_torch_model(dataset: str, is_rejector=False) -> torch.nn.Module:
    """Returns the pretrained Torch model for a given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Currently supported values
        are ['cifar10', 'cifar100', 'imagenet']
    is_rejector : bool
        If true, it adds an extra class, 'adversarial'

    Raises
    ------
    ValueError
        If the dataset is not supported.

    Returns
    -------
    torch.nn.Module
        The pretrained Torch model for the given dataset.
    """

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError('Dataset not supported')

    if is_rejector:
        num_classes = num_classes + 1

    # Use the models that have shown the best top-1 accuracy
    if dataset in ['cifar10', 'cifar100']:
        # For CIFAR10(0), we use a DenseNet with depth 100 and growth rate 12
        model = cifar_models.densenet(
            depth=100, growthRate=12, num_classes=num_classes)
    elif dataset == 'imagenet':
        # For ImageNet, we use a Densenet with depth 161 and growth rate 48
        model = torchvision.models.densenet161(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError('Dataset not supported.')

    return model


def get_pretrained_torch_model(dataset: str, base_model: torch.nn.Module, path: str, download: bool) -> torch.nn.Module:
    """Returns the pretrained Torch model for a given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Currently supported values
        are ['cifar10', 'cifar100', 'imagenet'].
    base_model : torch.nn.Model
        The model on which the pretrained weights will be applied.
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

    # We download the model structure, too

    path = pathlib.Path(path)
    state_dict_path = path.with_name(path.name.split('.')[0] + '_weights' + ''.join(path.suffixes))

    if not path.exists():
        if download:
            path.parent.mkdir(parents=True, exist_ok=True)
            _download_pretrained_model(dataset, str(state_dict_path))
            model_tools.load_state_dict(base_model, state_dict_path, False, False)
            torch.save(base_model, str(path))
        else:
            raise RuntimeError(
                'No pretrained model found: {}. Use --download-model to automatically download missing models.'.format(path))

    model = model_tools.load_state_dict(base_model, str(state_dict_path), False, False)
    # model = torch.load(str(path))
    return model


def get_normalisation_by_name(dataset: str) -> model_tools.Normalisation:
    """Returns the normalisation for a given dataset.

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
    model_tools.Normalisation
        The Normalisation module with the means and standard
        deviations for the given dataset.
    """

    if dataset in ['cifar10', 'cifar100']:
        # The pretrained CIFAR models use the same normalisation
        # for both versions
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2023, 0.1994, 0.2010)
    elif dataset == 'imagenet':
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    else:
        raise ValueError('Dataset not supported.')

    return model_tools.Normalisation(means, stds)

def apply_normalisation(model, normalisation, model_name, option_name):
    has_normalisation = model_tools.has_normalisation(model)

    logger.debug('{} has normalisation: {}'.format(model_name, has_normalisation))

    if not has_normalisation and normalisation is None:
        logger.warning('You are not applying any mean/stdev normalisation to the {}. '
                        'You can specify it by passing {} DATASET '
                        'or {} "red_mean green_mean blue_mean red_stdev green_stdev blue_stdev" (including quotes).'.format(model_name, option_name, option_name))

    if has_normalisation and normalisation is not None:
        logger.warning('You are applying mean/stdev normalisation to the {} multiple times.'.format(model_name))

    if normalisation is not None:
        logger.debug('Applying normalisation for the {}: {}'.format(model_name, normalisation))
        try:
            if normalisation in datasets:
                normalisation_module = _get_normalisation_by_name(normalisation)
            else:
                values = normalisation.split(' ')
                means = float(values[0]), float(
                    values[1]), float(values[2])
                stdevs = float(values[3]), float(
                    values[4]), float(values[5])
                normalisation_module = model_tools.Normalisation(means, stdevs)
        except:
            raise click.BadOptionUsage(option_name, 'Invalid normalisation format for the {}.'.format(model_name))

        model = torch.nn.Sequential(
            normalisation_module, model)

    return model


def get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError('Dataset not supported')

def parse_foolbox_distance(p):
    if p == 2:
        return foolbox.distances.MeanSquaredDistance
    elif np.isinf(p):
        return foolbox.distances.Linfinity
    else:
        raise NotImplementedError('Unsupported Lp.')

def parse_attack(attack_name, lp_distance, criterion, **attack_call_kwargs):
    attack_constructor = None
    p = lp_distance.p

    if attack_name == 'deepfool':
        if p == 2:
            attack_constructor = foolbox.attacks.DeepFoolL2Attack
        elif p == np.Infinity:
            attack_constructor = foolbox.attacks.DeepFoolLinfinityAttack
        else:
            raise ValueError('Deepfool supports L-2 and L-Infinity')
    elif attack_name == 'fgsm':
        attack_constructor = foolbox.attacks.FGSM
    elif attack_name == 'random_pgd':
        attack_constructor = foolbox.attacks.RandomPGD
    elif attack_name == 'boundary':
        attack_constructor = foolbox.attacks.BoundaryAttack
    else:
        raise ValueError('Attack not supported.')

    foolbox_distance = parse_foolbox_distance(p)

    attack = attack_constructor(None, criterion, foolbox_distance)

    if len(attack_call_kwargs) > 0:
        logger.debug('Added attack call keyword arguments: {}'.format(
            attack_call_kwargs))
        attack = attacks.AttackWithParameters(attack, **attack_call_kwargs)

    return attack


def parse_distance_tool(tool_name, options, failure_value):
    cuda = options['cuda']
    defense_lp_distance = options['defense_lp_distance']
    foolbox_model = options['foolbox_model']

    if tool_name == 'counter-attack':
        counter_attack = options['counter_attack']
        counter_attack_workers = options['counter_attack_workers']
        # Note: We use the Torch Foolbox model directly (without any defenses) because the counter-attack is the defense
        # We also use it because some attacks require the gradient.

        distance_tool = distance_tools.AdversarialDistance(foolbox_model, counter_attack,
                                                            defense_lp_distance, failure_value, cuda, counter_attack_workers)
    else:
        raise ValueError('Distance tool not supported.')

    return distance_tool


def parse_standard_detector(detector, options, failure_value):
    raise ValueError('Standard detector not supported.')


def parse_preprocessor(preprocessor, options):
    if preprocessor == 'spatial-smoothing':
        return art.defences.SpatialSmoothing(options['spatial_smoothing_window'])
    elif preprocessor == 'feature-squeezing':
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
