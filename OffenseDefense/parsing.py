import datetime
import functools
import logging
import pathlib
import os
import shutil

import click
import foolbox
import numpy as np
import sys
import tarfile
import torch
import torchvision

from . import batch_attack, cifar_models, detectors, distance_tools, loaders, model_tools, utils


datasets = ['cifar10', 'cifar100', 'imagenet']
supported_attacks = ['boundary', 'deepfool', 'fgsm']
parallelizable_attacks = ['deepfool', 'fgsm']
differentiable_attacks = ['deepfool', 'fgsm']
black_box_attacks = [
    x for x in supported_attacks if x not in differentiable_attacks]

supported_distance_tools = ['anti-attack']
supported_detectors = supported_distance_tools

supported_ps = ['2', 'inf']


logger = logging.getLogger(__name__)


def _get_results_default_path(test_name):
    return './results/{}/{:%Y-%m-%d %H-%M-%S}.csv'.format(test_name, datetime.datetime.now())


def _cifar_loader(dataset, path, train, download, batch_size, num_workers):
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
                                       shuffle=train,
                                       num_workers=num_workers)


def _download_imagenet(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    train_path = path / 'train'
    train_file_path = train_path / 'ILSVRC2012_img_train.tar'
    val_path = path / 'val'
    val_file_path = val_path / 'ILSVRC2012_img_val.tar'

    train_path.mkdir(parents=True, exist_ok=True)
    utils.download_from_config(
        'config.ini', train_file_path, 'dataset_links', 'imagenet_train')
    tarfile.open(train_file_path).extractall(train_path)
    os.remove(train_file_path)

    for file_name in os.listdir(train_path):
        logger.debug(file_name)
        # Skip files that are not tar files
        if not file_name.endswith('.tar'):
            continue

        class_file_path = train_path / file_name
        class_path = train_path / file_name[:-4]

        #Create /aaaaa
        os.mkdir(class_path)
        # Extract aaaaa.tar in /aaaaa
        tarfile.open(class_file_path).extractall(class_path)
        # Remove aaaaa.tar
        os.remove(class_file_path)

    val_path.mkdir(parents=True, exist_ok=True)
    utils.download_from_config(
        'config.ini', val_file_path, 'dataset_links', 'imagenet_val')
    tarfile.open(val_file_path).extractall(val_path)
    os.remove(val_file_path)

    ground_truths = utils.load_json('imagenet_val_ground_truth.txt')[0]
    classes = ground_truths['classes']
    labels = ground_truths['labels']

    for _class in classes:
        os.mkdir(val_path / _class)

    for file_name, label in labels.items():
        shutil.move(val_path / file_name, val_file_path / label)


def _imagenet_loader(path, train, download, batch_size, num_workers):
    if not pathlib.Path(path).exists():
        if download:
            _download_imagenet(path)
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
                                         shuffle=train,
                                         num_workers=num_workers)

    return loader


def _get_loaders(dataset, path, batch_size, num_workers, download):
    if dataset in ['cifar10', 'cifar100']:
        train_loader = _cifar_loader(
            dataset, path, True, download, batch_size, num_workers)
        test_loader = _cifar_loader(
            dataset, path, False, download, batch_size, num_workers)

    elif dataset == 'imagenet':
        train_loader = _imagenet_loader(
            path, True, download, batch_size, num_workers)
        test_loader = _imagenet_loader(
            path, False, download, batch_size, num_workers)
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


def _get_torch_model(dataset: str, path: str, download: bool) -> torch.nn.Module:
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
        The pretrained PyTorch model for the given dataset.
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

    if not pathlib.Path(path).exists():
        if download:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            _download_pretrained_model(dataset, path)
        else:
            raise RuntimeError(
                'No model found: {}. Use --download-model to automatically download missing models.'.format(path))

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


def _get_detector(detector, options):
    if detector in supported_distance_tools:
        return detectors.DistanceDetector(parse_distance_tool(detector, options))
    else:
        raise ValueError('Detector not supported.')


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


def parse_distance_tool(tool_name, options):
    anti_attack = options['anti_attack']
    anti_attack_p = options['anti_attack_p']
    attack_workers = options['attack_workers']
    batch_worker = options['batch_worker']
    foolbox_model = options['foolbox_model']
    parallelize_anti_attack = options['parallelize_anti_attack']

    if tool_name == 'anti-attack':
        # We treat failures as -Infinity because failed
        # detection means that the sample is likely adversarial

        if parallelize_anti_attack:
            distance_tool = distance_tools.AdversarialDistance(foolbox_model, anti_attack,
                                                               anti_attack_p, -np.Infinity, batch_worker, attack_workers)
        else:
            distance_tool = distance_tools.AdversarialDistance(foolbox_model, anti_attack,
                                                               anti_attack_p, -np.Infinity)
    else:
        raise ValueError('Distance tool not supported.')

    return distance_tool


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def global_options(func):
    @click.argument('dataset', type=click.Choice(datasets))
    @click.option('-dt', '--dataset-type', default='test', show_default=True, type=click.Choice(['train', 'test']),
                  help='Sets the dataset (train or test) that will be used.')
    @click.option('-df', '--data-folder', default=None, type=click.Path(file_okay=False, dir_okay=True),
                  help='The path to the folder where the dataset is stored (or will be downloaded). '
                  'If unspecified, it defaults to \'./data/genuine/$dataset$\'.')
    @click.option('-mp', '--model-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                  help='The path to the tar where the dataset is stored (or will be downloaded). '
                  'Ignored if dataset is \'imagenet\'. '
                  'If unspecified it defaults to \'./pretrained_models/$dataset$.pth.tar\'.')
    @click.option('-rp', '--results-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                  help='The path to the CSV file where the results will be saved. If unspecified '
                  'it defaults to \'./results/$command$/$datetime$.csv\'')
    @click.option('-b', '--batch', default=5, show_default=True,
                  type=click.IntRange(1, None))
    @click.option('-mb', '--max-batches', default=0, show_default=True, type=click.IntRange(0, None),
                  help='The maximum number of batches. 0 disables batch capping.')
    @click.option('-lw', '--loader-workers', default=2, show_default=True, type=click.IntRange(0, None),
                  help='The number of parallel workers that will load the samples from the dataset.'
                  '0 disables parallelization.')
    @click.option('-dm', '--download-model', is_flag=True,
                  help='If the model file does not exist, download the pretrained model for the corresponding dataset.')
    @click.option('-dd', '--download-dataset', is_flag=True,
                  help='If the dataset files do not exist, download them (not supported for ImageNet).')
    @click.option('-v', '--verbosity', default='info', show_default=True, type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
                  help='Sets the level of verbosity.')
    @click.pass_context
    @functools.wraps(func)
    def _parse_global_options(ctx, dataset, dataset_type, data_folder, model_path, results_path, batch, max_batches, loader_workers, download_model, download_dataset, verbosity, *args, **kwargs):
        if data_folder is None:
            data_folder = './data/genuine/' + dataset

        if model_path is None:
            model_path = './pretrained_models/' + dataset + '.pth.tar'

        if results_path is None:
            results_path = _get_results_default_path(ctx.command.name)

        command = ' '.join(sys.argv[1:])

        train_loader, test_loader = _get_loaders(
            dataset, data_folder, batch, loader_workers, download_dataset)

        if dataset_type == 'train':
            loader = train_loader
        else:
            loader = test_loader

        if max_batches > 0:
            loader = loaders.MaxBatchLoader(loader, max_batches)

        torch_model = _get_torch_model(dataset, model_path, download_model)
        torch_model = torch.nn.Sequential(
            _get_preprocessing(dataset), torch_model)
        torch_model.eval()

        num_classes = _get_num_classes(dataset)

        foolbox_model = foolbox.models.PyTorchModel(
            torch_model, (0, 1), num_classes, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0, 1))

        logger.setLevel(verbosity.upper())

        parsed_global_options = {
            'command': command,
            'dataset': dataset,
            'dataset_type': dataset_type,
            'foolbox_model': foolbox_model,
            'loader': loader,
            'torch_model': torch_model,
            'results_path': results_path
        }

        return func(parsed_global_options, *args, **kwargs)
    return _parse_global_options


def attack_options(attacks):
    def _attack_options(func):
        @global_options
        @click.argument('attack', type=click.Choice(attacks))
        @click.option('-p', default='inf', show_default=True, type=click.Choice(supported_ps),
                      help='The L_p distance of the attack.')
        @click.option('-aw', '--attack-workers', default=5, show_default=True, type=click.IntRange(1, None),
                      help='The number of parallel workers that will be used to speed up the attack (if possible).')
        @click.option('-lt', '--loader-type', default='standard', show_default=True, type=click.Choice(['standard', 'adversarial']),
                      help='The type of loader that will be used. \'standard\' uses the standard loader, \'adversarial\''
                      'replaces the samples with their adversarial samples (removing failed attacks).')
        @click.option('-la', '--loader-attack', default='deepfool', show_default=True, type=click.Choice(supported_attacks),
                      help='The attack that will be used by the loader. Ignored if loader-type is not \'adversarial\'.')
        @click.option('-lap', '--loader-attack-p', default='inf', show_default=True, type=click.Choice(supported_ps),
                      help='The L_p distance of the loader attack. Ignored if loader-type is not \'adversarial\'.')
        @click.option('-nap', '--no-attack-parallelization', is_flag=True,
                      help='Disables attack parallelization. This might increase the execution time.')
        @functools.wraps(func)
        def _parse_attack_options(parsed_global_options, attack, p, attack_workers, loader_type, loader_attack, loader_attack_p, no_attack_parallelization, *args, **kwargs):
            foolbox_model = parsed_global_options['foolbox_model']
            torch_model = parsed_global_options['torch_model']
            loader = parsed_global_options['loader']

            if p == '2':
                p = 2
            elif p == 'inf':
                p = np.inf

            enable_attack_parallelization = not no_attack_parallelization

            parallelize_attack = enable_attack_parallelization and attack in parallelizable_attacks

            batch_worker = batch_attack.TorchWorker(torch_model)

            if loader_type == 'standard':
                pass
            elif loader_type == 'adversarial':
                parallelize_loader_attack = enable_attack_parallelization and loader_attack in parallelizable_attacks

                loader_attack_constructor = parse_attack_constructor(
                    loader_attack, loader_attack_p)
                loader_attack = loader_attack_constructor(
                    foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(loader_attack_p))

                if parallelize_loader_attack:
                    loader = loaders.AdversarialLoader(
                        loader, foolbox_model, loader_attack, True, batch_worker, attack_workers)
                else:
                    loader = loaders.AdversarialLoader(
                        loader, foolbox_model, loader_attack, True)

            parsed_attack_options = dict(parsed_global_options)

            # We don't immediately parse 'attack' because every test needs a specific configuration
            parsed_attack_options['attack_name'] = attack

            parsed_attack_options['batch_worker'] = batch_worker
            parsed_attack_options['attack_workers'] = attack_workers
            parsed_attack_options['enable_attack_parallelization'] = enable_attack_parallelization
            parsed_attack_options['loader_attack'] = loader_attack
            parsed_attack_options['p'] = p
            parsed_attack_options['parallelize_attack'] = parallelize_attack
            parsed_attack_options['loader'] = loader

            return func(parsed_attack_options, *args, **kwargs)
        return _parse_attack_options
    return _attack_options

# Note: This decorator does not add the 'distance_tool' argument because
# different commands will accept different tools or non-tools. If you want to parse a
# distance_tool, call get_distance_tool.


def distance_options(attacks):
    def _distance_options(func):
        @attack_options(attacks)
        @click.option('-aa', '--anti-attack', default='deepfool', type=click.Choice(supported_attacks),
                      help='The anti-attack that will be used (if required).')
        @click.option('-aap', '--anti-attack-p', default=None, type=click.Choice(supported_ps),
                      help='The L_p distance of the anti-attack. If unspecified it defaults to p.')
        @functools.wraps(func)
        def _parse_distance_options(parsed_attack_options, anti_attack, anti_attack_p, *args, **kwargs):
            foolbox_model = parsed_attack_options['foolbox_model']
            p = parsed_attack_options['p']
            enable_attack_parallelization = parsed_attack_options['enable_attack_parallelization']

            if anti_attack_p is None:
                anti_attack_p = p

            parallelize_anti_attack = (
                anti_attack in parallelizable_attacks) and enable_attack_parallelization

            anti_attack_constructor = parse_attack_constructor(
                anti_attack, anti_attack_p)
            anti_attack = anti_attack_constructor(
                foolbox_model, foolbox.criteria.Misclassification(),
                distance_tools.LpDistance(anti_attack_p))

            parsed_distance_options = dict(parsed_attack_options)

            parsed_distance_options['anti_attack'] = anti_attack
            parsed_distance_options['anti_attack_p'] = anti_attack_p
            parsed_distance_options['parallelize_anti_attack'] = parallelize_anti_attack

            return func(parsed_distance_options, *args, **kwargs)
        return _parse_distance_options
    return _distance_options


def detector_options(attacks):
    def _detector_options(func):
        @distance_options(attacks)
        @click.argument('detector', type=click.Choice(supported_detectors))
        @functools.wraps(func)
        def _parse_detector_options(parsed_distance_options, detector, *args, **kwargs):
            parsed_detector_options = dict(parsed_distance_options)

            # detector must be parsed last
            detector = _get_detector(detector, parsed_detector_options)
            parsed_detector_options['detector'] = detector

            return func(parsed_detector_options, *args, **kwargs)
        return _parse_detector_options
    return _detector_options


def evasion_options(attacks):
    def _detector_options(func):
        @detector_options(attacks)
        @click.argument('threshold', type=float)
        @functools.wraps(func)
        def _parse_evasion_options(parsed_detector_options, threshold, *args, **kwargs):
            parsed_evasion_option = dict(parsed_detector_options)
            parsed_evasion_option['threshold'] = threshold
            return func(parsed_evasion_option, *args, **kwargs)
        return _parse_evasion_options
    return _detector_options
