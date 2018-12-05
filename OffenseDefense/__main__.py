import configparser
import functools
import logging
import pathlib
import queue
import sys

import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import click

import OffenseDefense
import OffenseDefense.attacks as attacks
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.detectors as detectors
import OffenseDefense.distance_tools as distance_tools
import OffenseDefense.loaders as loaders
import OffenseDefense.model_tools as model_tools
import OffenseDefense.tests as tests
import OffenseDefense.training as training
import OffenseDefense.utils as utils
import OffenseDefense.cifar_models as cifar_models


def batch_main():
    train_loader = model_tools.cifar10_train_loader(
        1, 10, flip=False, crop=False, normalize=False, shuffle=True)
    test_loader = model_tools.cifar10_test_loader(
        1, 10, normalize=False, shuffle=True)

    model = prepare_model()
    model.eval()

    foolbox_model = foolbox.models.PyTorchModel(
        model, (0, 1), 10, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0, 1))

    p = np.Infinity

    adversarial_criterion = foolbox.criteria.Misclassification()
    if p == 2:
        adversarial_attack = foolbox.attacks.DeepFoolLinfinityAttack(
            foolbox_model, adversarial_criterion, distance_tools.LpDistance(p))
    elif p == np.Infinity:
        adversarial_attack = foolbox.attacks.DeepFoolL2Attack(
            foolbox_model, adversarial_criterion, distance_tools.LpDistance(p))
    # adversarial_attack = attacks.RandomDirectionAttack(100, 100, 1e-2, 1e-5, foolbox_model, adversarial_criterion)

    # tests.image_test(foolbox_model, test_loader, adversarial_attack, adversarial_anti_attack)
    # direction_attack = attacks.RandomDirectionAttack(100, 100, 1e-2, 1e-5)

    direction_attack = attacks.RandomDirectionAttack(
        foolbox_model, foolbox.criteria.Misclassification(), p, 1000, 100, 0.05, 1e-7)
    black_box_attack = foolbox.attacks.BoundaryAttack(
        foolbox_model, foolbox.criteria.Misclassification(), distance=distance_tools.LpDistance(p))

    batch_worker = batch_attack.PyTorchWorker(model)
    num_workers = 50

    adversarial_distance_tool = distance_tools.AdversarialDistance(
        adversarial_attack.name(), foolbox_model, adversarial_attack, p, np.Infinity, batch_worker, num_workers)
    direction_distance_tool = distance_tools.AdversarialDistance(
        direction_attack.name(), foolbox_model, direction_attack, p, np.Infinity)
    black_box_distance_tool = distance_tools.AdversarialDistance(
        black_box_attack.name(), foolbox_model, direction_attack, p, np.Infinity)

    test_loader = loaders.TorchLoader(test_loader)
    adversarial_loader = loaders.AdversarialLoader(
        test_loader, foolbox_model, adversarial_attack, True, batch_worker, num_workers)
    random_noise_loader = loaders.RandomNoiseLoader(
        foolbox_model, 0, 1, [3, 32, 32], 10, 20)

    # Note: When running the distance_comparison_test, remember to use None as failure_value
    # tests.distance_comparison_test(foolbox_model, [
    #    adversarial_distance_tool, direction_distance_tool, black_box_distance_tool], adversarial_loader)
    # tests.attack_test(foolbox_model, test_loader, adversarial_attack, p, batch_worker, num_workers)
    # tests.accuracy_test(foolbox_model, test_loader, [1, 5])

    # train_loader = loaders.TorchLoader(train_loader)
    # training.train_torch(model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1), training.MaxEpoch(2), True)

    detector = detectors.DistanceDetector(adversarial_distance_tool)
    # tests.standard_detector_test(foolbox_model, test_loader, adversarial_attack, detector, batch_worker, num_workers)

    # load_pretrained_model('alexnet', 'cifar10', '')
    # tests.parallelization_test(foolbox_model, test_loader, adversarial_attack, p, batch_worker, num_workers)

    # Remember: attack_test removes misclassified samples. In this context, it means that it
    # will remove genuine samples that are rejected
    detector_model = detectors.CompositeDetectorModel(
        foolbox_model, detector, 1e-5)
    detector_aware_attack = foolbox.attacks.BoundaryAttack(detector_model,
                                                           foolbox.criteria.CombinedCriteria(
                                                               foolbox.criteria.Misclassification(), detectors.Undetected()),
                                                           distance=distance_tools.LpDistance(p))
    detector_aware_attack = attacks.AttackWithParameters(
        detector_aware_attack, verbose=True)
    tests.attack_test(detector_model,
                      adversarial_loader, detector_aware_attack, p)


cifar_names = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'


]

# if __name__ == '__main__':
#    pass
# main()
# attack_test()
#    batch_main()


def get_loaders(dataset, path, batch_size, num_workers, download):
    if dataset in ['cifar10', 'cifar100']:
        train_loader = model_tools.cifar_loader(
            dataset, path, True, download, batch_size, num_workers)
        test_loader = model_tools.cifar_loader(
            dataset, path, False, download, batch_size, num_workers)

        train_loader = loaders.TorchLoader(train_loader)
        test_loader = loaders.TorchLoader(test_loader)
    elif dataset == 'imagenet':
        raise NotImplementedError()
    else:
        raise ValueError('Dataset not supported.')

    return train_loader, test_loader


def download_pretrained_model(dataset, path):
    if dataset in ['cifar10', 'cifar100']:
        try:
            config = configparser.ConfigParser()
            config.read('config.ini')
            print(list(config.sections()))
            download_link = config.get('model_links', dataset)
        except KeyError:
            raise IOError(
                'config.ini does not contain the link for \'{}\''.format(dataset))
        except IOError:
            raise IOError('Could not read config.ini')

        try:
            utils.download(download_link, path)
        except:
            raise IOError(
                'Could not download the pretrained model for \'{}\' from \'{}\'. '
                'Please check that your internet connection is working, '
                'or download the model manually.'.format(dataset, download_link))
    elif dataset == 'imagenet':
        model = torchvision.models.densenet161(pretrained=True)
        model_tools.save_model(model, path)
    else:
        raise ValueError('Dataset not supported.')


def get_pytorch_model(dataset: str, path: str, download: bool) -> torch.nn.Module:
    """Returns the pretrained PyTorch model for a given dataset.

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
            download_pretrained_model(dataset, path)
        else:
            raise click.BadOptionUsage(
                'No model found: {}. Use --download-model to automatically download missing models.'.format(path))

    model = model_tools.load_model(model, path, False, False)
    return model


def get_preprocessing(dataset: str) -> model_tools.Preprocessing:
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


def get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError('Dataset not supported')


def get_attack_constructor(attack_name, p):
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


global_option_list = [
    click.option('-d', '--dataset', default='cifar10', show_default=True,
                 type=click.Choice(['cifar10', 'cifar100', 'imagenet'])),
    click.option('-df', '--data-folder', default=None, type=click.Path(file_okay=False, dir_okay=True),
                 help='The path to the folder where the dataset is stored (or will be downloaded). '
                 'If unspecified, it defaults to ./data/$dataset$'),
    click.option('-mp', '--model-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                 help='The path to the tar where the dataset is stored (or will be downloaded). '
                 'Ignored if dataset is \'imagenet\'. '
                 'If unspecified it defaults to ./pretrained_models/$dataset$.pth.tar . '),
    click.option('-rp', '--results-path', default=None, type=click.Path(file_okay=True, dir_okay=False),
                 help='The path to the CSV file where the results will be saved. If unspecified '
                 'it defaults to ./results/$command$/$datetime$.csv'),
    click.option('-b', '--batch', default=5, show_default=True,
                 type=click.IntRange(1, None)),
    click.option('-mb', '--max-batches', default=0, show_default=True, type=click.IntRange(0, None),
                 help='The maximum number of batches. 0 disables batch capping.'),
    click.option('-lw', '--loader-workers', default=2, show_default=True, type=click.IntRange(1, None),
                 help='The number of parallel workers that will load the samples from the dataset.'
                 '0 disables parallelization.'),
    click.option('-dm', '--download-model', is_flag=True,
                 help='If the model file does not exist, download the pretrained model for the corresponding dataset.'),
    click.option('-dd', '--download-dataset', is_flag=True,
                 help='If the model data files do not exist, download them.'),
    click.option('-v', '--verbose', is_flag=True)
]

attack_option_list = [
    click.option('-a', '--attack', default='deepfool', show_default=True,
                 type=click.Choice(['boundary', 'deepfool', 'fgsm'])),
    click.option('-p', default='inf', show_default=True,
                 type=click.Choice(['2', 'inf'])),
    click.option('-aw', '--attack-workers', default=5, show_default=True, type=click.IntRange(0, None),
                 help='The number of parallel workers that will be used to speed up the attack (if possible). '
                 '0 disables parallelization.')
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def global_options(func):
    @add_options(global_option_list)
    @click.pass_context
    @functools.wraps(func)
    def _parse_global_options(ctx, dataset, data_folder, model_path, results_path, batch, max_batches, loader_workers, download_dataset, download_model, verbose, *args, **kwargs):
        if data_folder is None:
            data_folder = './data/' + dataset

        if model_path is None:
            model_path = './pretrained_models/' + dataset + '.pth.tar'

        if results_path is None:
            results_path = utils.get_results_default_path(ctx.command.name)

        command = ' '.join(sys.argv[1:])

        train_loader, test_loader = get_loaders(
            dataset, data_folder, batch, loader_workers, download_dataset)

        if max_batches > 0:
            train_loader = loaders.MaxBatchLoader(train_loader, max_batches)
            test_loader = loaders.MaxBatchLoader(test_loader, max_batches)

        pytorch_model = get_pytorch_model(dataset, model_path, download_model)
        pytorch_model = torch.nn.Sequential(
            get_preprocessing(dataset), pytorch_model)
        pytorch_model.eval()

        num_classes = get_num_classes(dataset)

        foolbox_model = foolbox.models.PyTorchModel(
            pytorch_model, (0, 1), num_classes, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0, 1))

        parsed_global_options = {
            'command': command,
            'dataset': dataset,
            'foolbox_model': foolbox_model,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'pytorch_model': pytorch_model,
            'results_path': results_path,
            'verbose': verbose
        }

        return func(parsed_global_options, *args, **kwargs)
    return _parse_global_options


def attack_options(func):
    @global_options
    @add_options(attack_option_list)
    @functools.wraps(func)
    def _parse_attack_options(parsed_global_options, attack, p, attack_workers, *args, **kwargs):
        if p == '2':
            p = 2
        elif p == 'inf':
            p = np.inf

        pytorch_model = parsed_global_options['pytorch_model']

        if attack_workers == 0:
            batch_worker = None
        else:
            batch_worker = batch_attack.PyTorchWorker(pytorch_model)

        parsed_attack_options = dict(parsed_global_options)

        # We don't immediately parse 'attack' because every test needs a specific configuration
        parsed_attack_options['attack_name'] = attack

        parsed_attack_options['batch_worker'] = batch_worker
        parsed_attack_options['attack_workers'] = attack_workers
        parsed_attack_options['p'] = p

        return func(parsed_attack_options, *args, **kwargs)
    return _parse_attack_options


@click.group()
def main(*args):
    pass


@main.command()
@attack_options
def attack(options):
    command = options['command']
    attack_name = options['attack_name']
    attack_workers = options['attack_workers']
    batch_worker = options['batch_worker']
    foolbox_model = options['foolbox_model']
    test_loader = options['test_loader']
    p = options['p']
    results_path = options['results_path']
    verbose = options['verbose']

    attack_constructor = get_attack_constructor(attack_name, p)
    attack = attack_constructor(
        foolbox_model, foolbox.criteria.Misclassification(), distance_tools.LpDistance(p))

    distances, failure_count = tests.attack_test(foolbox_model, test_loader, attack, p,
                                                 batch_worker, attack_workers, verbose)

    average_distance, median_distance, success_rate, adjusted_median_distance = utils.distance_statistics(
        distances, failure_count)

    info = [['Success Rate'] + ['{:2.2f}%'.format(success_rate * 100.0)]] + [['Average Distance'] + ['{:2.2e}'.format(average_distance)]] + [
        ['Median Distance'] + ['{:2.2e}'.format(median_distance)]] + [['Adjusted Median Distance'] + ['{:2.2e}'.format(adjusted_median_distance)]]

    # TODO: Add statistics to the tests. Detection_test and its variants. ImageNet.

    header = ['Distances']

    utils.save_results(results_path, [distances], command,
                       info=info, header=header, transpose=True)


@main.command()
@global_options
@click.option('-tk', '--top-ks', nargs=2, type=int, default=(1, 5), show_default=True,
              help='The two top-k accuracies that will be computed.')
def accuracy(options, top_ks):
    command = options['command']
    foolbox_model = options['foolbox_model']
    test_loader = options['test_loader']
    results_path = options['results_path']
    verbose = options['verbose']

    accuracies = tests.accuracy_test(
        foolbox_model, test_loader, top_ks, verbose)
    accuracies = ['{:2.2f}%'.format(accuracy * 100.0)
                  for accuracy in accuracies]

    header = ['Top-{}'.format(top_k) for top_k in top_ks]
    utils.save_results(results_path, [accuracies], command, header=header)


if __name__ == '__main__':
    main()
