import configparser
import logging
import queue

import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

import OffenseDefense
import OffenseDefense.attacks as attacks
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.detectors as detectors
import OffenseDefense.distance_tools as distance_tools
import OffenseDefense.loaders as loaders
import OffenseDefense.model_tools as model_tools
import OffenseDefense.models as model
import OffenseDefense.tests as tests
import OffenseDefense.training as training
import OffenseDefense.utils as utils
import OffenseDefense.rejectors as rejectors
from OffenseDefense.models.pytorch.cifar.densenet import DenseNet


def prepare_model():
    base_model = DenseNet(100, num_classes=10)
    model = model_tools.load_model(
        base_model, './pretrained_models/cifar10/densenet-bc-100-12/model_best.pth.tar', True, True)
    model = nn.Sequential(model_tools.Preprocessing(means=(0.4914, 0.4822, 0.4465),
                                                    stds=(0.2023, 0.1994, 0.2010)), model)
    model.eval()

    return model


def load_pretrained_model(model_name, dataset, path, download=True):
    model_name = model_name.lower()
    dataset = dataset.lower()

    if dataset in ['cifar10', 'cifar100']:
        model_module = OffenseDefense.models.pytorch.cifar
    elif dataset == 'imagenet':
        if model_name in ['resnext', 'resnext50', 'resnext101', 'resnext152']:
            model_module = OffenseDefense.models.pytorch.imagenet
        else:
            # Pretrained ImageNet models published by Pytorch
            return getattr(torchvision.models, model_name)(pretrained=True)

    # Call the model_name constructor
    model = getattr(model_module, model_name)()

    # Find the correct download link
    try:
        config = configparser.ConfigParser()
        config.read('model_download_links.ini')
        download_link = config[dataset][model_name]
    except KeyError:
        raise ValueError('Unsupported model "{}" for dataset "{}". Check the available '
                         'models in model_download_links.ini'.format(model_name, dataset))

    # TODO: Complete


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

if __name__ == '__main__':
    # main()
    batch_main()
