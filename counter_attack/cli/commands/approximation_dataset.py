import pathlib

import click
import torch

from OffenseDefense import defenses, rejectors, training, utils
from OffenseDefense.cli import definitions, options, parsing

@click.group(name='approximation-dataset')
def approximation_dataset():
    pass

@approximation_dataset.command(name='preprocessor')
@options.global_options
@options.dataset_options('train', 'train')
@options.standard_model_options
@options.pretrained_model_options
@options.preprocessor_options
@options.adversarial_dataset_options
@options.approximation_dataset_options('preprocessor')
def approximation_dataset_preprocessor(options):
    """
    Generates the dataset to train a substitute model for models
    with preprocessors.

    Saves the labels predicted by the defended model, using the genuine
    dataset + an adversarial dataset. 
    """
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    preprocessor = options['preprocessor']

    defended_model = defenses.PreprocessorDefenseModel(
        foolbox_model, preprocessor)


    genuine_approximation_dataset = training.generate_approximation_dataset(defended_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(defended_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)

@approximation_dataset.command(name='model')
@options.global_options
@options.dataset_options('train', 'train')
@options.standard_model_options
@options.custom_model_options
@options.adversarial_dataset_options
@options.approximation_dataset_options('model')
def approximation_dataset_model(options):
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    custom_foolbox_model = options['custom_foolbox_model']
    genuine_loader = options['loader']

    genuine_approximation_dataset = training.generate_approximation_dataset(custom_foolbox_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(custom_foolbox_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)

@approximation_dataset.command(name='rejector')
@options.global_options
@options.dataset_options('train', 'train')
@options.standard_model_options
@options.pretrained_model_options
@options.distance_tool_options
@options.counter_attack_options(False)
@options.detector_options
@options.rejector_options
@options.adversarial_dataset_options
@options.approximation_dataset_options('rejector')
def approximation_dataset_rejector(options):
    adversarial_loader = options['adversarial_loader']
    approximation_dataset_path = options['approximation_dataset_path']
    foolbox_model = options['foolbox_model']
    genuine_loader = options['loader']
    rejector = options['rejector']

    defended_model = rejectors.RejectorModel(foolbox_model, rejector)

    genuine_approximation_dataset = training.generate_approximation_dataset(defended_model, genuine_loader, 'Genuine Approximation Dataset')
    adversarial_approximation_dataset = training.generate_approximation_dataset(defended_model, adversarial_loader, 'Adversarial Approximation Dataset')

    approximation_dataset = genuine_approximation_dataset + adversarial_approximation_dataset

    utils.save_zip(approximation_dataset, approximation_dataset_path)
