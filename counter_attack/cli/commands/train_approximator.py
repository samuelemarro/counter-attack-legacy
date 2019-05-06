import pathlib

import click
import torch

from counter_attack import loaders, model_tools, training, utils
from counter_attack.cli import definitions, options, parsing

@click.command()
@options.global_options
@options.train_options
@click.argument('defense_type', type=click.Choice(['model', 'preprocessor', 'rejector']))
@click.argument('approximation_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--base-weights-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the file where the base weights are stored. If unspecified, it defaults to the pretrained model for the dataset.')
@click.option('--normalisation', default=None,
            help='The normalisation that will be applied by the model. Supports both dataset names ({}) and '
            'channel stds-means (format: "red_mean green_mean blue_mean red_stdev green_stdev blue_stdev" including quotes).'.format(', '.join(definitions.datasets)))
@click.option('--trained-approximator-path', type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the approximator will be saved. If unspecified, it defaults to \'./trained_models/train_approximator/$defense_type/$dataset $start_time.pth.tar\'')
def train_approximator(options, defense_type, approximation_dataset_path, base_weights_path, normalisation, trained_approximator_path):
    batch_size = options['batch_size']
    cuda = options['cuda']
    dataset = options['dataset']
    epochs = options['epochs']
    max_batches = options['max_batches']
    optimiser_name = options['optimiser_name']
    shuffle = options['shuffle']
    start_time = options['start_time']

    if base_weights_path is None:
        base_weights_path = './pretrained_models/' + dataset + '_weights.pth.tar'

    if trained_approximator_path is None:
        trained_approximator_path = parsing.get_training_default_path(
            'train_approximator/' + defense_type, dataset, start_time)

    is_rejector = defense_type == 'rejector'
    model = parsing.get_torch_model(dataset, is_rejector)
    model_tools.load_partial_state_dict(model, base_weights_path)
    last_layer = model_tools.get_last_layer(model)

    model = parsing.apply_normalisation(model, normalisation, 'model', '--normalisation')

    model.train()

    if cuda:
        model.cuda()

    # Reinitialise the last layer
    last_layer.reset_parameters()

    optimiser = parsing.build_optimiser(optimiser_name, last_layer.parameters(), options)

    # Build the pretrained model and the final model (which might be n+1)
    # Transfer some layers (which? how?)
    # Train the remaining layers of the final model
    # Save the model

    approximation_data = utils.load_zip(approximation_dataset_path)
    loader = loaders.ListLoader(approximation_data, batch_size, shuffle)

    if max_batches is not None:
        loader = loaders.MaxBatchLoader(loader, max_batches)

    training.train_torch(model, loader, torch.nn.CrossEntropyLoss(),
                         optimiser, epochs, cuda, classification=True)

    pathlib.Path(trained_approximator_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model, trained_approximator_path)