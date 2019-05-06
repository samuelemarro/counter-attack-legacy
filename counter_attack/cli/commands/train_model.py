import click
import torch

from counter_attack import training
from counter_attack.cli import options, parsing

@click.command()
@options.global_options
@options.dataset_options('train', 'train')
@options.train_options
@click.option('--trained-model-path', type=click.Path(file_okay=True, dir_okay=False), default=None,
              help='The path to the file where the model will be saved. If unspecified, it defaults to \'./train_model/$dataset $start_time.pth.tar\'')
def train_model(options, trained_model_path):
    cuda = options['cuda']
    dataset = options['dataset']
    epochs = options['epochs']
    loader = options['loader']
    optimiser_name = options['optimiser_name']
    start_time = options['start_time']

    if trained_model_path is None:
        trained_model_path = parsing.get_training_default_path(
            'train_model', dataset, start_time)

    torch_model = parsing.get_torch_model(dataset)
    torch_model.train()

    if cuda:
        torch_model.cuda()

    optimiser = parsing.build_optimiser(optimiser_name, torch_model.parameters(), options)

    loss = torch.nn.CrossEntropyLoss()

    training.train_torch(torch_model, loader, loss,
                         optimiser, epochs, cuda, classification=True)

    torch.save(torch_model, trained_model_path)