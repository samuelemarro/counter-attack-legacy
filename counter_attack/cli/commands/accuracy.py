import click

from counter_attack import tests, utils
from counter_attack.cli import options

@click.command()
@options.global_options
@options.standard_model_options
@options.pretrained_model_options
@options.dataset_options('test', 'test')
@options.test_options('accuracy')
@click.option('--top-ks', nargs=2, type=click.Tuple([int, int]), default=(1, 5), show_default=True,
              help='The two top-k accuracies that will be computed.')
def accuracy(options, top_ks):
    """
    Computes the accuracy of the model.

    \b
    Stores the following results:
        Top-K Accuracies: The accuracies, where k values are configurable with --top-ks.
    """

    command = options['command']
    foolbox_model = options['foolbox_model']
    loader = options['loader']
    results_path = options['results_path']

    accuracies = tests.accuracy_test(
        foolbox_model, loader, top_ks)

    info = [['Top-{} Accuracy:'.format(top_k), '{:2.2f}%'.format(accuracy * 100.0)]
            for top_k, accuracy in zip(top_ks, accuracies)]
    utils.save_results(results_path, command=command, info=info)