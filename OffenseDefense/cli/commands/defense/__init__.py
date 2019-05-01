
import click

from . import model, preprocessor, rejector

@click.group(name='defense')
def defense():
    """
    Uses defenses against various attack strategies.
    """
    pass

defense.add_command(model.model_defense)
defense.add_command(preprocessor.preprocessor_defense)
defense.add_command(rejector.rejector_defense)