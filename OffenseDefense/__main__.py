import click
import logging
import sys

from OffenseDefense.cli import commands

# TODO: Test preprocessing options
# TODO: Allow for optional model weights?
# TODO: Check that the pretrained model does not contain normalisation inside?
# TODO: British vs American spelling
# TODO: Download the cifar100 weights for densenet-bc-100-12 (when available)
# TODO: Upload both of them and update the links in config.ini
# TODO: Sanity check: Difference between the original model and its trained approximator
# TODO: Verify that the transfer is successful
# TODO: Complete the substitutes
# TODO: In pretrained_model, you are passing the model path, not the weights one
# TODO: standard and parallel are treated completely differently, and might have different models or attacks
# TODO: Not all rejectors use [2|inf] from distance_tool_options. Merge distance_tool with counter_attack?
# TODO: preprocessor and model, from a defense point of view, are the same. The only difference is the arguments.
# I could theoretically load a foolbox model and use it, no matter what it contains.
# TODO: Threshold support for RandomDirectionAttack?
# TODO: In parallelization, attack workers should be an argument
# TODO: Check if rejector substitute has n+1 classes
# TODO: "Valid class" substitute attack

# IMPORTANT:
# Shallow attacks the standard model, then it is evaluated on the defended model
# Substitute and Black-Box attack the defended model
# This means that you cannot write the sanity check "Shallow is the same as
# a Substitute that uses the original as gradient estimator"

# Note: Not all adversarial attacks are successful. This means that an approximation
# dataset will have slightly less adversarial samples. This unbalanced dataset should
# not cause problems when training approximators, but it might cause problems when
# training adversarial classifiers.

# Note: When preparing the adversarial dataset, the model accuracy will always be around 100%
# because you're running the attack on the train set

# Note: For ImageNet, we evaluate on the validation set

# Note: We use L_p bound-normalized distance, which is averaged if p is not inf

@click.group()
def main(*args):
    logging.basicConfig()
    logging.captureWarnings(True)

    # Print the messages to console
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    root.addHandler(handler)

for command in commands:
    main.add_command(command)

if __name__ == '__main__':
    main()
