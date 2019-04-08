import pathlib

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from . import utils


class Normalisation(torch.nn.Module):
    def __init__(self, means, stdevs):
        super().__init__()
        self.means = torch.from_numpy(np.array(means).reshape((3, 1, 1)))
        self.stdevs = torch.from_numpy(np.array(stdevs).reshape((3, 1, 1)))

    def forward(self, input):
        means = self.means.to(input)
        stdevs = self.stdevs.to(input)
        return (input - means) / stdevs


def load_state_dict(base_model, path, training_model, data_parallel):
    if data_parallel:
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    checkpoint = torch.load(path)

    if training_model:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if data_parallel:
        model = model.module

    return model


def save_state_dict(model, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_partial_state_dict(model, path):
    pass
    # TODO:
    # Find all the keys in the target
    # Load the dict key by key
    # If there's a key or shape mismatch, skip
    # Return all parameters that were not updated

def has_normalisation(module):
    assert isinstance(module, torch.nn.Module)

    if isinstance(module, Normalisation):
        return True

    for _module in module.modules():
        assert isinstance(_module, torch.nn.Module)

        if isinstance(_module, Normalisation):
            return True

    return False
