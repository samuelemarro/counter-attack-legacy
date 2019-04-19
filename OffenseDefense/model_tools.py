import pathlib

import foolbox
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from . import utils

def max_batch_predictions(foolbox_model, images, max_batch_size):
    batch_predictions = []
    split_images = []
    minibatch_count = (
        len(images) + max_batch_size - 1) // max_batch_size

    for i in range(minibatch_count):
        split_images.append(
            np.array(images[i * max_batch_size:(i + 1) * max_batch_size]))

    for subsection in split_images:
        subsection_predictions = foolbox_model.batch_predictions(
            subsection.astype(np.float32))

        batch_predictions += list(subsection_predictions)

    return np.array(batch_predictions)

class MaxBatchModel(foolbox.models.DifferentiableModelWrapper):
    def __init__(self, foolbox_model, max_batch_size):
        super().__init__(foolbox_model)
        self.max_batch_size = max_batch_size

    def batch_predictions(self, images):
        if len(images) > self.max_batch_size:
            return max_batch_predictions(self.wrapped_model, images, self.max_batch_size)
        else:
            return self.wrapped_model.batch_predictions(images)


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
