import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from . import utils


class Preprocessing(torch.nn.Module):
    def __init__(self, means, stds):
        super(Preprocessing, self).__init__()
        self.means = torch.from_numpy(np.array(means).reshape((3, 1, 1)))
        self.stds = torch.from_numpy(np.array(stds).reshape((3, 1, 1)))

    def forward(self, input):
        means = self.means.to(input)
        stds = self.stds.to(input)
        return (input - means) / stds


class AdversarialDataset(data.Dataset):
    def __init__(self, path, transform=None, count_limit=None):
        try:
            self.dataset = utils.load_zip(path)
            self.count_limit = None
        except:
            raise ValueError('Invalid path')

        self.transform = transform

    def __getitem__(self, index):
        image, label, is_adversarial = self.dataset[index]
        image = torch.from_numpy(image)
        label = torch.FloatTensor([label])
        is_adversarial = torch.FloatTensor([is_adversarial])
        if self.transform is not None:
            image = self.transform(image)

        return image, label, is_adversarial

    def __len__(self):
        if self.count_limit is None:
            return len(self.dataset)
        return self.count_limit


def load_model(base_model, path, training_model, data_parallel):
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


def save_model(model, path):
    torch.save(model.state_dict(), path)
