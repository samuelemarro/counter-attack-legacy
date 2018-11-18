import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import OffenseDefense.utils as utils

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

def cifar10_trainloader(num_workers, batch_size, flip, crop, normalize, shuffle):
    transformations = [transforms.ToTensor()]
    if flip:
        transformations.append(transforms.RandomHorizontalFlip())
    if crop:
        transformations.append(transforms.RandomCrop(32, padding=4))
    if normalize:
        transformations.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    dataset = datasets.CIFAR10(root='./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose(transformations))
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

def cifar10_testloader(num_workers, batch_size, normalize, shuffle):
    transformations = [transforms.ToTensor()]

    if normalize:
        transformations.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    dataset = datasets.CIFAR10(root='./data',
                               train=False,
                               download=False,
                               transform=transforms.Compose(transformations))
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
