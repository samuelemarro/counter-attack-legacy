import collections
import random

import numpy as np
from . import batch_attack, utils


class Loader:
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class TorchLoaderWrapper(Loader):
    def __init__(self, torch_loader):
        self.torch_loader = torch_loader
        self._torch_iterator = None

    def __iter__(self):
        self._torch_iterator = iter(self.torch_loader)
        return self

    def __next__(self):
        try:
            images, labels = next(self._torch_iterator)

            images = images.numpy()
            labels = labels.numpy()

            return images, labels
        except StopIteration:
            raise

    def __len__(self):
        return len(self.torch_loader)


class ListLoader(Loader):
    def __init__(self, elements, batch_size, shuffle):
        self.elements = elements
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._iterator = None

    def __iter__(self):
        if self.shuffle:
            elements = list(self.elements)
            random.shuffle(elements)
        else:
            elements = self.elements

        batches = []

        for i in range((len(elements) + self.batch_size - 1) // self.batch_size):
            batch = elements[i * self.batch_size:(i + 1) * self.batch_size]

            if isinstance(elements[0], tuple):
                # Convert a list of tuples in separate lists
                new_batch = []
                for j in range(len(elements[0])):
                    new_batch.append([])

                for element in batch:
                    for j, value in enumerate(element):
                        new_batch[j].append(value)

                batch = new_batch

            # Convert each sub-batch (list of the same type of values) into a numpy array
            batch = [np.array(sub_batch) for sub_batch in batch]
            batches.append(batch)

        self._iterator = iter(batches)
        return self

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return (len(self.elements) + self.batch_size - 1) // self.batch_size


class MaxBatchLoader(Loader):
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self._batch_counter = 0
        self._loader_iterator = None

    def __iter__(self):
        self._batch_counter = 0
        self._loader_iterator = iter(self.loader)
        return self

    def __next__(self):
        if self._batch_counter == self.max_batches:
            raise StopIteration()

        self._batch_counter += 1

        return next(self._loader_iterator)

    def __len__(self):
        return self.max_batches
