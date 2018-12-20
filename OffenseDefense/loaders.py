import collections
import numpy as np
from . import batch_attack, utils


class Loader:
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class TorchLoader(Loader):
    def __init__(self, torch_loader):
        self.torch_loader = torch_loader
        self.torch_iterator = None

    def __iter__(self):
        self.torch_iterator = iter(self.torch_loader)
        return self

    def __next__(self):
        try:
            images, labels = next(self.torch_iterator)

            images = images.numpy()
            labels = labels.numpy()

            return images, labels
        except StopIteration:
            raise

    def __len__(self):
        return len(self.torch_loader)


class DetectorLoader(Loader):
    def __init__(self, images_loader, detector, failure_value):
        self.values = []
        self.iterator = None

        for images, _ in images_loader:
            f = utils.Filter()
            f['images'] = images

            scores = detector.get_scores(images)

            f['scores'] = scores
            valid_indices = [i for i in range(
                len(images)) if scores[i] is not None]
            valid_indices = np.array(valid_indices)
            f.filter(valid_indices)

            self.values.append((images, np.array(scores)))

    def __iter__(self):
        self.iterator = iter(self.values)
        return self

    def __next__(self):
        try:
            images, scores = next(self.iterator)

            return images, scores
        except StopIteration:
            raise

    def __len__(self):
        return len(self.values)
