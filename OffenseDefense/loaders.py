import collections
import numpy as np
import OffenseDefense.batch_attack as batch_attack

class Loader:
    def __iter__(self):
        return self
    def __next__(self):
        pass

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

class AdversarialLoader(Loader):
    def __init__(self, loader, foolbox_model, attack, adversarial_labels, batch_worker=None, num_workers=50):
        self.loader = loader
        self.loader_iterator = None
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.adversarial_labels = adversarial_labels
        self.batch_worker = batch_worker
        self.num_workers = num_workers

    def __iter__(self):
        self.loader_iterator = iter(self.loader)
        return self

    def __next__(self):
        try:
            images, labels = next(self.loader_iterator)

            adversarial_filter = batch_attack.get_adversarials(self.foolbox_model,
                                                               images,
                                                               labels,
                                                               self.attack,
                                                               self.batch_worker,
                                                               self.num_workers)

            adversarials = adversarial_filter['adversarials']
            if self.adversarial_labels:
                output_labels = adversarial_filter['adversarial_labels']
            else:
                output_labels = adversarial_filter['image_labels']
            return adversarials, output_labels

        except StopIteration:
            raise

class RandomNoiseLoader:
    def __init__(self, foolbox_model, output_min, output_max, output_shape, batch_size, batch_count):
        self.foolbox_model = foolbox_model
        self.output_min = output_min
        self.output_max = output_max

        if isinstance(output_shape, collections.Iterable):
            self.output_shape = output_shape
        else:
            self.output_shape = [output_shape]

        self.batch_size = batch_size
        self.batch_count = batch_count
        self._batch_counter = 0

    def __iter__(self):
        self._batch_counter = 0
        return self

    def __next__(self):
        if self._batch_counter == self.batch_count:
            raise StopIteration()
        
        samples = np.random.uniform(self.output_min, self.output_max, [self.batch_size] + self.output_shape).astype(np.float32)
        labels = np.argmax(self.foolbox_model.batch_predictions(samples), axis=1).astype(int)

        self._batch_counter += 1

        return samples, labels
