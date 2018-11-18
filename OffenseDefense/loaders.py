import collections
import OffenseDefense.batch_attack as batch_attack

class Loader:
    def __iter__(self):
        return self
    def __next__(self):
        pass

class TorchLoader(Loader):
    def __init__(self, torch_loader):
        self.torch_iterator = iter(torch_loader)

    def __next__(self):
        try:
            images, labels = next(self.torch_iterator)

            images = images.numpy()
            labels = labels.numpy()

            return images, labels
        except StopIteration:
            raise

class AdversarialLoader(Loader):
    def __init__(self, loader, foolbox_model, attack, parallelize):
        self.loader_iterator = iter(loader)
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.parallelize = parallelize

    def __next__(self):
        try:
            images, labels = next(self.loader_iterator)

            adversarial_filter = batch_attack.get_adversarials(self.foolbox_model, images, labels, self.attack, self.parallelize)
            return adversarial_filter['adversarials'], adversarial_filter['adversarial_labels']

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
        
        output = np.random.uniform(self.output_min, self.output_max, [self.batch_size] + self.output_shape)

        self._batch_counter += 1
        print(output.shape)

        return output, self.foolbox_model.batch_predictions(output)
