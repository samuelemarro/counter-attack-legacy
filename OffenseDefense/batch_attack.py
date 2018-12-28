import queue
import logging
import numpy as np
import foolbox
import torch

from . import batch_processing, utils

logger = logging.getLogger(__name__)


class TorchWorker(batch_processing.BatchWorker):
    """A BatchWorker that wraps a Torch model.
    """

    def __init__(self,
                 torch_model: torch.nn.Module):
        """Initializes the TorchWorker.

        Parameters
        ----------
        torch_model : torch.nn.Module
            The Torch model that will be used to perform the predictions.

        """

        self.torch_model = torch_model

    def __call__(self, inputs):
        images = [np.array(x[0]) for x in inputs]
        labels = [np.array(x[1]) for x in inputs]

        images = torch.from_numpy(np.array(images))
        labels = torch.from_numpy(np.array(labels))

        images.requires_grad_()

        # Convert to CUDA tensors, if available
        if next(self.torch_model.parameters()).is_cuda:
            images = images.cuda()
            labels = labels.cuda()

        outputs = self.torch_model(images)

        cross_entropy = torch.nn.CrossEntropyLoss()

        losses = [cross_entropy(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(
            labels[i], 0)) for i in range(outputs.shape[0])]
        grads = torch.autograd.grad(losses, images)[0]

        outputs = [output.cpu().detach().numpy()[0]
                   for output in torch.split(outputs, 1)]
        grads = [grad.cpu().numpy()[0] for grad in torch.split(grads, 1)]

        return zip(outputs, grads)


class QueueAttackWorker(batch_processing.ThreadWorker):
    """A ThreadWorker that supports attacking samples in a queue-like
    fashion. Useful for situations where the input_queue is bigger than
    the model can support.
    """

    def __init__(self,
                 attack: foolbox.attacks.Attack,
                 foolbox_model: foolbox.models.Model,
                 input_queue: queue.Queue):
        """Initializes the QueueAttackWorker.

        Parameters
        ----------
        attack : foolbox.attacks.Attack
            The attack that will be used to find adversarial samples.
        foolbox_model : foolbox.models.Model
            The model that will be attacked.
        input_queue : queue.Queue
            The Queue from which the worker will read the images to
            attack. Must be the same Queue passed to run_queue_threads.

        """

        self.attack = attack
        self.foolbox_model = foolbox_model
        self.input_queue = input_queue

    def __call__(self, pooler, return_queue):
        parallel_model = ParallelModel(pooler,
                                       self.foolbox_model,
                                       self.foolbox_model.bounds(),
                                       self.foolbox_model.channel_axis(),
                                       self.foolbox_model._preprocessing)
        while True:
            try:
                i, (image, label) = self.input_queue.get(timeout=1e-2)
                adversarial = foolbox.Adversarial(parallel_model,
                                                  self.attack._default_criterion,
                                                  image,
                                                  label,
                                                  self.attack._default_distance,
                                                  self.attack._default_threshold)
                adversarial_image = self.attack(adversarial)

                return_queue.put((i, adversarial_image))
            except queue.Empty:
                # No more inputs, we can stop
                return


class ParallelModel(foolbox.models.DifferentiableModel):
    def __init__(self, pooler, foolbox_model, bounds, channel_axis, preprocessing=(0, 1)):
        super().__init__(bounds, channel_axis, preprocessing)
        self.pooler = pooler
        self.foolbox_model = foolbox_model

    def predictions_and_gradient(self, image, label):
        return self.pooler.call((image, label))

    def backward(self, gradient, image):
        raise NotImplementedError()

    def batch_predictions(self, images):
        return self.foolbox_model.batch_predictions(images)

    def num_classes(self):
        return self.foolbox_model.num_classes()


def run_batch_attack(foolbox_model, batch_worker, attack, images, labels, num_workers=50):
    assert len(images) == len(labels)

    input_queue = queue.Queue()
    data = zip(images, labels)

    attack_workers = [QueueAttackWorker(
        attack, foolbox_model, input_queue) for _ in range(num_workers)]

    results = batch_processing.run_queue_threads(
        batch_worker, attack_workers, input_queue, data)

    assert len(results) == len(images)

    return results


def run_individual_attack(attack, images, labels):
    assert len(images) == len(labels)

    adversarials = []

    for image, label in zip(images, labels):
        adversarial = attack(image, label)
        adversarials.append(adversarial)

    return adversarials


def get_correct_samples(foolbox_model: foolbox.models.PyTorchModel,
                        images: np.ndarray,
                        labels: np.ndarray):
    _filter = utils.Filter()
    _filter['images'] = images
    _filter['ground_truth_labels'] = labels

    _filter['image_predictions'] = foolbox_model.batch_predictions(
        _filter['images'])
    _filter['image_labels'] = np.argmax(_filter['image_predictions'], axis=-1)
    correctly_classified = np.nonzero(np.equal(_filter['image_labels'],
                                               _filter['ground_truth_labels']))[0]

    _filter.filter(correctly_classified, 'successful_classification')

    return _filter['images'], _filter['image_labels']


"""
Finds the adversarial samples.

Note: Some adversarial samples might sometimes be non-adversarial, due to the fact that
they're close to the boundary and can switch class depending on the approximation.
"""


def get_adversarials(foolbox_model: foolbox.models.PyTorchModel,
                     images: np.ndarray,
                     labels: np.ndarray,
                     adversarial_attack: foolbox.attacks.Attack,
                     remove_failed: bool,
                     batch_worker: batch_processing.BatchWorker = None,
                     num_workers: int = 50):
    _filter = utils.Filter()
    _filter['images'] = images
    _filter['image_labels'] = labels

    if batch_worker is not None:
        _filter['adversarials'] = run_batch_attack(foolbox_model,
                                                   batch_worker,
                                                   adversarial_attack,
                                                   _filter['images'],
                                                   _filter['image_labels'],
                                                   num_workers=num_workers)
    else:
        _filter['adversarials'] = run_individual_attack(
            adversarial_attack, _filter['images'], _filter['image_labels'])

    successful_adversarial_indices = [i for i in range(
        len(_filter['adversarials'])) if _filter['adversarials'][i] is not None]
    successful_adversarial_indices = np.array(
        successful_adversarial_indices, dtype=np.int)

    if remove_failed:
        _filter.filter(successful_adversarial_indices,
                       'successful_adversarial')

        # If there are no successful attacks, return early
        if len(successful_adversarial_indices) == 0:
            logger.warning(
                'Could not find an adversarial sample for any of the provided samples.')
            return _filter

        # Convert to Numpy array after the failed samples have been removed
        _filter['adversarials'] = np.array(_filter['adversarials'])

    return _filter['adversarials'], _filter['images'], _filter['image_labels']
