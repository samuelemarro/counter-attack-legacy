import queue
import logging
import numpy as np
import foolbox
import torch

from . import batch_processing, utils

logger = logging.getLogger(__name__)

class ModelWorker(batch_processing.BatchWorker):
    def has_gradient(self):
        return NotImplementedError()

class TorchModelWorker(ModelWorker):
    """
    A ModelWorker that wraps a Torch model (with autograd support).
    """

    def __init__(self,
                 torch_model: torch.nn.Module):
        """Initializes the TorchModelWorker.

        Parameters
        ----------
        torch_model : torch.nn.Module
            The Torch model that will be used to perform the predictions and compute the gradients.

        """

        self.torch_model = torch_model

    def has_gradient(self):
        return True

    def batch_predictions(self, data):
        images = data
        images = torch.from_numpy(np.array(images))

        if next(self.torch_model.parameters()).is_cuda:
            images = images.cuda()

        outputs = self.torch_model(images)
        outputs = [output.cpu().detach().numpy()[0]
                for output in torch.split(outputs, 1)]

        return outputs

    def batch_predictions_and_gradients(self, data):
        images = [np.array(x[0]) for x in data]
        labels = [np.array(x[1]) for x in data]

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

        return list(zip(outputs, grads))

    def __call__(self, inputs):
        get_grad = np.array([x[0] for x in inputs])
        data = [x[1] for x in inputs]

        gradient_indices = [i for i in range(len(get_grad)) if get_grad[i]]
        predictions_indices = [i for i in range(len(get_grad)) if not get_grad[i]]

        gradient_data = []
        predictions_data = []

        for gradient_index in gradient_indices:
            gradient_data.append(data[gradient_index])

        for predictions_index in predictions_indices:
            predictions_data.append(data[predictions_index])

        if gradient_data:
            gradient_outputs = self.batch_predictions_and_gradients(gradient_data)
        else:
            gradient_outputs = []

        if predictions_data:
            predictions_outputs = self.batch_predictions(predictions_data)
        else:
            predictions_outputs = []

        assert len(gradient_data) == len(gradient_outputs)
        assert len(predictions_data) == len(predictions_outputs)

        outputs = [None] * len(inputs)

        for gradient_index, gradient_output in zip(gradient_indices, gradient_outputs):
            outputs[gradient_index] = gradient_output

        for predictions_index, predictions_output in zip(predictions_indices, predictions_outputs):
            outputs[predictions_index] = predictions_output

        for output in outputs:
            assert output is not None

        return outputs


class FoolboxModelWorker(ModelWorker):
    """
    A BatchWorker that wraps a Foolbox model.
    """

    def __init__(self,
                 foolbox_model: foolbox.models.Model):
        """Initializes the TorchModelWorker.

        Parameters
        ----------
        foolbox_model : torch.nn.Module
            The foolbox model that will be used to perform the predictions.

        """

        self.foolbox_model = foolbox_model

    def has_gradient(self):
        return True

    def __call__(self, inputs):
        get_grad = np.array([x[0] for x in inputs])

        # Gradient is not supported
        assert np.all(np.logical_not(get_grad))

        data = [x[1] for x in inputs]

        images = np.array(data)
        outputs = self.foolbox_model.batch_predictions(images)
        return outputs

class CompositeModelWorker(ModelWorker):
    """
    A BatchWorker that uses a Foolbox model for predictions and a Torch Model for gradients.
    """

    def __init__(self, predictions_foolbox_model, estimate_torch_model):
        self.predictions_foolbox_worker = FoolboxModelWorker(predictions_foolbox_model)
        self.estimate_torch_worker = TorchModelWorker(estimate_torch_model)

    def has_gradient(self):
        return True

    def __call__(self, inputs):
        #print(inputs[0][0].shape)

        data = [x[1] for x in inputs]
        images = [np.array(x[0]) for x in data]

        foolbox_inputs = [(False, image) for image in images]
        foolbox_predictions = self.predictions_foolbox_worker(foolbox_inputs)


        torch_predictions_and_grads = self.estimate_torch_worker(inputs)

        torch_grads = [grad for _, grad in torch_predictions_and_grads]

        assert len(foolbox_predictions) == len(torch_grads)
        
        return zip(foolbox_predictions, torch_grads)


class QueueAttackWorker(batch_processing.ThreadWorker):
    """A ThreadWorker that supports attacking samples in a queue-like
    fashion. Useful for situations where the batch size is bigger than
    what the model can support.
    """

    def __init__(self,
                 attack: foolbox.attacks.Attack,
                 gradient: bool,
                 foolbox_model: foolbox.models.Model,
                 input_queue: queue.Queue):
        """Initializes the QueueAttackWorker.

        Parameters
        ----------
        attack : foolbox.attacks.Attack
            The attack that will be used to find adversarial samples.
        gradient : bool
            Whether to use gradient computation.
        foolbox_model : foolbox.models.Model
            The model that will be attacked.
        input_queue : queue.Queue
            The Queue from which the worker will read the images to
            attack. Must be the same Queue passed to run_queue_threads.

        """

        self.attack = attack
        self.gradient = gradient
        self.foolbox_model = foolbox_model
        self.input_queue = input_queue

    def __call__(self, pooler, return_queue):
        if self.gradient:
            parallel_model = DifferentiableParallelModel(
                pooler, self.foolbox_model)
        else:
            parallel_model = ParallelModel(pooler,
                                           self.foolbox_model)
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


class ParallelModel(foolbox.models.ModelWrapper):
    def __init__(self, pooler, foolbox_model):
        super().__init__(foolbox_model)
        self.pooler = pooler

    def predictions(self, image):
        return self.pooler.call((False, image))


class DifferentiableParallelModel(foolbox.models.DifferentiableModelWrapper):
    def __init__(self, pooler, foolbox_model):
        super().__init__(foolbox_model)
        self.pooler = pooler

    def predictions_and_gradient(self, image, label):
        return self.pooler.call((True, (image, label)))


def run_batch_attack(foolbox_model, batch_worker, attack, images, labels, num_workers):
    assert len(images) == len(labels)

    input_queue = queue.Queue()
    data = list(zip(images, labels))

    gradient = batch_worker.has_gradient()

    attack_workers = [QueueAttackWorker(
        attack, gradient, foolbox_model, input_queue) for _ in range(num_workers)]

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


def get_correct_samples(foolbox_model: foolbox.models.Model,
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

    _filter.filter(correctly_classified)

    return _filter['images'], _filter['image_labels']


def get_approved_samples(foolbox_model: foolbox.models.Model,
                         images: np.ndarray,
                         labels: np.ndarray,
                         rejector):
    _filter = utils.Filter()
    _filter['images'] = images
    _filter['image_labels'] = labels

    batch_valid = rejector.batch_valid(_filter['images'])
    approved_indices = np.nonzero(batch_valid)[0]
    _filter.filter(approved_indices)

    return _filter['images'], _filter['image_labels']


"""
Finds the adversarial samples.

Note: Some adversarial samples might sometimes be non-adversarial, due to the fact that
they're close to the boundary and can switch class depending on the approximation.
"""


def get_adversarials(foolbox_model: foolbox.models.Model,
                     images: np.ndarray,
                     labels: np.ndarray,
                     adversarial_attack: foolbox.attacks.Attack,
                     remove_failed: bool,
                     batch_worker: batch_processing.BatchWorker = None,
                     num_workers: int = 50):
    if len(images) != len(labels):
        raise ValueError('images and labels must have the same length.')

    _filter = utils.Filter()
    _filter['images'] = images
    _filter['image_labels'] = labels

    if batch_worker is None:
        _filter['adversarials'] = run_individual_attack(
            adversarial_attack, _filter['images'], _filter['image_labels'])

    else:
        _filter['adversarials'] = run_batch_attack(foolbox_model,
                                                   batch_worker,
                                                   adversarial_attack,
                                                   _filter['images'],
                                                   _filter['image_labels'],
                                                   num_workers)

    successful_adversarial_indices = [i for i in range(
        len(_filter['adversarials'])) if _filter['adversarials'][i] is not None]
    successful_adversarial_indices = np.array(
        successful_adversarial_indices, dtype=np.int)

    if remove_failed:
        _filter.filter(successful_adversarial_indices)

        # Convert to Numpy array after the failed samples have been removed
        _filter['adversarials'] = np.array(_filter['adversarials'])

    return _filter['adversarials'], _filter['images'], _filter['image_labels']
