import copy
import logging
import multiprocessing
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

        # Either all require grad, or none
        # assert np.all(get_grad) or np.all(np.logical_not(get_grad))

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
        images = np.array(inputs)
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
        images = [np.array(x[0]) for x in inputs]
        foolbox_predictions = self.predictions_foolbox_worker(images)
        torch_predictions_and_grads = self.estimate_torch_worker(inputs)

        torch_grads = [grad for _, grad in torch_predictions_and_grads]

        assert len(foolbox_predictions) == len(torch_grads)
        
        return zip(foolbox_predictions, torch_grads)


class AttackWorker(batch_processing.ThreadWorker):
    def __init__(self,
                 attack: foolbox.attacks.Attack,
                 gradient: bool,
                 template_foolbox_model : foolbox.models.Model):
        """Initializes the AttackWorker.

        Parameters
        ----------
        attack : foolbox.attacks.Attack
            The attack that will be used to find adversarial samples.
        gradient : bool
            Whether to use gradient computation.
        template_foolbox_model : foolbox.models.Model
            This Foolbox model will be used to define the properties
            (i.e. num_classes, bounds...) of the parallel model.

        """
        attack = copy.copy(attack)
        attack._default_model = None

        self.attack = attack
        self.gradient = gradient
        self.num_classes = template_foolbox_model.num_classes()
        self.bounds = template_foolbox_model.bounds()
        self.channel_axis = template_foolbox_model.channel_axis()

    def __call__(self, _input, pooler_interface):
        if self.gradient:
            parallel_model = DifferentiableParallelModel(
                pooler_interface, self.num_classes, self.bounds, self.channel_axis)
        else:
            parallel_model = ParallelModel(pooler_interface, self.num_classes, self.bounds, self.channel_axis)

        image, label = _input
        adversarial = foolbox.Adversarial(parallel_model,
                                        self.attack._default_criterion,
                                        image,
                                        label,
                                        self.attack._default_distance,
                                        self.attack._default_threshold)
        adversarial_image = self.attack(adversarial)
        
        return adversarial_image

def __identity(x):
    return x

class ParallelModel:
    def __init__(self, pooler_interface, num_classes, bounds, channel_axis):
        assert len(bounds) == 2
        self._bounds = bounds
        self._channel_axis = channel_axis
        self._num_classes = num_classes

        #print('Creating standard!')
        
        self.pooler_interface = pooler_interface
        
    def _process_input(self, x):
        return x, __identity

    def _process_gradient(self, backward, dmdp):
        """
        backward: `callable`
            callable that backpropagates the gradient of the model w.r.t to
            preprocessed input through the preprocessing to get the gradient
            of the model's output w.r.t. the input before preprocessing
        dmdp: gradient of model w.r.t. preprocessed input
        """
        if backward is None:  # pragma: no cover
            raise ValueError('Your preprocessing function does not provide'
                             ' an (approximate) gradient')
        dmdx = backward(dmdp)
        assert dmdx.dtype == dmdp.dtype
        return dmdx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def batch_predictions(self, images):
        # Input format: ((get_grad, images), batch)
        return self.pooler_interface.call((False, images), True)

    def predictions(self, image):
        # Input format: ((get_grad, image), batch)
        return self.pooler_interface.call((False, image), False)

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def num_classes(self):
        return self._num_classes


class DifferentiableParallelModel(ParallelModel):
    def __init__(self, pooler_interface, num_classes, bounds, channel_axis):
        #print('Creating differentiable!')
        super().__init__(pooler_interface, num_classes, bounds, channel_axis)

    def predictions_and_gradient(self, image, label):
        #print('Passing label {}'.format(label))
        # Input format: ((get_grad, (image, label)), batch)
        return self.pooler_interface.call((True, (image, label)), False)

    def gradient(self, image, label):
        _, grad = self.predictions_and_gradient(image, label)
        return grad
        #logger.error('Called gradient on DifferentiableParallelModel')
        #raise NotImplementedError()

    def backward(self, gradient, image):
        logger.error('Called backward on DifferentiableParallelModel')
        raise NotImplementedError()


def run_batch_attack(parallel_pooler, images, labels):
    data = list(zip(images, labels))
    adversarials = parallel_pooler.run(data)

    assert len(adversarials) == len(images)

    return adversarials


def run_individual_attack(attack, foolbox_model, images, labels):
    assert len(images) == len(labels)

    attack = copy.copy(attack)
    attack._default_model = foolbox_model #TODO: Ugly
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

    _filter.filter(correctly_classified, 'successful_classification')

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
    _filter.filter(approved_indices, name='Approved')

    return _filter['images'], _filter['image_labels']

def get_adversarials(images: np.ndarray,
                     labels: np.ndarray,
                     foolbox_model : foolbox.models.Model,
                     adversarial_attack: foolbox.attacks.Attack,
                     parallel_pooler,
                     remove_failed: bool):
    """
    Finds adversarial samples.

    Note: Some adversarial samples might sometimes be non-adversarial, due to the fact that
    they're close to the boundary and can switch class depending on the approximation.
    """
    if len(images) != len(labels):
        raise ValueError('images and labels must have the same length.')

    _filter = utils.Filter()
    _filter['images'] = images
    _filter['image_labels'] = labels

    if parallel_pooler is None:
        _filter['adversarials'] = run_individual_attack(
            foolbox_model, adversarial_attack, _filter['images'], _filter['image_labels'])

    else:
        _filter['adversarials'] = run_batch_attack(parallel_pooler, images, labels)

    successful_adversarial_indices = [i for i in range(
        len(_filter['adversarials'])) if _filter['adversarials'][i] is not None]
    successful_adversarial_indices = np.array(
        successful_adversarial_indices, dtype=np.int)

    if remove_failed:
        _filter.filter(successful_adversarial_indices,
                       'successful_adversarial')

        # Convert to Numpy array after the failed samples have been removed
        _filter['adversarials'] = np.array(_filter['adversarials'])

    return _filter['adversarials'], _filter['images'], _filter['image_labels']
