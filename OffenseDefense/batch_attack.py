import copy
import logging
import pickle
import queue

import dill
import numpy as np
import foolbox
import torch
import torch.multiprocessing

from . import utils

logger = logging.getLogger(__name__)


class NoPreprocessing:
    def __call__(self, x):
        return x

class SerializableFunction:
    def __init__(self, serialized_function):
        self.__serialized_function = serialized_function
        self.__function = dill.loads(serialized_function)

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)

    def __reduce__(self):
        return (self.__class__, (self.__serialized_function, ))

def sanitize_foolbox_models(obj, memo=None):
    def pickable(obj):
        try:
            pickle.dumps(obj)
            return True
        except pickle.PickleError:
            return False

    if memo is None:
        memo = set()

    memo.add(id(obj))

    sanitized = False

    # Replace the preprocessing with a serializable version
    if isinstance(obj, foolbox.models.Model) and not isinstance(obj._preprocessing, SerializableFunction):
        serialized_preprocessing = dill.dumps(obj._preprocessing)
        obj._preprocessing = SerializableFunction(serialized_preprocessing)
        sanitized = True

    unserializable_fields = [x for x in vars(obj) if not pickable(x)]

    for unserializable_field in unserializable_fields:
        if id(unserializable_field) not in memo: # Avoid circular dependencies
            sanitized_field = sanitize_foolbox_models(unserializable_field)

            # If the field was sanitized, sanitized is set to True
            sanitized = sanitized or sanitized_field

    return sanitized
    



def _identity(x):
    return x

# We'll keep it for a while to simplify debugging
class MultiprocessingCompatibleTorchModel(foolbox.models.PyTorchModel):
    def __init__(self, model, num_classes, bounds=(0,1), channel_axis=1, device=None):
        super().__init__(model, bounds, num_classes, channel_axis, device, preprocessing=NoPreprocessing())
        
    def _process_input(self, x):
        return x, _identity

def attack_worker(_data):
    image, label, attack = _data

    return attack(image, label)

def run_batch_attack(foolbox_model, attack, images, labels, num_workers):
    assert len(images) == len(labels)

    #attack._default_model = MultiprocessingCompatibleTorchModel(foolbox_model._model, foolbox_model.num_classes(), foolbox_model.bounds(), foolbox_model.channel_axis(), 'cpu:0')

    sanitized = sanitize_foolbox_models(attack._default_model)
    if sanitized:
        logger.debug('The foolbox model contained unserializable preprocessings, which were sanitized.')

    data = []
    for image, label in zip(images, labels):
        data.append((image, label, attack))

    pool = torch.multiprocessing.Pool(num_workers)
    try:
        adversarials = pool.map(attack_worker, data)
    except AttributeError:
        logger.error('Parallel attacking failed. One of the possible causes is that PyTorch\'s '
            'custom Pickler failed to serialize the Foolbox model, the images, the labels or the attack. '
            'If so, a common reason is a Foolbox model with an unserializable preprocessing (not to be '
            'confused with our mean/stdev preprocessing or defensive preprocessing).')
        raise
        
    adversarials = list(adversarials)

    assert len(adversarials) == len(images)

    return adversarials


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
                     num_workers: int = 0):
    if len(images) != len(labels):
        raise ValueError('images and labels must have the same length.')

    _filter = utils.Filter()
    _filter['images'] = images
    _filter['image_labels'] = labels

    if num_workers == 0:
        _filter['adversarials'] = run_individual_attack(
            adversarial_attack, _filter['images'], _filter['image_labels'])

    else:
        _filter['adversarials'] = run_batch_attack(foolbox_model,
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
