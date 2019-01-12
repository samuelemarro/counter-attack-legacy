import collections
import configparser
import csv
import datetime
import gzip
import itertools
import json
import logging
import pathlib
import pickle
import urllib.request
from typing import Tuple

import numpy as np
import sklearn.metrics as metrics


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def save_zip(object, path, protocol=0):
    """
    Saves a compressed object to disk
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    file = gzip.GzipFile(path, 'wb')
    pickled = pickle.dumps(object, protocol)
    file.write(pickled)
    file.close()


def load_zip(path):
    """
    Loads a compressed object from disk
    """
    file = gzip.GzipFile(path, 'rb')
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object


def roc_curve(positive_values, negative_values):
    ground_truths = np.concatenate(
        [np.zeros_like(negative_values), np.ones_like(positive_values)], 0)
    predictions = np.concatenate([negative_values, positive_values], 0)

    # Since sklearn.metrics.roc_curve cannot handle infinity, we
    # use the maximum float value
    max_value = np.finfo(np.float).max
    predictions = [np.sign(x) * max_value if np.isinf(x)
                   else x for x in predictions]
    predictions = np.array(predictions)

    # Create the ROC curve
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        ground_truths, predictions, pos_label=1)

    # Replace max_value with np.Infinity
    thresholds = [np.sign(x) * np.Infinity if np.abs(x) ==
                  max_value else x for x in thresholds]
    thresholds = np.array(thresholds)

    return false_positive_rate, true_positive_rate, thresholds


def get_best_threshold(true_positive_rates, false_positive_rates, thresholds):
    """
    Computes the best threshold, corresponding to the threshold with the maximum value of Youden's Index
    """

    # Standard way to compute Youden's Index: sensitivity + specificity - 1
    # Since sensitivity = tpr and specificity = 1 - fpr, we use:
    # youden = tpr + (1 - fpr) - 1 = tpr - fpr

    youden_indices = true_positive_rates - false_positive_rates

    # Find the index of the best youden index
    best_threshold_index = np.argmax(youden_indices, 0)
    return thresholds[best_threshold_index].item(), true_positive_rates[best_threshold_index].item(), false_positive_rates[best_threshold_index].item()


def accuracy_distortion_curve(base_accuracy, base_attack_success_rate, distances):
    ascending_distances = np.sort(distances)
    x = [0]
    y = [base_accuracy]

    for i, distance in enumerate(ascending_distances):

        # If we're considering the nth distance, it means that n attacks are successful,
        # so the success rate for the nth distance is n / number of distances. Since the
        # distances are only computed for the successful attacks, we have to multiply by
        # the attack base success rate

        attack_success_rate = base_attack_success_rate * \
            (i + 1) / len(distances)

        accuracy = base_accuracy - attack_success_rate

        # If distance is already in x, it means that two or more distances are equal.
        # In that case, update the accuracy for the corrensponding distance

        if distance in x:
            assert x[-1] == distance
            y[-1] = accuracy
        else:
            x.append(distance)
            y.append(accuracy)

    x = np.array(x)
    y = np.array(y)

    return x, y


def lp_distance(x, y, p, batch, broadcast=False):
    """
    Computes the L_p distance between two points. Works with matrices
    of rank above 2 and supports batches, unlike numpy.linanlg.norm
    """
    if p < 0:
        raise ValueError('p must be positive or zero')

    if len(x) != len(y) and batch and not broadcast:
        raise ValueError(
            '\'x\' and \'y\' must have the same length if batch=True and broadcast=False.')

    # If x or y are empty, we retun an empty array
    empty_x = hasattr(x, '__len__') and len(x) == 0
    empty_y = hasattr(y, '__len__') and len(y) == 0
    if batch and (empty_x or empty_y):
        return np.array([], dtype=np.float)

    if broadcast:
        x, y = np.broadcast_arrays(x, y)

    def single_image(diff):
        # L_infinity: Maximum difference
        if np.isinf(p):
            return np.max(np.abs(diff))
        # L_0: Count of different values
        elif p == 0:
            return len(np.nonzero(np.reshape(diff, -1))[0])
        # L_p: p-root of the sum of diff^p
        else:
            return np.power(np.sum(np.power(np.abs(diff), p)), 1 / p)

    if batch:
        return np.array([single_image(_x - _y) for _x, _y in zip(x, y)])
    else:
        return single_image(x - y)


def filter_lists(condition, *lists):
    final_lists = []
    is_numpy = [isinstance(_list, np.ndarray) for _list in lists]

    for _list in lists:
        final_lists.append([])

    for _tuple in zip(*lists):
        if(condition(*_tuple)):
            for i, value in enumerate(_tuple):
                final_lists[i].append(value)

    for i in range(len(lists)):
        if is_numpy[i]:
            final_lists[i] = np.array(i)

    return final_lists


def is_top_k(predictions, label, k=1):
    sorted_args = np.argsort(predictions)
    top_k = sorted_args[-k:][::-1]

    return label in top_k


def top_k_count(batch_predictions, labels, k=1):
    correct_samples = [is_top_k(predictions, label, k)
                       for predictions, label in zip(batch_predictions, labels)]
    return len(np.nonzero(correct_samples)[0])


def distance_statistics(distances: np.ndarray, failure_count: int) -> Tuple[float, float, float, float]:
    """Computes the distance statistics, treating failures as Infinity.

    Parameters
    ----------
    distances : numpy.ndarray
        A 1D array with the computed distances (without the failed ones).
    failure_count : [type]
        The number of failed distance computations.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple composed of:
            The average distance
            The median distance
            The success rate
            The adjusted median distance, which treats failures as Infinity.
    """

    average_distance = np.average(distances)
    median_distance = np.median(distances)
    adjusted_median_distance = np.median(
        distances + [np.Infinity] * failure_count)
    success_rate = len(distances) / (len(distances) + failure_count)
    return average_distance, median_distance, success_rate, adjusted_median_distance


def save_results(path, table=None, command=None, info=None, header=None, delimiter='\t', transpose=False):
    # Add the command used to the info
    if command is not None:
        if info is None:
            info = [['Command:'] + [command]]
        else:
            info = [['Command:'] + [command]] + info

    # Add an empty row
    info += []

    # Transpose the table
    if transpose and table is not None:
        table = itertools.zip_longest(*table, fillvalue='')

    save_csv(path, table=table, info=info, header=header, delimiter=delimiter)


def save_csv(path, table=None, info=None, header=None, delimiter='\t'):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as file:
        wr = csv.writer(file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)

        if info is not None:
            for row in info:
                wr.writerow(row)

        if header is not None:
            wr.writerow(header)

        if table is not None:
            for row in table:
                if not isinstance(row, collections.Iterable):
                    row = [row]
                wr.writerow(row)


def download(url, path):
    r = urllib.request.urlopen(url)
    content = r.read()
    open(path, 'wb').write(content)


def download_from_config(config_path, download_path, link_section, link_name):
    try:
        config = configparser.ConfigParser()
        config.read(config_path)

        download_link = config.get(link_section, link_name)
    except KeyError:
        raise IOError(
            'The configuration file does not contain the link for \'{}\''.format(link_name))
    except IOError:
        raise IOError('Could not read the configuration file.')
    try:
        logger.info('Downloading from {}'.format(download_link))
        download(download_link, download_path)
    except IOError as e:
        raise IOError(
            'Could not download from \'{}\': {}. '
            'Please check that your internet connection is working, '
            'or download the model manually and store it in {}.'.format(download_link, e, download_path)) from e


def load_json(path):
    with open('data.json') as f:
        data = json.load(f)
    return data


class Filter(dict):
    def __init__(self, o=None, custom_filters={}, **kwArgs):
        if o is None:
            o = {}
        super().__init__(o, **kwArgs)
        self.custom_filters = {}
        self.filter_stats = collections.OrderedDict({})

    def __setitem__(self, key, value):
        value_length = len(value)
        for existing_key, existing_value in self.items():
            if value_length != len(existing_value):
                raise ValueError('The inserted value must have the same length (along the first dimension)'
                                 'of the other values (inserted value of length {}, mismatched with \"{}\": {})'.format(value_length, existing_key, len(existing_value)))
        super().__setitem__(key, value)

    def empty(self):
        for key in self.keys():
            super().__setitem__(key, [])

    def filter(self, indices, name=None):
        if name in self.filter_stats:
            raise ValueError('\'name\' must be unique')

        if name == None:
            name = 'filter_{}'.format(len(self.filter_stats))

        if len(indices) == 0:
            self.empty()
            return

        for key in self.keys():
            if key in self.custom_filters:
                super().__setitem__(
                    key, self.custom_filters[key](self[key], indices))
            elif isinstance(self[key], np.ndarray):
                super().__setitem__(key, self[key][indices])
            elif isinstance(self[key], list):
                super().__setitem__(key, [self[key][i] for i in indices])
            else:
                raise NotImplementedError('The filter for collection "{}" of type {} is not implemented.'
                                          'Please define a custom filter by setting custom_filters[\'{}\']'.format(key, type(key), key))

        self.filter_stats[name] = len(indices)
