import gzip
import pickle
import collections
import csv
import numpy as np
import sklearn.metrics as metrics


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


def save_zip(object, filename, protocol=0):
    """
    Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    pickled = pickle.dumps(object, protocol)
    file.write(pickled)
    file.close()


def load_zip(filename):
    """
    Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
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
    # Create the ROC curve
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        ground_truths, predictions, pos_label=1)

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


def lp_distance(x, y, p, batch, broadcast=True):
    """
    Computes the L_p distance between two points. Works with matrices
    of rank above 2 and supports batches, unlike numpy.linanlg.norm
    """
    if p < 0:
        raise ValueError('p must be positive or zero')

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


def is_top_k(predictions, label, k=1):
    sorted_args = np.argsort(predictions)
    top_k = sorted_args[-k:][::-1]

    return label in top_k


def top_k_count(batch_predictions, labels, k=1):
    correct_samples = [is_top_k(predictions, label, k)
                       for predictions, label in zip(batch_predictions, labels)]
    return len(np.nonzero(correct_samples)[0])


def save_csv_table(path, table, header=None):
    with open(path, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)

        if header is not None:
            wr.writerow(header)

        for row in table:
            wr.writerow(row)


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

    def filter(self, indices, name=None):
        if name in self.filter_stats:
            raise ValueError('\'name\' must be unique')

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
