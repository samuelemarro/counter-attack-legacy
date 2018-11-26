import foolbox
import numpy as np


class Rejector:
    def valid(self, image, predictions):
        raise NotImplementedError()

    def batch_valid(self, images, batch_predictions):
        raise NotImplementedError()


class DistanceRejector(Rejector):
    def __init__(self, distance_tool, threshold, valid_if_failed):
        self.distance_tool = distance_tool
        self.threshold = threshold
        self.valid_if_failed = valid_if_failed

    def valid(self, image, predictions):
        distance = self.distance_tool.get_distance(image)

        if distance is None:
            return self.valid_if_failed

        return distance >= self.threshold

    def batch_valid(self, images, batch_predictions):
        distances = self.distance_tool.get_distances(images)

        responses = []

        for distance in distances:
            if distance is None:
                responses.append(self.valid_if_failed)
            else:
                responses.append(distance >= self.threshold)

        return responses


class DummyRejector(Rejector):
    def valid(self, image, predictions):
        return True

    def batch_valid(self, images, batch_predictions):
        return np.ones([len(images)], dtype=np.bool)


class RejectorAdversarial(foolbox.Adversarial):
    """
    A custom adversarial that treats an evasion target as standard target.
    It only has two labels, 0 (rejected or correctly classified) and 1 (unrejected and misclassified).
    """

    def __init__(self,
                 model,
                 rejector,
                 criterion,
                 original_image,
                 original_class,
                 distance=foolbox.distances.MSE,
                 threshold=None,
                 verbose=False):
        self.rejector = rejector
        self.original_image_class = original_class
        super().__init__(model, criterion, original_image, 0, distance, threshold, verbose)

    def target_class(self):
        try:
            self._criterion.target_class()
            # If it has a target class, we set it to 1
            target_class = None
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        return 2

    def compound_predictions(self, image):
        predictions = self._model.predictions(image)
        is_adversarial = self._criterion.is_adversarial(
            predictions, self.original_image_class)

        # If the sample isn't adversarial, don't bother running the rejector
        if not is_adversarial:
            return np.array([1., 0.])

        is_valid = self.rejector.valid(image, predictions)
        if is_adversarial and is_valid:
            output = np.array([0., 1.])
        else:
            output = np.array([1., 0.])
        return output

    def compound_batch_predictions(self, images):
        # The default output is [0., 1.]
        outputs = np.tile([1., 0.], [len(images), 1])

        batch_predictions = self._model.batch_predictions(images)
        batch_is_adversarial = [self._criterion.is_adversarial(
            predictions, self.original_image_class) for predictions in batch_predictions]
        adversarial_indices = [i for i in range(
            len(batch_is_adversarial)) if batch_is_adversarial[i]]

        batch_is_valid = self.rejector.batch_valid(
            images, batch_predictions[adversarial_indices])

        for i, original_index in enumerate(adversarial_indices):
            if batch_is_valid[i]:
                outputs[original_index] = np.array([0., 1.])
        if len(np.nonzero(batch_is_valid)[0]) > 0:
            print('Found one valid sample')

        return outputs

    def has_gradient(self):
        return False

    def predictions(self, image, strict=True, return_details=False):
        """Interface to model.predictions for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = self.compound_predictions(image)
        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
            image, predictions, in_bounds)

        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def batch_predictions(
            self, images, greedy=False, strict=True, return_details=False):
        """Interface to model.batch_predictions for attacks.

        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of images with shape (batch size, height, width, channels).
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        if strict:
            in_bounds = self.in_bounds(images)
            assert in_bounds

        self._total_prediction_calls += len(images)
        predictions = self.compound_batch_predictions(images)

        assert predictions.ndim == 2
        assert predictions.shape[0] == images.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(images[i])
            is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
                images[i], predictions[i], in_bounds_i)
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == images.shape[0]

        return predictions, is_adversarial

    def gradient(self, image=None, label=None, strict=True):
        raise NotImplementedError()

    def predictions_and_gradient(
            self, image=None, label=None, strict=True, return_details=False):
        raise NotImplementedError()

    def backward(self, gradient, image=None, strict=True):
        raise NotImplementedError()
