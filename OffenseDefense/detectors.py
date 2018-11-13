import foolbox
import numpy as np
import traceback

class Detector:
    def __call__(self, image, predictions):
        pass

class AntiAdversarialDistance:
    def __init__(self, attack, minimum_distance):
        self.attack = attack
        self.minimum_distance = minimum_distance
    def __call__(self, image, predictions):
        try:
            anti_image = self.attack(image, np.argmax(predictions))
            if anti_image is None:
                print('Cannot create Anti-Image in detector')
                return False
            anti_distance = np.average(np.power((anti_image - image), 2))
            print(anti_distance)
            return anti_distance >= self.minimum_distance
        except Exception:
            print('Detection Error')
            traceback.print_exc()
            return False

class DetectionAwareAdversarial(foolbox.Adversarial):
    def __init__(self, model, criterion, original_image, original_class, detector : Detector, distance = foolbox.distances.MSE, threshold = None, verbose = False):
        super(DetectionAwareAdversarial, self).__init__(model, criterion, original_image, original_class, distance, threshold, verbose)
        self.detector = detector

    #If you're wondering what's going on: Adversarial implements a method called __is_adversarial.
    #Python replaces the double underscore with _(Class Name)__, so __is_adversarial becomes
    ##_Adversarial__is_adversarial

    def _Adversarial__is_adversarial(self, image, predictions, in_bounds):
        """Interface to criterion.is_adversarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.

        """
        is_adversarial = super()._criterion.is_adversarial(
            predictions, super().original_class)


        if is_adversarial and self.detector is not None:
            is_adversarial = self.detector(image, predictions)

        assert isinstance(is_adversarial, bool) or \
            isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self._Adversarial__new_adversarial(
                image, predictions, in_bounds)
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance
