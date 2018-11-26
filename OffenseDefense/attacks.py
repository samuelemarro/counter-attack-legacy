import logging
import numpy as np
import foolbox
from . import distance_tools, utils


class RandomDirectionAttack(foolbox.attacks.Attack):
    """A gradient-free attack that tries to push the sample along several directions
    and finetunes the magnitude of the first perturbation that crosses the decision
    boundary.
    Note: The finetuning process is not guaranteed to improve the estimate because it
    finetunes the magnitudes, not the L_p distances. Nevertheless, foolbox automatically
    tracks the best adversarial sample found, regardless of whether it was found thanks
    to finetuning.
    """

    def __init__(self,
                 model: foolbox.models.Model,
                 criterion: foolbox.criteria.Criterion,
                 p: np.float,
                 directions: int,
                 search_steps: int,
                 search_epsilon: float,
                 finetuning_precision: float,
                 random_state: np.random.RandomState = None):
        """Initializes the attack.

        Parameters
        ----------
        model : foolbox.models.Model
            The model that will be attacked. Can be overriden during the call by passing
            a foolbox.Adversarial with a different model.
        criterion : foolbox.criteria.Criterion
            The criterion that wiil be used. Can be overriden during the call by passing
            a foolbox.Adversarial with a different model.
        p : numpy.float
            The p of the L_p norm that will be used to compare samples.
        directions : int
            The number of directions that will be explored. More directions means
            better results but slower execution.
        search_steps : int
            The number of steps that will be computed along a certain direction. More 
            search steps means better results but slower execution.
        search_epsilon : float
            The size of each step along a certain direction. Remember that the attack will
            finetune the magnitude, so don't be afraid to use high values (as long as they
            are small enough that a single step doesn't leap over an entire classification
            region).
        finetuning_precision : float
            The precision to which the magnitude will be finetuned. Since the attack finetunes
            using binary search, a small value doesn't have a big effect on execution time.
        random_state : numpy.random.RandomState, optional
            The RandomState that will be used to generate the random directions. If None, the
            attack will use NumPy's default random module.
        """

        super().__init__(model, criterion, distance=distance_tools.LpDistance(p), threshold=None)
        self.p = p
        self.directions = directions
        self.search_steps = search_steps
        self.search_epsilon = search_epsilon
        self.finetuning_precision = finetuning_precision
        self.random_state = random_state

    # Uses Marsaglia's method for picking a random point on a unit sphere
    # Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math. Stat. 43, 645-646, 1972.
    def _random_directions(self, dimensions):
        if self.random_state is None:
            directions = np.random.randn(dimensions).astype(np.float)
        else:
            directions = self.random_state.randn(dimensions).astype(np.float)

        directions /= np.linalg.norm(directions, axis=0)
        return directions.astype(np.float32)

    # Avoids Out of Memory errors by splitting the batch in minibatches
    def _safe_batch_predictions(self, foolbox_adversarial, images, n=1):
        batch_predictions = []
        batch_is_adversarial = []
        split_images = [np.array(images[i * n:(i + 1) * n])
                        for i in range((len(images) + n - 1) // n)]
        for subsection in split_images:
            subsection_predictions, subsection_is_adversarial = foolbox_adversarial.batch_predictions(
                subsection.astype(np.float32))

            batch_predictions += list(subsection_predictions)
            batch_is_adversarial += list(subsection_is_adversarial)

        return np.array(batch_predictions), np.array(batch_is_adversarial)

    def _get_successful_adversarials(self, foolbox_adversarial, vectors, magnitude):
        bound_min, bound_max = foolbox_adversarial.bounds()
        image = foolbox_adversarial.original_image
        potential_adversarials = np.clip(
            vectors * magnitude + image, bound_min, bound_max).astype(np.float32)
        _, batch_is_adversarial = self._safe_batch_predictions(
            foolbox_adversarial, potential_adversarials, n=50)

        successful_adversarial_indices = np.nonzero(batch_is_adversarial)[0]

        return potential_adversarials[successful_adversarial_indices], vectors[successful_adversarial_indices]

    def _find_closest_sample(self, foolbox_adversarial):
        logger = logging.getLogger(__name__)

        image = foolbox_adversarial.original_image

        vectors = [self._random_directions(image.size)
                   for i in range(self.directions)]
        vectors = [np.reshape(vector, list(image.shape)) for vector in vectors]
        vectors = np.array(vectors)

        # First, find the closest samples that are adversarials
        search_magnitude = self.search_epsilon

        for _ in range(self.search_steps):
            successful_adversarials, successful_vectors = self._get_successful_adversarials(
                foolbox_adversarial, vectors, search_magnitude)

            # The first samples to cross the boundary are the potential candidates, so we drop the rest
            if len(successful_adversarials) > 0:
                logger.info('Found {} successful adversarials with search magnitude {}'.format(
                    len(successful_adversarials), search_magnitude))
                vectors = successful_vectors
                break

            search_magnitude += self.search_epsilon

        if len(successful_adversarials) == 0:
            logger.warning('Couldn\'t find an adversarial sample')
            return

        # Finetuning: Use binary search to find the distance with high precision
        # If no sample is adversarial, we have to increase the finetuning magnitude. If at least one sample is adversarial, drop the others
        min_ = search_magnitude - self.search_epsilon
        max_ = search_magnitude

        while (max_ - min_) > self.finetuning_precision:
            finetuning_magnitude = (max_ + min_) / 2

            successful_adversarials, successful_vectors = self._get_successful_adversarials(
                foolbox_adversarial, vectors, finetuning_magnitude)

            if len(successful_adversarials) == 0:
                min_ = finetuning_magnitude
            else:
                max_ = finetuning_magnitude

        logger.info('Finetuned from magnitude {} to {}'.format(
            search_magnitude, finetuning_magnitude))

    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        self._find_closest_sample(input_or_adv)
