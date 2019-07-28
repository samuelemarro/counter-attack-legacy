import logging
import numpy as np
import foolbox
from . import utils


class AttackWithParameters(foolbox.attacks.Attack):
    """Many foolbox attacks can accept keyword arguments when called. This
    attack allows us to set the arguments that will be passed before calling.
    """

    def __init__(self, attack: foolbox.attacks.Attack, **call_kwargs):
        """Initializes the AttackWithParameters.

        Parameters
        ----------
        attack : foolbox.attacks.Attack
            The attack that will be called.
        **call_kwargs : 
            The keyword arguments that will be passed to the attack.
        """

        super().__init__(attack._default_model, attack._default_criterion,
                         attack._default_distance, attack._default_threshold)
        self.attack = attack
        self.call_kwargs = call_kwargs

    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        self.attack(input_or_adv, label, unpack, **self.call_kwargs)

    def name(self):
        return self.attack.name() + ' (with parameters)'


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
                 p : float,
                 directions: int,
                 search_steps: int,
                 search_epsilon: float,
                 finetuning_precision: float,
                 max_batch_size: int = 50,
                 random_state: np.random.RandomState = None):
        """Initializes the attack.

        Parameters
        ----------
        model : foolbox.models.Model
            The model that will be attacked. Can be overriden during the call by passing
            a foolbox.Adversarial with a different model.
        criterion : foolbox.criteria.Criterion
            The criterion that will be used. Can be overriden during the call by passing
            a foolbox.Adversarial with a different model.
        p : float
            The Lp distance that will be used to compute the distance.
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
        max_batch_size : int, optional
            The maximum size of the batch that will be fed to the model. A bigger batch size
            means faster search but more memory usage.
        random_state : numpy.random.RandomState, optional
            The RandomState that will be used to generate the random directions. If None, the
            attack will use NumPy's default random module.
        """

        foolbox_distance = utils.p_to_foolbox(p)
        super().__init__(model, criterion, distance=foolbox_distance, threshold=None) # No threshold support

        self.directions = directions
        self.search_steps = search_steps
        self.search_epsilon = search_epsilon
        self.finetuning_precision = finetuning_precision
        self.max_batch_size = max_batch_size
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
    def _safe_batch_predictions(self, foolbox_adversarial, images):
        batch_predictions = []
        batch_is_adversarial = []
        split_images = []
        minibatch_count = (
            len(images) + self.max_batch_size - 1) // self.max_batch_size

        for i in range(minibatch_count):
            split_images.append(
                np.array(images[i * self.max_batch_size:(i + 1) * self.max_batch_size]))

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
            foolbox_adversarial, potential_adversarials)

        return potential_adversarials[batch_is_adversarial], vectors[batch_is_adversarial]

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
                logger.debug('Found {} successful adversarials with search magnitude {}'.format(
                    len(successful_adversarials), search_magnitude))
                vectors = successful_vectors
                break

            search_magnitude += self.search_epsilon

        if len(successful_adversarials) == 0:
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

        logger.debug('Finetuned from magnitude {} to {}'.format(
            search_magnitude, finetuning_magnitude))

    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        self._find_closest_sample(input_or_adv)


class ImageBlendingAttack(foolbox.attacks.Attack):
    """
    Uses binary search to find the closest adversarial in a straight line
    between the image and another adversarial
    """
    def __init__(self, model, criterion, distance, threshold, initialization_attack = foolbox.attacks.SaltAndPepperNoiseAttack, precision=1e-5):
        super().__init__(model, criterion, distance, threshold)
        self.initialization_attack = initialization_attack(model, criterion, distance, threshold)
        self.precision = precision

    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        
        original_image = input_or_adv.original_image
        other_image = self.initialization_attack(input_or_adv)

        if other_image is None:
            # The adversarial attack failed, aborting
            return

        adversarial = input_or_adv
        del input_or_adv

        # Save the other image
        adversarial.predictions(other_image)

        min_blending = 0
        max_blending = 1

        while (max_blending - min_blending) > self.precision:
            blending = (min_blending + max_blending) / 2

            image = original_image * blending + other_image * (1 - blending)

            _, is_adversarial = adversarial.predictions(image)

            if is_adversarial:
                # We overshot, reduce blending
                max_blending = blending
            else:
                # We undershot, increase blending
                min_blending = blending




