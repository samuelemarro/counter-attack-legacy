import numpy as np
import foolbox
import OffenseDefense.utils as utils

class RandomDirectionAttack(foolbox.attacks.Attack):
    def __init__(self, model, criterion, p, directions, search_steps, search_epsilon, finetuning_precision):
        super().__init__(model, criterion, distance=foolbox.distances.MSE, threshold=None)
        self.p = p
        self.directions = directions
        self.search_steps = search_steps
        self.search_epsilon = search_epsilon
        self.finetuning_precision = finetuning_precision

    #Uses Marsaglia's method for picking a random point on a unit sphere
    #Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math. Stat. 43, 645-646, 1972.
    def _random_directions(self, dimensions):
        directions = np.random.randn(dimensions).astype(np.float)
        directions /= np.linalg.norm(directions, axis=0)
        return directions.astype(np.float32)

    #Avoids Out of Memory errors by feeding n samples each time
    def _safe_batch_predictions(self, foolbox_adversarial, images, n=1):
        batch_predictions = []
        batch_is_adversarial = []
        split_images = [np.array(images[i * n:(i + 1) * n]) for i in range((len(images) + n - 1) // n )] 
        for subsection in split_images:
            subsection_predictions, subsection_is_adversarial = foolbox_adversarial.batch_predictions(subsection.astype(np.float32))

            batch_predictions += list(subsection_predictions)
            batch_is_adversarial += list(subsection_is_adversarial)

        return np.array(batch_predictions), np.array(batch_is_adversarial)

    def _get_successful_adversarials(self, foolbox_adversarial, vectors, magnitude):
        bound_min, bound_max = foolbox_adversarial.bounds()
        image = foolbox_adversarial.original_image
        potential_adversarials = np.clip(vectors * magnitude + image, bound_min, bound_max).astype(np.float32)
        adversarial_predictions, is_adversarials = self._safe_batch_predictions(foolbox_adversarial, potential_adversarials, n=50)

        successful_adversarial_indices = np.nonzero(is_adversarials)[0]

        return potential_adversarials[successful_adversarial_indices], vectors[successful_adversarial_indices]

    def _get_closest_sample(self, foolbox_adversarial):
        logger = logging.getLogger(__name__)

        image = foolbox_adversarial.original_image

        vectors = [self._random_directions(image.size) for i in range(self.directions)]
        vectors = [np.reshape(vector, list(image.shape)) for vector in vectors]
        vectors = np.array(vectors)

        #First, find the closest samples that are adversarials
        search_magnitude = self.search_epsilon
        best_adversarial = None

        for _ in range(self.search_steps):
            successful_adversarials, successful_vectors = self._get_successful_adversarials(foolbox_adversarial, vectors, search_magnitude)

            #The first samples to cross the boundary are the potential candidates, so we drop the rest
            if len(successful_adversarials) > 0:
                logger.info('Found {} successful adversarials with search magnitude {}'.format(len(successful_adversarials), search_magnitude))
                vectors = successful_vectors

                best_adversarial_index = np.argmin(utils.lp_distance(successful_adversarials, image, self.p, True))
                best_adversarial = successful_adversarials[best_adversarial_index]
                break

            search_magnitude += self.search_epsilon
        
        if len(successful_adversarials) == 0:
            logger.warning('Couldn\'t find an adversarial sample')
            return None

        #Finetuning: Use binary search to find the distance with high precision
        #If no sample is adversarial, we have to increase the finetuning magnitude. If at least one sample is adversarial, drop the others
        min_ = search_magnitude - self.search_epsilon
        max_ = search_magnitude

        while (max_ - min_) > self.finetuning_precision:
            finetuning_magnitude = (max_ + min_) / 2

            successful_adversarials, successful_vectors = self._get_successful_adversarials(foolbox_adversarial, vectors, finetuning_magnitude)

            if len(successful_adversarials) == 0:
                min_ = finetuning_magnitude
            else:
                max_ = finetuning_magnitude

                best_adversarial_index = np.argmin(utils.lp_distance(successful_adversarials, image, self.p, True))
                best_adversarial = successful_adversarials[best_adversarial_index]
                vectors = successful_vectors

        assert best_adversarial is not None

        logger.info('Finetuned from magnitude {} to {}'.format(search_magnitude, finetuning_magnitude))

        return best_adversarial
    
    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label = None, unpack = True, **kwargs):
        best_adversarial = self._get_closest_sample(input_or_adv)

        #This is temporary, it's just to make sure that the best adversarial according to L_p
        #is actually the one we found -- 15/11/2018
        input_or_adv._reset()
        input_or_adv.predictions(best_adversarial)
