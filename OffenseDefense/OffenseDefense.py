import queue
import numpy as np
import torch
import torch.nn as nn
import foolbox
import matplotlib.pyplot as plt
import utils
import tests
import visualisation
import model_tools
from pytorch_classification.models.cifar.densenet import DenseNet

#NOTA: La precisione del floating point si traduce in un errore medio dello 0,01% (con punte dello 0,5%)
#Questo errore può essere diminuito passando al double, ma è un suicidio computazionale perché raddoppia la
#memoria richiesta e PyTorch è ottimizzato per float

#NOTA 2: A volte questa differenza di precisione è sufficiente a rendere un'immagine sul boundary
#talvolta genuina e talvolta avversariale. Ciò non è un problema per l'anti-attack (dopotutto non gli importa
#la vera label), ma lo è per l'attack.

class FineTuningAttack(foolbox.attacks.Attack):
    def __init__(self, attack, p, epsilon=1e-6, max_steps=100):
        super().__init__(attack._default_model,
                         attack._default_criterion,
                         attack._default_distance,
                         attack._default_threshold)
        self.attack = attack
        self.p = p
        self.epsilon = epsilon
        self.max_steps = max_steps

    @foolbox.attacks.base.call_decorator
    def __call__(self, input_or_adv, label = None, unpack = True, **kwargs):
        foolbox_adversarial = self.attack(input_or_adv, label, False, **kwargs)
        min_, max_ = foolbox_adversarial.bounds()
        model = foolbox_adversarial._model
        
        criterion = foolbox_adversarial._criterion
        original_image = foolbox_adversarial.original_image
        adversarial_image = foolbox_adversarial.image
        
        if adversarial_image is None:
            return

        adversarial_label = foolbox_adversarial.adversarial_class

        previous_image = np.copy(adversarial_image)
        previous_distance = utils.lp_distance(original_image, adversarial_image, self.p)

        i = 0
        while i < self.max_steps:
            predictions, gradient = model.predictions_and_gradient(adversarial_image, adversarial_label)

            #Normalize the gradient
            gradient_norm = np.sqrt(np.mean(np.square(gradient)))
            gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)

            perturbed = adversarial_image - gradient * self.epsilon
            perturbed = np.clip(perturbed, min_, max_)

            perturbed_distance = utils.lp_distance(original_image, perturbed, self.p)

            if criterion.is_adversarial(predictions, label) and perturbed_distance < previous_distance:
                previous_image = perturbed
                previous_distance = perturbed_distance
            else:
                break

            i += 1

        #previous_image = None
        #input_or_adv._Adversarial_best_adversarial = previous_image
        input_or_adv.predictions(previous_image)
        #print(np.all(np.equal(previous_image, input_or_adv.image)))

class BoundaryDistanceCalculator:
    def __init__(self, bound_min, bound_max, directions, search_steps, search_epsilon, finetuning_precision):
        self.bound_min = bound_min
        self.bound_max = bound_max
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
    def _safe_batch_predictions(self, model, images, n=1):
        outputs = []
        split_images = [np.array(images[i * n:(i + 1) * n]) for i in range((len(images) + n - 1) // n )] 
        for subsection in split_images:
            outputs += list(model.batch_predictions(subsection.astype(np.float32)))
        return np.array(outputs)

    def _get_successful_adversarials(self, model, criterion, image, label, vectors, magnitude):
        potential_adversarials = np.clip(vectors * magnitude + image, self.bound_min, self.bound_max).astype(np.float32)
        adversarial_predictions = self._safe_batch_predictions(model, potential_adversarials, n=50)
        successful_adversarial_indices = np.array([i for i in range(len(potential_adversarials)) if criterion.is_adversarial(adversarial_predictions[i], label)]).astype(int)
        return potential_adversarials[successful_adversarial_indices], vectors[successful_adversarial_indices]

    def get_closest_sample(self, image, label, model, criterion, p):
        vectors = [self._random_directions(image.size) for i in range(self.directions)]
        vectors = [np.reshape(vector, list(image.shape)) for vector in vectors]
        vectors = np.array(vectors)

        #First, find the closest samples that are adversarials
        search_magnitude = self.search_epsilon
        best_adversarial = None

        for _ in range(self.search_steps):
            successful_adversarials, successful_vectors = self._get_successful_adversarials(model, criterion, image, label, vectors, search_magnitude)

            #The first samples to cross the boundary are the potential candidates, so we drop the rest
            if len(successful_adversarials) > 0:
                print('Found {} successful adversarials with search magnitude {}'.format(len(successful_adversarials), search_magnitude))
                vectors = successful_vectors

                best_adversarial_index = np.argmin(utils.lp_distance(successful_adversarials, image, p, True))
                best_adversarial = successful_adversarials[best_adversarial_index]
                break

            search_magnitude += self.search_epsilon
        
        if len(successful_adversarials) == 0:
            print('Couldn\'t find an adversarial sample')
            return None

        #Finetuning: Use binary search to find the distance with high precision
        #If no sample is adversarial, we have to increase the finetuning magnitude. If at least one sample is adversarial, drop the others
        min_ = search_magnitude - self.search_epsilon
        max_ = search_magnitude

        while (max_ - min_) > self.finetuning_precision:
            finetuning_magnitude = (max_ + min_) / 2

            successful_adversarials, successful_vectors = self._get_successful_adversarials(model, criterion, image, label, vectors, finetuning_magnitude)

            if len(successful_adversarials) == 0:
                min_ = finetuning_magnitude
            else:
                max_ = finetuning_magnitude

                best_adversarial_index = np.argmin(utils.lp_distance(successful_adversarials, image, p, True))
                best_adversarial = successful_adversarials[best_adversarial_index]
                vectors = successful_vectors

        assert best_adversarial is not None

        print('Finetuned from magnitude {} to {}'.format(search_magnitude, finetuning_magnitude))

        return best_adversarial



def prepare_model():
    model = DenseNet(100, num_classes=10)
    model = nn.DataParallel(model)
    model_tools.load_model(model, './pretrained_models/cifar10/densenet-bc-100-12/model_best.pth.tar')
    model = model.module
    model = nn.Sequential(model_tools.Preprocessing(means=(0.4914, 0.4822, 0.4465),
                                                    stds=(0.2023, 0.1994, 0.2010)), model)
    model.eval()

    return model




"""
Requires that the difference between the top class and the following one (regardless of
whether the image is adversarial) is above a threshold. 
Important: This criterion doesn't check if the image is adversarial, you're supposed to
combine it with another criterion using foolbox.criteria.CombinedCriteria.

Useful to prevent bugs due to approximation errors (e.g. when a 1e-7 difference
between the 1st and 2nd prediction causes the top class to be sometimes different).
Usually a 1e-5 threshold is sufficient. If you're not sure, assert that the top labels
of the genuine images are different from the top labels of the adversarial images.

Warning: Using this criterion makes the attack slightly overshoot. This might not be
a problem for adversarial attacks themselves, but it might be if you're estimating the
distance to the decision boundary. We therefore recommend using it only when it's
truly necessary.
"""
class TargetTop2Difference(foolbox.criteria.Criterion):
    def __init__(self, threshold):
        self.threshold = threshold
    def is_adversarial(self, predictions, label):
        difference = utils.top_k_difference(predictions, 2)
        if difference < self.threshold:
            print(difference)

        return difference > self.threshold
        

    


def image_test(model, loader, adversarial_attack, anti_attack):
    model.eval()
    foolbox_model = foolbox.models.PyTorchModel(model, (0, 1), 10, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0, 1))
    
    plt.ion()
    for i, data in enumerate(loader):
        images, labels = data
        images = images.numpy()
        labels = labels.numpy()

        if i == 0:
            f, subfigures = plt.subplots(images.shape[0], 4, sharex='all', sharey='all', gridspec_kw={'wspace': 0, 'hspace' : 0})
            #f.set_size_inches(18.5, 10.5, forward=True)

        images, labels, adversarials, anti_genuines, anti_adversarials, _ = get_samples(foolbox_model, images, labels, adversarial_attack, anti_attack)

        #f, subfigures = plt.subplots(images.shape[0], 4, sharex='all', sharey='all', gridspec_kw={'wspace': 0, 'hspace' : 0})

        images = np.moveaxis(images, 1, -1)
        adversarials = np.moveaxis(adversarials, 1, -1)
        anti_adversarials = np.moveaxis(anti_adversarials, 1, -1)
        anti_genuines = np.moveaxis(anti_genuines, 1, -1)

        f.clf()


        for j in range(images.shape[0]):
            image = images[j]
            adversarial = adversarials[j]
            anti_adversarial = anti_adversarials[j]
            anti_genuine = anti_genuines[j]

            subfigures[j, 0].imshow(image)
            subfigures[j, 0].axis('off')

            subfigures[j, 1].imshow(adversarial)
            subfigures[j, 1].axis('off')

            subfigures[j, 2].imshow(anti_adversarial)
            subfigures[j, 2].axis('off')

            subfigures[j, 3].imshow(anti_genuine)
            subfigures[j, 3].axis('off')

        plt.draw()
        plt.pause(0.001)
        

def batch_main():
    trainloader = model_tools.cifar10_trainloader(1, 10, flip=False, crop=False, normalize=False, shuffle=True)
    testloader = model_tools.cifar10_testloader(1, 10, normalize=False, shuffle=True)

    model = prepare_model()
    model.eval()

    p = np.Infinity

    adversarial_criterion = foolbox.criteria.Misclassification()
    adversarial_criterion = foolbox.criteria.CombinedCriteria(adversarial_criterion, TargetTop2Difference(1e-5))
    if p == 2:
        adversarial_attack = foolbox.attacks.DeepFoolLinfinityAttack(criterion=adversarial_criterion)
    elif p == np.Infinity:
        adversarial_attack = foolbox.attacks.DeepFoolL2Attack(criterion=adversarial_criterion)
    #adversarial_attack = RandomDirectionAttack(100, 100, 1e-2, 1e-5, criterion=adversarial_criterion)
    #adversarial_attack = FineTuningAttack(adversarial_attack, p)

    anti_adversarial_criterion = foolbox.criteria.Misclassification()
    #anti_adversarial_criterion = foolbox.criteria.CombinedCriteria(anti_adversarial_criterion, TargetTop2Difference(1e-5))
    if p == 2:
        adversarial_anti_attack = foolbox.attacks.DeepFoolL2Attack(criterion=anti_adversarial_criterion)
    elif p == np.Infinity:
        adversarial_anti_attack = foolbox.attacks.DeepFoolLinfinityAttack(criterion=anti_adversarial_criterion)
    #adversarial_anti_attack = FineTuningAttack(adversarial_attack, p)

    #basic_test(model, testloader, adversarial_attack, adversarial_anti_attack, p)
    #tests.image_test(model, testloader, adversarial_attack, adversarial_anti_attack)
    #direction_attack = RandomDirectionAttack(100, 100, 1e-2, 1e-5)

    distance_calculator = BoundaryDistanceCalculator(0, 1, 1000, 100, 0.05, 1e-7)

    tests.approximation_test(model, testloader, adversarial_anti_attack, distance_calculator, p, adversarial_attack=adversarial_attack)



cifar_names = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
    ]

if __name__ == '__main__':
    #main()
    batch_main()