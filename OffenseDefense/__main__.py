import queue
import logging
import numpy as np
import torch
import torch.nn as nn
import foolbox
import matplotlib.pyplot as plt
import OffenseDefense.utils as utils
import OffenseDefense.tests as tests
import OffenseDefense.model_tools as model_tools
import OffenseDefense.distance_tools as distance_tools
import OffenseDefense.loaders as loaders
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.attacks as attacks
import OffenseDefense.training as training
from OffenseDefense.pytorch_classification.models.cifar.densenet import DenseNet

#NOTA: La precisione del floating point si traduce in un errore medio dello 0,01% (con punte dello 0,5%)
#Questo errore può essere diminuito passando al double, ma è un suicidio computazionale perché raddoppia la
#memoria richiesta e PyTorch è ottimizzato per float

#NOTA 2: A volte questa differenza di precisione è sufficiente a rendere un'immagine sul boundary
#talvolta genuina e talvolta avversariale. Ciò non è un problema per l'anti-attack (dopotutto non gli importa
#la vera label), ma lo è per l'attack.

#NOTE: Using an adversarial loader means that failed samples are automatically removed


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
        

    


def image_test(foolbox_model, loader, adversarial_attack, anti_attack):
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
    train_loader = model_tools.cifar10_train_loader(1, 10, flip=False, crop=False, normalize=False, shuffle=True)
    test_loader = model_tools.cifar10_test_loader(1, 40, normalize=False, shuffle=True)
    
    model = prepare_model()
    model.eval()
    
    foolbox_model = foolbox.models.PyTorchModel(model, (0, 1), 10, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0, 1))

    p = np.Infinity

    adversarial_criterion = foolbox.criteria.Misclassification()
    adversarial_criterion = foolbox.criteria.CombinedCriteria(adversarial_criterion, TargetTop2Difference(1e-5))
    if p == 2:
        adversarial_attack = foolbox.attacks.DeepFoolLinfinityAttack(foolbox_model, adversarial_criterion)
    elif p == np.Infinity:
        adversarial_attack = foolbox.attacks.DeepFoolL2Attack(foolbox_model, adversarial_criterion)
    #adversarial_attack = attacks.RandomDirectionAttack(100, 100, 1e-2, 1e-5, foolbox_model, adversarial_criterion)
    #adversarial_attack = FineTuningAttack(adversarial_attack, p)

    anti_adversarial_criterion = foolbox.criteria.Misclassification()
    #anti_adversarial_criterion = foolbox.criteria.CombinedCriteria(anti_adversarial_criterion, TargetTop2Difference(1e-5))
    if p == 2:
        adversarial_anti_attack = foolbox.attacks.DeepFoolL2Attack(foolbox_model, anti_adversarial_criterion)
    elif p == np.Infinity:
        adversarial_anti_attack = foolbox.attacks.DeepFoolLinfinityAttack(foolbox_model, anti_adversarial_criterion)
    #adversarial_anti_attack = FineTuningAttack(adversarial_attack, p)

    #basic_test(foolbox_model, test_loader, adversarial_attack, adversarial_anti_attack, p)
    #tests.image_test(foolbox_model, test_loader, adversarial_attack, adversarial_anti_attack)
    #direction_attack = attacks.RandomDirectionAttack(100, 100, 1e-2, 1e-5)

    direction_attack = attacks.RandomDirectionAttack(foolbox_model, foolbox.criteria.Misclassification(), p, 1000, 100, 0.05, 1e-7)

    batch_worker = batch_attack.PyTorchWorker(model)
    num_workers = 50

    adversarial_distance_tool = distance_tools.AdversarialDistance(type(adversarial_attack).__name__, foolbox_model, adversarial_attack, batch_worker, num_workers)
    direction_distance_tool = distance_tools.AdversarialDistance(type(direction_attack).__name__, foolbox_model, direction_attack)

    test_loader = loaders.TorchLoader(test_loader)
    adversarial_loader = loaders.AdversarialLoader(test_loader, foolbox_model, adversarial_attack, True, batch_worker, num_workers)
    random_noise_loader = loaders.RandomNoiseLoader(foolbox_model, 0, 1, [3, 32, 32], 10, 20)

    #tests.distance_comparison_test(foolbox_model, [adversarial_distance_tool, direction_distance_tool], p, adversarial_loader, num_workers)
    #tests.attack_test(foolbox_model, test_loader, adversarial_attack, p, batch_worker, num_workers)
    #model_tools.accuracy_test(foolbox_model, test_loader, set([1, 5]))

    train_loader = loaders.TorchLoader(train_loader)
    #training.train_torch(model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1), training.MaxEpoch(2), True)



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