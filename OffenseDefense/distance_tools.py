import batch_attack
import utils

class DistanceTool:
    def __init__(self, name):
        self.name = name
    def get_distance(self, image, label, p):
        raise NotImplementedError()
    def get_distances(self, images, labels, p):
        raise NotImplementedError()

class AdversarialDistance(DistanceTool):
    def __init__(self, name, foolbox_model, attack, parallelize):
        self.name = name
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.parallelize = parallelize

    def get_distance(self, image, label, p):
        adversarial = self.attack(image, label)
        distance = utils.lp_distance(adversarial, image, p, False)
        return distance

    def get_distances(self, images, labels, p):
        adversarial_filter = batch_attack.get_adversarials(self.foolbox_model, images, labels, self.attack, self.parallelize)

        #We use adversarial_filter['images'] instead of images because the filter remove the genuine samples
        #which could not be attacked
        distances = utils.lp_distance(adversarial_filter['adversarials'], adversarial_filter['images'], p, True)
        return distances
