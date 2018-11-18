import foolbox
import OffenseDefense.batch_processing as batch_processing
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.utils as utils

class DistanceTool:
    def __init__(self, name):
        self.name = name
    def get_distance(self, image, label, p):
        raise NotImplementedError()
    def get_distances(self, images, labels, p):
        raise NotImplementedError()

class AdversarialDistance(DistanceTool):
    def __init__(self,
                 name : str,
                 foolbox_model : foolbox.models.Model,
                 attack : foolbox.attacks.Attack,
                 batch_worker : batch_processing.BatchWorker = None,
                 num_workers : int = 50):
        self.name = name
        self.foolbox_model = foolbox_model
        self.attack = attack
        self.batch_worker = batch_worker
        self.num_workers = num_workers

    def get_distance(self, image, label, p):
        adversarial = self.attack(image, label)
        distance = utils.lp_distance(adversarial, image, p, False)
        return distance

    def get_distances(self, images, labels, p):
        adversarial_filter = batch_attack.get_adversarials(self.foolbox_model,
                                                           images,
                                                           labels,
                                                           self.attack,
                                                           self.batch_worker,
                                                           self.num_workers)

        #We use adversarial_filter['images'] instead of images because the filter remove the genuine samples
        #which could not be attacked
        distances = utils.lp_distance(adversarial_filter['adversarials'], adversarial_filter['images'], p, True)
        return distances
