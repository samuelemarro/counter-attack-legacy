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

"""
Returns the unnormalized L_p distance.

If you're wondering what's going on: foolbox attacks accept
a distance type instead of an actual object. So, if you want
to use the MSE distance, you have to pass foolbox.distances.MSE.
foolbox then calls the type and builds the distance (e.g. MSE(...)).
This usually works well, but it means that we can't pass any other
arguments to the distance. We therefore use this wrapper trick to 
pass the argument 'p': we init and pass the LpDistance object.
foolbox will attempt to create the distance by calling distance(...)
However, since it's an instance, calls to the class are handled by
__call__. In __call__, we init WrappedLpDistance with the provided
arguments (in addition to p) and return it.
"""
class LpDistance(foolbox.distances.Distance):
    class WrappedLpDistance(foolbox.distances.Distance):
        def __init__(self, p, reference = None, other = None, bounds = None, value = None):
            self.p = p
            super().__init__(reference, other, bounds, value)

        def _calculate(self):
            value = utils.lp_distance(self.other, self.reference, self.p, False)
            gradient = None
            return value, gradient

        @property
        def gradient(self):
            raise NotImplementedError
        def name(self):
            return 'L{} Distance'.format(self.p)

    def __init__(self, p):
        self.p = p
    def __call__(self,
                 reference=None,
                 other=None,
                 bounds=None,
                 value=None):
        return LpDistance.WrappedLpDistance(self.p, reference, other, bounds, value)
    def _calculate(self):
        raise NotImplementedError()
