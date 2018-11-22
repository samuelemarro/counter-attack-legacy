import queue
import numpy as np
import foolbox
import OffenseDefense.batch_processing as batch_processing
import OffenseDefense.utils as utils
import torch

class PyTorchWorker(batch_processing.BatchWorker):
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        images = [np.array(x[0]) for x in inputs]
        labels = [np.array(x[1]) for x in inputs]

        images = torch.from_numpy(np.array(images))
        labels = torch.from_numpy(np.array(labels))

        images.requires_grad_()

        #Convert to CUDA tensors, if available
        if next(self.model.parameters()).is_cuda:
            images = images.cuda()
            labels = labels.cuda()

        outputs = self.model(images)

        cross_entropy = torch.nn.CrossEntropyLoss()

        losses = [cross_entropy(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(labels[i], 0)) for i in range(outputs.shape[0])]
        grads = torch.autograd.grad(losses, images)[0]

        outputs = [output.cpu().detach().numpy()[0] for output in torch.split(outputs, 1)]
        grads = [grad.cpu().numpy()[0] for grad in torch.split(grads, 1)]

        return zip(outputs, grads)

class QueueAttackWorker(batch_processing.ThreadWorker):
    def __init__(self, attack, model, input_queue):
        self.attack = attack
        self.model = model
        self.input_queue = input_queue

    def __call__(self, pooler, return_queue):
        parallel_model = ParallelModel(pooler,
                                       self.model,
                                       self.model.bounds(),
                                       self.model.channel_axis(),
                                       self.model._preprocessing)
        while True:
            try:
                i, (image, label) = self.input_queue.get(timeout=1e-2)
                adversarial = foolbox.Adversarial(parallel_model,
                                                  self.attack._default_criterion,
                                                  image,
                                                  label,
                                                  self.attack._default_distance,
                                                  self.attack._default_threshold)
                adversarial_image = self.attack(adversarial)

                return_queue.put((i, adversarial_image))
            except queue.Empty:
                #No more inputs, we can stop
                return

class ParallelModel(foolbox.models.DifferentiableModel):
    def __init__(self, pooler, foolbox_model, bounds, channel_axis, preprocessing = (0,1)):
        super().__init__(bounds, channel_axis, preprocessing)
        self.pooler = pooler
        self.foolbox_model = foolbox_model

    def predictions_and_gradient(self, image, label):
        return self.pooler.call((image, label))

    def backward(self, gradient, image):
        raise NotImplementedError()

    def batch_predictions(self, images):
        return self.foolbox_model.batch_predictions(images)

    def num_classes(self):
        return self.foolbox_model.num_classes()


def run_batch_attack(foolbox_model, batch_worker, attack, images, labels, num_workers=50):
    assert len(images) == len(labels)

    input_queue = queue.Queue()
    data = zip(images, labels)

    attack_workers = [QueueAttackWorker(attack, foolbox_model, input_queue) for _ in range(num_workers)]

    results = batch_processing.run_queue_threads(batch_worker, attack_workers, input_queue, data)
    
    assert len(results) == len(images)

    return results

def run_individual_attack(attack, images, labels):
    assert len(images) == len(labels)
    
    adversarials = []

    for image, label in zip(images, labels):
        adversarial = attack(image, label)
        adversarials.append(adversarial)

    return adversarials

def get_correct_samples(foolbox_model : foolbox.models.PyTorchModel,
                        images : np.ndarray,
                        labels : np.ndarray):
    filter = utils.Filter()
    filter['images'] = images
    filter['ground_truth_labels'] = labels

    filter['image_predictions'] = foolbox_model.batch_predictions(filter['images'])
    filter['image_labels'] = np.argmax(filter['image_predictions'], axis=-1)
    correctly_classified = np.nonzero(np.equal(filter['image_labels'],
                                               filter['ground_truth_labels']))[0]
    
    filter.filter(correctly_classified, 'successful_classification')

    return filter


def get_adversarials(foolbox_model : foolbox.models.PyTorchModel,
                     images : np.ndarray,
                     labels : np.ndarray,
                     adversarial_attack : foolbox.attacks.Attack,
                     batch_worker : batch_processing.BatchWorker = None,
                     num_workers: int = 50,
                     adversarial_approximation_threshold: float = None):
    filter = get_correct_samples(foolbox_model, images, labels)

    if batch_worker is not None:
        filter['adversarials'] = run_batch_attack(foolbox_model,
                                                  batch_worker,
                                                  adversarial_attack,
                                                  filter['images'],
                                                  filter['image_labels'],
                                                  num_workers=num_workers)
    else:
        filter['adversarials'] = run_individual_attack(adversarial_attack, filter['images'], filter['image_labels'])
    successful_adversarials = [i for i in range(len(filter['adversarials'])) if filter['adversarials'][i] is not None]
    filter.filter(successful_adversarials, 'successful_adversarial')

    #Convert to Numpy array after the failed samples have been removed
    filter['adversarials'] = np.array(filter['adversarials'])
    filter['adversarial_predictions'] = foolbox_model.batch_predictions(filter['adversarials'])
    filter['adversarial_labels'] = np.argmax(filter['adversarial_predictions'], axis=-1)

    if adversarial_approximation_threshold is not None:
        for x in filter['adversarial_predictions']:
            assert utils.top_k_difference(x, 2) > adversarial_approximation_threshold

    return filter

def get_anti_adversarials(foolbox_model : foolbox.models.PyTorchModel,
                          images : np.ndarray,
                          labels : np.ndarray,
                          adversarial_attack : foolbox.attacks.Attack,
                          anti_attack : foolbox.attacks.Attack,
                          batch_worker : batch_processing.BatchWorker = None,
                          num_workers: int = 50,
                          adversarial_approximation_threshold : float = 1e-6,
                          anti_adversarial_approximation_threshold : float = None):

    filter = get_adversarials(foolbox_model,
                              images,
                              labels,
                              adversarial_attack,
                              batch_worker,
                              num_workers,
                              adversarial_approximation_threshold)

    print('Anti-Genuines')
    if batch_worker is not None:
        filter['anti_genuines'] = run_batch_attack(foolbox_model,
                                                   batch_worker,
                                                   anti_attack,
                                                   filter['images'],
                                                   filter['image_labels'],
                                                   num_workers=num_workers)
    else:
        filter['anti_genuines'] = run_individual_attack(anti_attack, filter['images'], filter['image_labels'])
    successful_anti_genuines = np.array([i for i in range(len(filter['anti_genuines'])) if filter['anti_genuines'][i] is not None], np.int32)

    print('Anti-Adversarials')
    if batch_worker is not None:
        filter['anti_adversarials'] = run_batch_attack(foolbox_model,
                                                       batch_worker,
                                                       anti_attack,
                                                       filter['adversarials'],
                                                       filter['adversarial_labels'],
                                                       num_workers=num_workers)
    else:
        filter['anti_adversarials'] = run_individual_attack(anti_attack, filter['adversarials'], filter['adversarial_labels'])

    successful_anti_adversarials = np.array([i for i in range(len(filter['anti_adversarials'])) if filter['anti_adversarials'][i] is not None], np.int32)

    successful_intersection = np.intersect1d(successful_anti_genuines, successful_anti_adversarials)
    filter.filter(successful_intersection, 'successful_intersection')

    #Convert to Numpy array after the failed samples have been removed
    filter['anti_genuines'] = np.array(filter['anti_genuines'])
    filter['anti_adversarials'] = np.array(filter['anti_adversarials'])

    filter['anti_genuine_predictions'] = foolbox_model.batch_predictions(filter['anti_genuines'])
    filter['anti_genuine_labels'] = np.argmax(filter['anti_genuine_predictions'], axis=-1)
    filter['anti_adversarial_predictions'] = foolbox_model.batch_predictions(filter['anti_adversarials'])
    filter['anti_adversarial_labels'] = np.argmax(filter['anti_adversarial_predictions'], axis=-1)

    #These asserts fail only when the samples are so close to the boundary that the
    #nondeterministic approximations by CUDA cause a difference in predictions (around 1e-6)
    #that causes the top class to be different. Since this doesn't concern us (we only want to
    #estimate the distance), we default to not checking
    if anti_adversarial_approximation_threshold is not None:
        for x in filter['anti_genuine_predictions']:
            assert utils.top_k_difference(x, 2) > adversarial_approximation_threshold
        for x in filter['anti_adversarial_predictions']:
            assert utils.top_k_difference(x, 2) > adversarial_approximation_threshold

    return filter, len(successful_anti_genuines), len(successful_anti_adversarials)