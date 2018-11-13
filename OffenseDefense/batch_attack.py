import numpy as np
import foolbox
import queue
import torch
import batch_processing
import utils


class ModelBatchWorker(batch_processing.BatchWorker):
    def __init__(self, model):
        self.model = model
    def __call__(self, inputs):
        images = [x[0] for x in inputs]
        targets = [x[1] for x in inputs]

        images = torch.stack(images)
        outputs = self.model(images)

        ce = torch.nn.CrossEntropyLoss()
        
        losses = [ce(torch.unsqueeze(outputs[i], 0), targets[i]) for i in range(outputs.shape[0])] #torch.split(torch.sum(outputs, dim=1), 1)
        grads = torch.autograd.grad(losses, images)[0]
        #print(grads)

        outputs = torch.split(outputs, 1)
        grads = torch.split(grads, 1)

        return zip(outputs, grads)

class QueueAttackWorker(batch_processing.ThreadWorker):
    def __init__(self, attack, model, input_queue):
        self.attack = attack
        self.model = model
        self.input_queue = input_queue

    def __call__(self, pooler, return_queue):
        wrapped_model = ParallelPytorchModel(pooler, self.model._model, self.model.bounds(), self.model.num_classes(), self.model.channel_axis(), self.model.device, self.model._preprocessing)
        while True:
            try:
                i, (image, label) = self.input_queue.get(timeout=1e-2)
                adversarial = foolbox.Adversarial(wrapped_model, self.attack._default_criterion, image, label, self.attack._default_distance, self.attack._default_threshold)
                adversarial_image = self.attack(adversarial)
                #print('Inserting Adversarial')
                return_queue.put((i, adversarial_image))
            except queue.Empty:
                #print('Found empty')
                return

class ParallelPytorchModel(foolbox.models.PyTorchModel):
    def __init__(self, 
                 pooler, 
                 model,
                 bounds,
                 num_classes,
                 channel_axis=1,
                 device=None,
                 preprocessing=(0, 1)):
        super().__init__(model, bounds, num_classes, channel_axis, device, preprocessing)
        self.pooler = pooler

    def predictions_and_gradient(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).to(self.device)

        image = torch.from_numpy(image).to(self.device)

        if self._old_pytorch():  # pragma: no cover
            target = Variable(target)
            image = Variable(image, requires_grad=True)
        else:
            image.requires_grad_()

        predictions, grad = self.pooler.call((image, target))

        if self._old_pytorch():  # pragma: no cover
            predictions = predictions.data
        predictions = predictions.to("cpu")

        if not self._old_pytorch():
            predictions = predictions.detach()
        predictions = predictions.numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        if self._old_pytorch():  # pragma: no cover
            grad = grad.data
        grad = grad.to("cpu")
        if not self._old_pytorch():
            grad = grad.detach()
        grad = grad.numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad


def run_attack(foolbox_model, attack, images, labels, workers=50):
    assert len(images) == len(labels)
    model_worker = ModelBatchWorker(foolbox_model._model)

    input_queue = queue.Queue()
    data = zip(images, labels)

    attack_workers = [QueueAttackWorker(attack, foolbox_model, input_queue) for _ in range(workers)]

    results = batch_processing.run_queue_threads(model_worker, attack_workers, input_queue, data)
    
    assert len(results) == len(images)

    return results

def get_adversarials(foolbox_model, images, labels, adversarial_attack, workers=50, adversarial_approximation_check=None):
    f = utils.Filter()

    f['images'] = images
    f['ground_truth_labels'] = labels

    f['image_predictions'] = foolbox_model.batch_predictions(f['images'])
    f['image_labels'] = np.argmax(f['image_predictions'], axis=-1)
    correctly_classified = np.nonzero(np.equal(f['image_labels'], f['ground_truth_labels']))[0]
    print(correctly_classified)
    print(correctly_classified.shape)
    print(len(correctly_classified))
    f.filter(correctly_classified, 'successful_classification')

    f['adversarials'] = run_attack(foolbox_model, adversarial_attack, f['images'], f['image_labels'], workers=workers)
    successful_adversarials = [i for i in range(len(f['adversarials'])) if f['adversarials'][i] is not None]
    f.filter(successful_adversarials, 'successful_adversarial')
    #Convert to Numpy array after the failed samples have been removed
    f['adversarials'] = np.array(f['adversarials'])

    f['adversarial_predictions'] = foolbox_model.batch_predictions(f['adversarials'])
    f['adversarial_labels'] = np.argmax(f['adversarial_predictions'], axis=-1)

    if adversarial_approximation_check is not None:
        for x in f['adversarial_predictions']:
            assert utils.utils.top_k_difference(x, 2) > adversarial_approximation_check

    return f

def get_anti_adversarials(foolbox_model, images, labels, adversarial_attack, anti_attack, workers=50, adversarial_approximation_check=1e-6, anti_approximation_check=None):
    f = get_adversarials(foolbox_model, images, labels, adversarial_attack, workers, adversarial_approximation_check)

    print('Anti-Genuines')
    f['anti_genuines'] = run_attack(foolbox_model, anti_attack, f['images'], f['image_labels'], workers=workers)
    successful_anti_genuines = np.array([i for i in range(len(f['anti_genuines'])) if f['anti_genuines'][i] is not None], np.int32)

    print('Anti-Adversarials')
    f['anti_adversarials'] = run_attack(foolbox_model, anti_attack, f['adversarials'], f['adversarial_labels'], workers=workers)
    successful_anti_adversarials = np.array([i for i in range(len(f['anti_adversarials'])) if f['anti_adversarials'][i] is not None], np.int32)

    successful_intersection = np.intersect1d(successful_anti_genuines, successful_anti_adversarials)
    f.filter(successful_intersection, 'successful_intersection')

    #Convert to Numpy array after the failed samples have been removed
    f['anti_genuines'] = np.array(f['anti_genuines'])
    f['anti_adversarials'] = np.array(f['anti_adversarials'])

    f['anti_genuine_predictions'] = foolbox_model.batch_predictions(f['anti_genuines'])
    f['anti_genuine_labels'] = np.argmax(f['anti_genuine_predictions'], axis=-1)
    f['anti_adversarial_predictions'] = foolbox_model.batch_predictions(f['anti_adversarials'])
    f['anti_adversarial_labels'] = np.argmax(f['anti_adversarial_predictions'], axis=-1)

    #These asserts fail only when the samples are so close to the boundary that the
    #nondeterministic approximations by CUDA cause a difference in predictions (around 1e-6)
    #that causes the top class to be different. Since this doesn't concern us (we only want to
    #estimate the distance), we default to not checking
    if anti_approximation_check is not None:
        for x in f['anti_genuine_predictions']:
            assert utils.top_k_difference(x, 2) > adversarial_approximation_check
        for x in f['anti_adversarial_predictions']:
            assert utils.top_k_difference(x, 2) > adversarial_approximation_check

    return f, len(successful_anti_genuines), len(successful_anti_adversarials)