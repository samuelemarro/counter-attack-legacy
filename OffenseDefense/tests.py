import warnings
import numpy as np
import torch
import foolbox
import matplotlib.pyplot as plt
import utils
import batch_attack

def basic_test(foolbox_model, loader, adversarial_attack, anti_attack, p):
    average_anti_genuine = utils.AverageMeter()
    average_anti_adversarial = utils.AverageMeter()
    average_adversarial = utils.AverageMeter()

    classification_rate = utils.AverageMeter()
    adversarial_rate = utils.AverageMeter()
    anti_rate = utils.AverageMeter()

    warnings.filterwarnings('error', 'Not running')
    
    for data in loader:
        images, labels = data
        images = images.numpy()
        labels = labels.numpy()
        
        original_count = len(images)

        filter, anti_genuine_count, anti_adversarial_count = batch_attack.get_anti_adversarials(foolbox_model, images, labels, adversarial_attack, anti_attack)

        images = filter['images']
        labels = filter['image_labels']
        adversarials = filter['adversarials']
        anti_genuines = filter['anti_genuines']
        anti_adversarials = filter['anti_adversarials']
        classification_count = filter.filter_stats['successful_classification']
        adversarial_count = filter.filter_stats['successful_adversarial']

        print(filter.filter_stats)
        #print([np.average(x) for x in images])
        #print([np.average(x) for x in adversarials])
        #print([np.average(x) for x in anti_genuines])
        #print([np.average(x) for x in anti_adversarials])

        print(adversarials.shape[0])
        print(anti_adversarials.shape[0])
        print(anti_genuines.shape[0])

        classification_rate.update(1, classification_count)
        classification_rate.update(0, original_count - classification_count)
        adversarial_rate.update(1, adversarial_count)
        adversarial_rate.update(0, classification_count - adversarial_count)
        anti_rate.update(1, anti_genuine_count)
        anti_rate.update(1, anti_adversarial_count)
        anti_rate.update(0, adversarial_count - anti_genuine_count)
        anti_rate.update(0, adversarial_count - anti_adversarial_count)

        #Compute the distances
        adversarial_distances = utils.lp_distance(adversarials, images, p, True)
        anti_adversarial_distances = utils.lp_distance(adversarials, anti_adversarials, p, True)
        anti_genuine_distances = utils.lp_distance(images, anti_genuines, p, True)

        #print('Distances:')
        #print(adversarial_distances)

        average_adversarial.update(np.sum(adversarial_distances), adversarial_distances.shape[0])
        average_anti_adversarial.update(np.sum(anti_adversarial_distances), anti_adversarial_distances.shape[0])
        average_anti_genuine.update(np.sum(anti_genuine_distances), anti_genuine_distances.shape[0])

        print('Average Adversarial: {:2.2e}'.format(average_adversarial.avg))
        print('Average Anti Adversarial: {:2.2e}'.format(average_anti_adversarial.avg))
        print('Average Anti Genuine: {:2.2e}'.format(average_anti_genuine.avg))

def approximation_test(foolbox_model, loader, adversarial_anti_attack, distance_calculator, p, adversarial_attack=None):

    adversarial_distances = []
    direction_distances = []
    
    for data in loader:
        images, labels = data
        images = images.numpy()
        labels = labels.numpy()

        #If requested, test using adversarial samples (which are close to the boundary)
        if adversarial_attack is not None:
            adversarial_filter = batch_attack.get_adversarials(foolbox_model, images, labels, adversarial_attack)
            images = adversarial_filter['adversarials']
            labels = adversarial_filter['adversarial_labels']

        anti_adversarial_filter = batch_attack.get_adversarials(foolbox_model, images, labels, adversarial_anti_attack)

        closest_samples = []
        for image, label in zip(anti_adversarial_filter['images'], anti_adversarial_filter['image_labels']):
            closest_samples.append(distance_calculator(image, label))
        anti_adversarial_filter['closest_samples'] = closest_samples
        successful_closest_samples = np.array([i for i in range(len(closest_samples)) if closest_samples[i] is not None], dtype=int)

        anti_adversarial_filter.filter(successful_closest_samples, 'successful_closest_samples')
        anti_adversarial_filter['closest_samples'] = np.array(anti_adversarial_filter['closest_samples'])
        closest_samples = anti_adversarial_filter['closest_samples']

        images = anti_adversarial_filter['images']
        labels = anti_adversarial_filter['image_labels']
        adversarials = anti_adversarial_filter['adversarials']

        #Compute the distances
        adversarial_distances += list(utils.lp_distance(adversarials, images, p, True))
        direction_distances += list(utils.lp_distance(closest_samples, images, p, True))

        ratios = np.array([adversarial / direction for adversarial, direction in zip(adversarial_distances, direction_distances)])

        #visualisation.plot_histogram(np.array(adversarial_distances), 'blue')
        #visualisation.plot_histogram(np.array(direction_distances), 'red')

        average_adversarial_distance = np.average(np.array(adversarial_distances))
        average_direction_distance = np.average(np.array(direction_distances))
        average_ratio = np.average(ratios)

        median_adversarial_distance = np.median(np.array(adversarial_distances))
        median_direction_distance = np.median(np.array(direction_distances))
        median_ratio = np.median(ratios)

        print('Average Adversarial: {:2.2e}'.format(average_adversarial_distance))
        print('Average Direction: {:2.2e}'.format(average_direction_distance))
        print('Average Ratio: {:2.2e}'.format(average_ratio))
        print()
        print('Median Adversarial: {:2.2e}'.format(median_adversarial_distance))
        print('Median Direction: {:2.2e}'.format(median_direction_distance))
        print('Median Ratio: {:2.2e}'.format(median_ratio))