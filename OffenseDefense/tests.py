import warnings
from typing import List
import numpy as np
import torch
import foolbox
import matplotlib.pyplot as plt
import OffenseDefense.utils as utils
import OffenseDefense.batch_processing as batch_processing
import OffenseDefense.batch_attack as batch_attack
import OffenseDefense.loaders as loaders
import OffenseDefense.distance_tools as distance_tools

def basic_test(foolbox_model, loader, adversarial_attack, anti_attack, p, batch_worker=None, num_workers=50):
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

        filter, anti_genuine_count, anti_adversarial_count = batch_attack.get_anti_adversarials(foolbox_model, images, labels, adversarial_attack, anti_attack, batch_worker, num_workers)

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

def accuracy_test(foolbox_model : foolbox.models.Model,
                  loader : loaders.Loader,
                  top_ks : List[int],
                  verbose : bool = True):
    accuracies = [utils.AverageMeter() for _ in range(len(top_ks))]

    for images, labels in loader:
        batch_predictions = foolbox_model.batch_predictions(images)

        for i, top_k in enumerate(top_ks):
            correct_samples_count = utils.top_k_count(batch_predictions, labels, top_k)
            accuracies[i].update(1, correct_samples_count)
            accuracies[i].update(0, len(images) - correct_samples_count)

        if verbose:
            for top_k in sorted(top_ks):
                print('Top-{} Accuracy: {:2.2f}%'.format(top_k, accuracies[top_k].avg * 100.0))

            print('\n============\n')

    #Return accuracies instead of AverageMeters
    return [accuracy.avg for accuracy in accuracies]

def distance_comparison_test(foolbox_model : foolbox.models.Model,
                    test_distance_tools,
                    p : np.float,
                    loader : loaders.Loader,
                    verbose : bool = True):

    final_distances = {}
    success_rates = {}

    for distance_tool in test_distance_tools:
        final_distances[distance_tool.name] = []
        success_rates[distance_tool.name] = utils.AverageMeter()
    
    for images, labels in loader:
        #Remove misclassified samples
        correct_classification_filter = batch_attack.get_correct_samples(foolbox_model, images, labels)
        images = correct_classification_filter['images']
        labels = correct_classification_filter['image_labels']

        for distance_tool in test_distance_tools:
            distances = distance_tool.get_distances(images, labels, p)
            final_distances[distance_tool.name] += list(distances)

            success_rates[distance_tool.name].update(1, len(distances))
            success_rates[distance_tool.name].update(0, len(images) - len(distances))

            tool_distances = final_distances[distance_tool.name]
            success_rate = success_rates[distance_tool.name].avg

            average_distance = np.average(tool_distances)
            median_distance = np.median(tool_distances)

            #Treat failures as samples with distance=Infinity
            failure_count = success_rates[distance_tool.name].count - success_rates[distance_tool.name].sum
            adjusted_median_distance = np.median(tool_distances + [np.Infinity] * failure_count)
            
            if verbose:
                print('{}:'.format(distance_tool.name))
                print('Average Distance: {:2.2e}'.format(average_distance))
                print('Median Distance: {:2.2e}'.format(median_distance))
                print('Success Rate: {:2.2f}%'.format(success_rate * 100.0))
                print('Adjusted Median Distance: {:2.2e}'.format(adjusted_median_distance))

                print('\n============\n')

        if verbose:
            print('\n====================\n')

    #Replace AverageMeters with the final averages
    for key, value in success_rates.items():
        success_rates[key] = value.avg

    return final_distances, success_rates

def attack_test(foolbox_model : foolbox.models.Model,
                loader : loaders.Loader,
                attack : foolbox.attacks.Attack,
                p : int,
                batch_worker : batch_processing.BatchWorker = None,
                num_workers : int = 50,
                verbose : bool = True):

    success_rate = utils.AverageMeter()
    distances = []

    for images, labels in loader:
        adversarial_filter = batch_attack.get_adversarials(foolbox_model, images, labels, attack, batch_worker, num_workers)
        adversarials = adversarial_filter['adversarials']
        failure_count = len(images) - len(adversarials)

        success_rate.update(1, len(adversarials))
        success_rate.update(0, failure_count)

        distances += list(utils.lp_distance(adversarial_filter['adversarials'], adversarial_filter['images'], p, True))

        average_distance = np.average(distances)
        median_distance = np.median(distances)
        adjusted_median_distance = np.median(distances + [np.Infinity] * failure_count)
            
        if verbose:
            print('Average Distance: {:2.2e}'.format(average_distance))
            print('Median Distance: {:2.2e}'.format(median_distance))
            print('Success Rate: {:2.2f}%'.format(success_rate.avg * 100.0))
            print('Adjusted Median Distance: {:2.2e}'.format(adjusted_median_distance))

            print('\n============\n')

    return success_rate.avg, distances
        
