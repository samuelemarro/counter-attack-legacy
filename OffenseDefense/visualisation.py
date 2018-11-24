import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors.kde import KernelDensity

from . import utils

def show_image(image, channels_first=True):
    #imshow requires images in format (W, H, C)
    if channels_first:
        image = np.moveaxis(image, 0, -1)

    plt.imshow(image)
    plt.show()

def plot_histogram(values, color, log_x=False):
    sorted_values = np.sort(values)
    if log_x:
        sorted_values = np.log10(sorted_values)
    #fit_values = stats.norm.pdf(sorted_values, np.mean(sorted_values), np.std(sorted_values))  #this is a fitting indeed
    #plt.xscale('log')
    #plt.plot(sorted_values, sorted_values, color=color)


    if sorted_values.shape[0] > 1:
        #print(sorted_values.shape)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sorted_values[:, np.newaxis].reshape(-1, 1))
        kde_x = np.arange(np.min(sorted_values), np.max(sorted_values), 0.1)
        kde_y = kde.score_samples(kde_x[:, np.newaxis].reshape(-1, 1))
        kde_y = np.exp(kde_y)

        plt.plot(kde_x, kde_y, color=color, alpha=0.25)
    
    plt.scatter(sorted_values, np.zeros_like(sorted_values), color=color, alpha=0.5)


def plot_roc(genuine_values, adversarial_values):
    false_positive_rates, true_positive_rates, thresholds = utils.roc_curve(adversarial_values, genuine_values)

    best_threshold, best_tpr, best_fpr = utils.get_best_threshold(true_positive_rates, false_positive_rates, thresholds)

    print('Best Threshold: {}\nBest TPR: {}\nBest FPR: {}'.format(best_threshold, best_tpr, best_fpr))
    area_under_curve = metrics.auc(false_positive_rates, true_positive_rates)
    
    lw = 2
    plt.plot(false_positive_rates, true_positive_rates, color='darkorange', lw=lw, label='ROC curve (area = {:.2f}%)'.format(area_under_curve * 100.0))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([best_fpr, best_fpr], [0, best_tpr], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (Best = {:2.2f}%)'.format(best_fpr * 100.0))
    plt.ylabel('True Positive Rate (Best = {:2.2f}%)'.format(best_tpr * 100.0))
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
