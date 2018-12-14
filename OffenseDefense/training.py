import logging

import numpy as np
import torch
from . import utils

logger = logging.getLogger(__name__)


class StopCriterion:
    # Note: epoch is 1-indexed because 0 is the pre-epoch
    def proceed(self, epoch, loss, accuracy):
        pass


class MaxEpoch(StopCriterion):
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def proceed(self, epoch, loss, accuracy):
        return epoch <= self.max_epoch


def train_torch(model, loader, loss_fn, optimizer, stop_criterion, use_cuda):
    model.train()

    epoch = 0
    proceed = True

    while proceed:
        top1_accuracy = utils.AverageMeter()
        top5_accuracy = utils.AverageMeter()
        average_loss = utils.AverageMeter()

        for images, labels in loader:
            torch_images = torch.from_numpy(images)
            # Torch uses long ints for the labels
            torch_labels = torch.from_numpy(labels).long()

            torch_images.requires_grad_()

            if use_cuda:
                torch_images = torch_images.cuda()
                torch_labels = torch_labels.cuda(async=True)

            # Compute the outputs
            outputs = model(torch_images)
            loss = loss_fn(outputs, torch_images, torch_labels)

            # Compute the statistics
            outputs = outputs.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            top1_count = utils.top_k_count(outputs, labels, k=1)
            top5_count = utils.top_k_count(outputs, labels, k=5)

            average_loss.update(loss, len(images))

            top1_accuracy.update(1, top1_count)
            top1_accuracy.update(0, len(images) - top1_count)
            top5_accuracy.update(1, top5_count)
            top5_accuracy.update(0, len(images) - top5_count)

            # Update the parameters.
            # Since the stop_criterion might decide that we do not
            # need any training at all, in the first epoch we do not
            # perform any updates and only compute the statistics
            if epoch != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch += 1

        logger.info('\n=========')
        if epoch == 0:
            logger.info('Pre-Epoch (Epoch 0)')
        else:
            logger.info('Epoch {}'.format(epoch))
        logger.info('=========\n')
        logger.info('Average Loss: {:2.2e}'.format(average_loss.avg))
        logger.info(
            'Top-1 Accuracy: {:2.2f}%'.format(top1_accuracy.avg * 100.0))
        logger.info(
            'Top-5 Accuracy: {:2.2f}%'.format(top5_accuracy.avg * 100.0))

        proceed = stop_criterion.proceed(
            epoch, average_loss.avg, top1_accuracy.avg)


def train_detector(model, detector, failure_value, loader, optimizer, stop_criterion, use_cuda, verbose=True):
    # TODO: Should it precompute the distances?

    def loss_fn(outputs, images, labels):
        scores = detector.get_scores(images.cpu().numpy())

        # Remove failed scores
        scores = [score for score in scores if score is not failure_value]
        scores = np.array(scores)
        scores = torch.from_numpy(scores)

        loss = torch.nn.MSELoss()(outputs, scores)
        return loss

    train_torch(model, loader, loss_fn, optimizer,
                stop_criterion, use_cuda, verbose)
