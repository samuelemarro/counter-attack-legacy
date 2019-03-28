import logging
from progress.bar import IncrementalBar

import numpy as np
import torch
from . import utils

logger = logging.getLogger(__name__)


def train_torch(model, loader, loss_fn, optimizer, epochs, use_cuda, classification=True, name='Train'):
    model.train()

    epoch = 0

    top1_accuracy = None
    top5_accuracy = None

    if classification:
        top1_accuracy = utils.AverageMeter()
        top5_accuracy = utils.AverageMeter()

    average_loss = utils.AverageMeter()

    bar = None

    if logger.getEffectiveLevel() == logging.INFO:
        bar = IncrementalBar(name, max=epochs*len(loader))

    for i in range(epochs):
        if bar is not None:
            bar.suffix = 'Epoch {}/{}'.format(i + 1, epochs)

        for images, targets in loader:
            torch_images = torch.from_numpy(images)

            torch_targets = torch.from_numpy(targets)
            if classification:
                # Torch uses long ints for the labels
                torch_targets = torch_targets.long()

            torch_images.requires_grad_()

            if use_cuda:
                torch_images = torch_images.cuda()
                torch_targets = torch_targets.cuda()

            # Compute the outputs
            outputs = model(torch_images)
            loss = loss_fn(outputs, torch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the statistics
            outputs = outputs.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()
            average_loss.update(loss, len(images))

            if classification:
                top1_count = utils.top_k_count(outputs, targets, k=1)
                top5_count = utils.top_k_count(outputs, targets, k=5)

                top1_accuracy.update(1, top1_count)
                top1_accuracy.update(0, len(images) - top1_count)
                top5_accuracy.update(1, top5_count)
                top5_accuracy.update(0, len(images) - top5_count)

            if bar is not None:
                bar.next()

        logger.debug('\n=========')
        logger.debug('Epoch {}'.format(epoch))
        logger.debug('=========\n')
        logger.debug('Average Loss: {:2.2e}'.format(average_loss.avg))

        if classification:
            logger.debug(
                'Top-1 Accuracy: {:2.2f}%'.format(top1_accuracy.avg * 100.0))
            logger.debug(
                'Top-5 Accuracy: {:2.2f}%'.format(top5_accuracy.avg * 100.0))
    if bar is not None:
        bar.finish()

    if classification:
        top1_accuracy = top1_accuracy.avg
        top5_accuracy = top5_accuracy.avg

    return average_loss.avg, top1_accuracy, top5_accuracy
